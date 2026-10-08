"""
views/evaluation_skript.py — Evaluation mit evaluate_petra.py (Stand v4).

Dieselbe Messung wie in der Arbeit (RAGAS mit lokalem Judge, deterministische
Kennzahlen, Zielwerte), bedienbar aus Streamlit:

  1. Lauf starten: alle 75 Fälle, eine Auswahl einzelner Fälle oder
     "schwache Fälle" eines früheren Laufs (Teiltest).
  2. Ergebnisse: Kennzahlen gegen Zielwerte, Tabelle je Fall mit Filter
     "nur schwache Fälle", Antwort/Referenz/Kontext eines Falls.
  3. Vergleich: zwei Läufe auf ihren GEMEINSAMEN Fällen, Veränderung je Fall.
  4. Download jeder Ergebnis-Excel (mit Blatt "Diagramm").

Läufe laufen als Hintergrundprozess weiter, auch wenn die Seite geschlossen wird.
Ergebnisse liegen in PETRA_EVAL_DIR (Standard /data/eval); Läufe des
Skripts run_eval_schritt.sh (/data/eval_*.xlsx) werden ebenfalls angezeigt.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

EVAL_DIR = Path(os.getenv("PETRA_EVAL_DIR", "/data/eval"))
WEITERE_DIRS = [Path("/data")]
SKRIPT = EVAL_DIR / "evaluate_petra.py"
TESTSET = EVAL_DIR / "testset_v2.xlsx"
BASELINE = EVAL_DIR / "ragas_evaluation_20260923_213553.xlsx"
PID_DATEI = EVAL_DIR / "laufender_lauf.pid"
N8N = os.getenv("PETRA_N8N_URL", "http://n8n:5678").rstrip("/")
WEBHOOK = os.getenv("PETRA_N8N_QUERY_WEBHOOK_URL", f"{N8N}/webhook/petra-query")
API = os.getenv("PETRA_API_URL", "http://ingestion-api:8001").rstrip("/")
OLLAMA = os.getenv("PETRA_OLLAMA_URL", "http://ollama:11434").rstrip("/")
JUDGES = {"qwen2.5:32b (wie in der Arbeit)": "ollama:qwen2.5:32b",
          "ohne RAGAS (nur Antworten + deterministische Kennzahlen)": None}
METRIKEN = [("ragas_faithfulness", "Faithfulness", 0.78, 0.85),
            ("ragas_context_precision", "Context Precision", 0.74, 0.80),
            ("ragas_answer_relevancy", "Answer Relevancy", None, 0.80),
            ("ragas_context_recall", "Context Recall", None, None)]
SPALTEN = ["nr", "query_class", "ragas_faithfulness", "ragas_context_precision", "ragas_answer_relevancy",
           "ragas_context_recall", "doc_hit", "wert_recall", "ablehnung_ok", "latency_ms", "frage"]


# ── Hilfsfunktionen ──────────────────────────────────────────────────
def _laufender_prozess() -> tuple[int | None, str | None]:
    try:
        pid_s, name = PID_DATEI.read_text(encoding="utf-8").split("\n", 1)
        pid = int(pid_s)
    except (OSError, ValueError):
        return None, None
    try:   # beendeten Kindprozess einsammeln (sonst Zombie = "läuft")
        if os.waitpid(pid, os.WNOHANG)[0] == pid:
            return None, None
    except ChildProcessError:
        pass
    try:
        os.kill(pid, 0)
        stat = Path(f"/proc/{pid}/stat").read_text()
        if stat.rsplit(")", 1)[1].split()[0] == "Z":
            return None, None
        return pid, name.strip()
    except (OSError, IndexError):
        return None, None


def _log_ende(pfad: Path, zeilen: int = 25) -> str:
    try:
        text = pfad.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    except OSError:
        return "(noch kein Protokoll)"
    return "\n".join([z for z in text.splitlines() if z.strip()][-zeilen:])


def _ergebnisdateien() -> list[Path]:
    dateien = list(EVAL_DIR.glob("*.xlsx")) + [p for d in WEITERE_DIRS for p in d.glob("eval_*.xlsx")]
    dateien = [p for p in dateien if p.name not in (TESTSET.name, BASELINE.name) and not p.name.startswith("testset")]
    dateien = [p for p in set(dateien) if _ergebnis(p) is not None]   # nur Ergebnisse von evaluate_petra.py
    return sorted(dateien, key=lambda p: p.stat().st_mtime, reverse=True)


@st.cache_data(show_spinner=False)
def _lade(pfad: str, mtime: float) -> pd.DataFrame | None:
    """Blatt "Ergebnisse" von evaluate_petra.py; andere Excel-Dateien -> None."""
    try:
        df = pd.read_excel(pfad, sheet_name="Ergebnisse")
        df["nr"] = pd.to_numeric(df["nr"], errors="coerce")
        return df.dropna(subset=["nr"]).astype({"nr": int})
    except Exception:
        return None


def _ergebnis(p: Path) -> pd.DataFrame | None:
    return _lade(str(p), p.stat().st_mtime)


def _num(df: pd.DataFrame, spalte: str) -> pd.Series:
    if spalte in df:
        return pd.to_numeric(df[spalte], errors="coerce")
    return pd.Series(float("nan"), index=df.index)


@st.cache_data(show_spinner=False)
def _testfaelle(mtime: float) -> pd.DataFrame:
    t = pd.read_excel(TESTSET)
    t["nr"] = t["nr"].astype(int)
    return t


def _ist_negativ(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "ja", "wahr")


def _schwach(df: pd.DataFrame, f_grenze: float, cp_grenze: float) -> pd.DataFrame:
    """Schwach = F oder CP unter Grenze, Datenblatt fehlt oder Negativfall nicht abgelehnt."""
    neg = df["nicht_beantwortbar"].map(_ist_negativ) if "nicht_beantwortbar" in df \
        else pd.Series(False, index=df.index)
    maske = (~neg & ((_num(df, "ragas_faithfulness") < f_grenze) | (_num(df, "ragas_context_precision") < cp_grenze)
                     | (_num(df, "doc_hit") == 0))) | (neg & (_num(df, "ablehnung_ok") == 0))
    return df[maske.fillna(False).astype(bool)]


def _start(name: str, judge: str | None, faelle: list[int] | None) -> None:
    befehl = [sys.executable, str(SKRIPT), "--testset", str(TESTSET), "--webhook", WEBHOOK,
              "--api", API, "--ollama", OLLAMA, "--out", str(EVAL_DIR / f"{name}.xlsx"),
              "--kontext-ansicht", "bloecke"]
    if BASELINE.exists():
        befehl += ["--baseline", str(BASELINE)]
    befehl += ["--judge", judge] if judge else ["--skip-ragas"]
    if faelle:
        befehl += ["--faelle", ",".join(str(n) for n in sorted(faelle))]
    log = open(EVAL_DIR / f"{name}.log", "w", encoding="utf-8")
    proc = subprocess.Popen(befehl, cwd=EVAL_DIR, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    PID_DATEI.write_text(f"{proc.pid}\n{name}", encoding="utf-8")


def _fmt(df: pd.DataFrame) -> pd.DataFrame:
    zeig = [c for c in SPALTEN if c in df.columns]
    out = df[zeig].copy()
    if "latency_ms" in out:
        out["latency_ms"] = (pd.to_numeric(out["latency_ms"], errors="coerce") / 1000).round(1)
    return out.rename(columns={"query_class": "Klasse", "ragas_faithfulness": "F", "ragas_context_precision": "CP",
                               "ragas_answer_relevancy": "AR", "ragas_context_recall": "CR",
                               "latency_ms": "Latenz s", "frage": "Frage"})


# ── Bereiche der Seite ───────────────────────────────────────────────
def _bereich_start() -> None:
    pid, lauf = _laufender_prozess()
    if pid:
        st.info(f"Lauf **{lauf}** läuft (Prozess {pid}). Die Seite kann geschlossen werden.")
        st.code(_log_ende(EVAL_DIR / f"{lauf}.log"), language=None)
        c1, c2 = st.columns(2)
        if c1.button("🔄 Fortschritt aktualisieren", use_container_width=True):
            st.rerun()
        if c2.button("⏹ Lauf abbrechen", use_container_width=True):
            try:
                os.killpg(pid, signal.SIGTERM)
            except OSError:
                pass
            time.sleep(1)
            st.rerun()
        return

    t = _testfaelle(TESTSET.stat().st_mtime)
    with st.form("eval_start"):
        name = st.text_input("Name des Laufs", value=time.strftime("eval_%Y%m%d_%H%M"))
        judge_label = st.selectbox("Judge", list(JUDGES))
        umfang = st.radio("Fälle", ["Alle 75 Fälle", "Schwache Fälle eines früheren Laufs", "Einzelne Fälle auswählen"])
        laeufe = _ergebnisdateien()
        quelle = st.selectbox("Früherer Lauf (für „schwache Fälle“)", [p.name for p in laeufe] or ["–"])
        c1, c2 = st.columns(2)
        f_grenze = c1.number_input("schwach: Faithfulness unter", 0.0, 1.0, 0.6, 0.05)
        cp_grenze = c2.number_input("schwach: Context Precision unter", 0.0, 1.0, 0.5, 0.05)
        kontrolle = st.number_input("zusätzlich gute Kontrollfälle (0 = keine)", 0, 30, 12)
        auswahl = st.multiselect("Einzelne Fälle", t["nr"].tolist(),
                                 format_func=lambda n: f"{n} – {str(t.set_index('nr').loc[n, 'frage'])[:70]}")
        st.caption("Der Judge qwen2.5:32b braucht etwa 2–3 Minuten je Fall. Während des Laufs keine Container neu starten.")
        if st.form_submit_button("▶ Evaluation starten", use_container_width=True):
            faelle = None
            if umfang.startswith("Einzelne"):
                faelle = list(auswahl)
            elif umfang.startswith("Schwache"):
                p = next((x for x in laeufe if x.name == quelle), None)
                if p is None:
                    st.error("Kein früherer Lauf gewählt."); return
                alt = _ergebnis(p)
                schwach = _schwach(alt, f_grenze, cp_grenze)
                faelle = schwach["nr"].tolist()
                if kontrolle:
                    gut = alt[~alt["nr"].isin(faelle)]
                    gut = gut[~gut["nicht_beantwortbar"].map(_ist_negativ)] if "nicht_beantwortbar" in gut else gut
                    if "ragas_faithfulness" in gut:
                        gut = gut.sort_values("ragas_faithfulness", ascending=False)
                    faelle += gut["nr"].head(int(kontrolle)).tolist()
            if faelle is not None and not faelle:
                st.error("Keine Fälle ausgewählt."); return
            sauber = "".join(ch for ch in name if ch.isalnum() or ch in "_-") or "eval_streamlit"
            _start(sauber, JUDGES[judge_label], faelle)
            st.success(f"Lauf {sauber} gestartet ({len(faelle) if faelle else 75} Fälle).")
            time.sleep(1)
            st.rerun()


def _bereich_ergebnisse() -> None:
    laeufe = _ergebnisdateien()
    if not laeufe:
        st.caption("Noch keine Ergebnisse."); return
    wahl = st.selectbox("Lauf", [p.name for p in laeufe], key="erg_lauf")
    p = next(x for x in laeufe if x.name == wahl)
    df = _ergebnis(p)
    beantw = df[~df["nicht_beantwortbar"].map(_ist_negativ)] if "nicht_beantwortbar" in df else df
    cols = st.columns(5)
    for c, (spalte, label, z1, z2) in zip(cols, METRIKEN):
        if spalte in beantw:
            v = pd.to_numeric(beantw[spalte], errors="coerce").mean()
            ziel = z2 or z1
            c.metric(label, "–" if pd.isna(v) else f"{v:.3f}",
                     None if (ziel is None or pd.isna(v)) else f"{v - ziel:+.3f} zum Ziel {ziel}")
    if "latency_ms" in df:
        cols[4].metric("Latenz Ø", f"{pd.to_numeric(df['latency_ms'], errors='coerce').mean() / 1000:.1f} s", "Ziel 5 s",
                       delta_color="off")
    werte = {label: pd.to_numeric(beantw[s], errors="coerce").mean() for s, label, _, _ in METRIKEN if s in beantw}
    if werte and not all(pd.isna(v) for v in werte.values()):
        st.bar_chart(pd.DataFrame({"Wert (0 bis 1)": werte}))
    nur_schwach = st.checkbox("Nur schwache Fälle zeigen (F < 0,6, CP < 0,5, Datenblatt fehlt, Ablehnung falsch)")
    zeig = _schwach(df, 0.6, 0.5) if nur_schwach else df
    st.caption(f"{len(zeig)} von {len(df)} Fällen")
    st.dataframe(_fmt(zeig), hide_index=True, use_container_width=True)
    fall = st.selectbox("Fall im Detail", zeig["nr"].tolist(), key="erg_fall")
    if fall is not None:
        r = df[df["nr"] == fall].iloc[0]
        st.markdown(f"**Frage:** {r.get('frage', '')}")
        c1, c2 = st.columns(2)
        c1.markdown("**Antwort des Systems**"); c1.write(r.get("generierte_antwort", ""))
        c2.markdown("**Referenzantwort**"); c2.write(r.get("referenzantwort", ""))
        with st.expander("Quellen und Kontext"):
            st.write(r.get("quellen", ""))
            st.text(str(r.get("kontext", ""))[:6000])
    st.download_button("⬇ Diese Excel herunterladen", data=p.read_bytes(), file_name=p.name, key=f"dl_{p.name}",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


def _bereich_vergleich() -> None:
    laeufe = _ergebnisdateien()
    if len(laeufe) < 2:
        st.caption("Für einen Vergleich werden mindestens zwei Läufe benötigt."); return
    namen = [p.name for p in laeufe]
    c1, c2 = st.columns(2)
    a = c1.selectbox("Vorher", namen, index=min(1, len(namen) - 1), key="vgl_a")
    b = c2.selectbox("Nachher", namen, index=0, key="vgl_b")
    da = _ergebnis(next(x for x in laeufe if x.name == a))
    db = _ergebnis(next(x for x in laeufe if x.name == b))
    gemeinsam = sorted(set(da["nr"]) & set(db["nr"]))
    st.caption(f"Verglichen werden nur die {len(gemeinsam)} gemeinsamen Fälle (gleiche Fälle, gleicher Judge).")
    m = da.set_index("nr").loc[gemeinsam].join(db.set_index("nr").loc[gemeinsam], lsuffix="_vorher", rsuffix="_nachher")
    zeilen = []
    for spalte, label, _, _ in METRIKEN + [("doc_hit", "Datenblatt im Kontext", None, None),
                                           ("wert_recall", "Referenzwerte in Antwort", None, None)]:
        v, n = f"{spalte}_vorher", f"{spalte}_nachher"
        if v in m and n in m:
            x, y = pd.to_numeric(m[v], errors="coerce"), pd.to_numeric(m[n], errors="coerce")
            d = (y - x).dropna()
            zeilen.append({"Metrik": label, "vorher": round(x.mean(), 3), "nachher": round(y.mean(), 3),
                           "Δ": round(y.mean() - x.mean(), 3), "besser": int((d > 0.05).sum()),
                           "schlechter": int((d < -0.05).sum())})
    st.dataframe(pd.DataFrame(zeilen), hide_index=True, use_container_width=True)
    st.caption("Unterschiede unter etwa 0,02 liegen im Bereich der Judge-Streuung. "
               "„besser/schlechter“ zählt Fälle mit einer Änderung von mehr als 0,05.")
    je_fall = pd.DataFrame({"nr": gemeinsam})
    for spalte, kurz in (("ragas_faithfulness", "F"), ("ragas_context_precision", "CP"), ("doc_hit", "Datenblatt")):
        if f"{spalte}_vorher" in m:
            je_fall[f"{kurz} vorher"] = m[f"{spalte}_vorher"].values
            je_fall[f"{kurz} nachher"] = m[f"{spalte}_nachher"].values
    if "frage_vorher" in m:
        je_fall["Frage"] = m["frage_vorher"].values
    st.dataframe(je_fall, hide_index=True, use_container_width=True)


def render() -> None:
    st.title("Evaluation")
    st.caption("Dieselbe Messung wie in der Arbeit: evaluate_petra.py, RAGAS mit lokalem Judge, "
               "deterministische Kennzahlen, Zielwerte Kap. 4.7.2.")
    fehlend = [p.name for p in (SKRIPT, TESTSET) if not p.exists()]
    if fehlend:
        st.error(f"In {EVAL_DIR} fehlen: {', '.join(fehlend)}."); return
    tab1, tab2, tab3 = st.tabs(["▶ Lauf starten", "📊 Ergebnisse je Fall", "⇄ Läufe vergleichen"])
    with tab1:
        _bereich_start()
    with tab2:
        _bereich_ergebnisse()
    with tab3:
        _bereich_vergleich()
