#!/usr/bin/env python3
# ============================================================
# evaluate_petra.py — Evaluation PETRA-RAG v3.1–v3.6 (ohne kostenpflichtige API)
#
# v3.6: --kontext-ansicht bloecke|roh und --answers (siehe main()). Ab Workflow
# v3.6 enthält "contexts" genau die Blöcke, die das Generierungsmodell sah
# (inkl. Quellen-Kopf); "contexts_roh" den reinen Chunk-Text wie bis v3.5.
#
# Behebt die Messfehler der Evaluation vom 23.09.2026:
#   1. RAGAS erhält die VOLLEN Kontext-Chunks (Feld "contexts" der
#      Workflow-Antwort v3.1) statt der auf 220 Zeichen gekürzten
#      Quellen-Textauszüge.
#   2. Judge frei wählbar: lokal über Ollama (Standard, kostenlos) oder
#      Google Gemini (kostenloser API-Schlüssel, optional).
#   3. Deterministische Metriken OHNE LLM-Judge – reproduzierbar und
#      unabhängig von der Judge-Qualität:
#        doc_hit        Datenblatt der erwarteten Artikelnummer im Kontext
#        wert_recall    Anteil der Zahlenwerte der Referenz, die in der
#                       Antwort vorkommen (Answer-Correctness-Näherung)
#        beleg_quote    Anteil der Zahlenwerte der Antwort, die im Kontext stehen
#        ablehnung_ok   Negativfälle korrekt abgelehnt (F-13)
#   4. --baseline: berechnet dieselben deterministischen Metriken für eine
#      alte Ergebnisdatei (Vorher/Nachher-Vergleich ohne neuen Judge-Lauf).
#
# Installation (einmalig, in einer venv auf dem Host):
#   pip install "ragas>=0.2" langchain-ollama pandas openpyxl httpx
#   # optional für Gemini:  pip install langchain-google-genai
#   ollama pull qwen2.5:14b      # Judge (≈9 GB, läuft teils auf CPU – langsam, aber nur offline)
#   #   oder: ollama pull qwen2.5:7b   (passt komplett in 8 GB VRAM)
#
# Aufrufbeispiele:
#   python3 evaluate_petra.py --testset testfaelle.xlsx
#   python3 evaluate_petra.py --testset testfaelle.xlsx --judge ollama:qwen2.5:7b
#   python3 evaluate_petra.py --testset testfaelle.xlsx --skip-ragas      # nur deterministisch, schnell
#   GOOGLE_API_KEY=... python3 evaluate_petra.py --testset t.xlsx --judge gemini:gemini-2.5-flash
#   python3 evaluate_petra.py --baseline ragas_evaluation_20260923_213553.xlsx --skip-run
# ============================================================
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd

# ── Deterministische Metriken ─────────────────────────────────────────
WERT_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(mm²|mm2|mm|cm|kV|mV|V|mA|kA|A|kW|mW|W|kHz|MHz|Hz|°C|kg|g|Nm|mΩ|kΩ|MΩ|Ω|%|ms|AWG)"
    r"(?![A-Za-zÄÖÜäöüß0-9²])"
)
ARTNR_RE = re.compile(r"\b\d{10}\b")
ABLEHNUNG = ("nicht belegt", "ich weiß es nicht", "ich weiss es nicht", "nicht eindeutig",
             "keine angabe", "nicht enthalten", "nicht verfügbar", "nicht verfuegbar",
             "liegen keine", "nicht angegeben", "keine informationen",
             # v3.6: Formulierungen des Produkt-Guards (N2) und der Funktionsregel (N5)
             "keine technischen daten", "nicht aufgeführt", "nicht aufgefuehrt")

ZIELE = {  # Kap. 4.7.2
    "ragas_faithfulness": (0.78, 0.85),
    "ragas_context_precision": (0.74, 0.80),
    "ragas_answer_relevancy": (None, 0.80),
}


def norm_wert(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).lower().replace(",", ".").replace("mm2", "mm²")


def werte(text: str) -> list[str]:
    seen, out = set(), []
    clean = re.sub(r"\[Quelle:[^\]]*\]", " ", str(text or ""))
    for m in WERT_RE.finditer(clean):
        t = norm_wert(m.group(1) + m.group(2))
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def wert_recall(referenz: str, antwort: str) -> float:
    ref = werte(referenz)
    if not ref:
        return math.nan
    a = norm_wert(antwort)
    return sum(1 for w in ref if w in a) / len(ref)


def beleg_quote(antwort: str, kontexte: list[str]) -> float:
    aw = werte(antwort)
    if not aw:
        return math.nan
    k = norm_wert(" ".join(kontexte))
    return sum(1 for w in aw if w in k) / len(aw)


def doc_hit(erwartet: str, quellen_text: str) -> float:
    nummern = ARTNR_RE.findall(str(erwartet or ""))
    if not nummern:
        return math.nan
    return float(any(n in str(quellen_text or "") for n in nummern))


def ist_ablehnung(antwort: str) -> bool:
    a = str(antwort or "").lower()
    return any(p in a for p in ABLEHNUNG)


def truthy(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "ja", "yes", "wahr")


# ── Workflow-Aufruf ───────────────────────────────────────────────────
EINSTELLUNGEN: dict = {}   # s7: optionale Generator-Einstellungen (--temperature/--model/--top-k)


def ask(client, webhook: str, frage: str) -> dict:
    t0 = time.time()
    try:
        r = client.post(webhook, json={"query": frage, "request_id": f"eval-{uuid.uuid4().hex[:10]}",
                                       **EINSTELLUNGEN})
        data = r.json()
        if isinstance(data, list) and data:
            data = data[0]
        err = None if r.status_code < 400 else f"HTTP {r.status_code}"
    except Exception as exc:  # noqa: BLE001
        data, err = {}, str(exc)
    data["_latency_ms_client"] = int((time.time() - t0) * 1000)
    data["_error"] = err
    return data


def run_system(cases: pd.DataFrame, webhook: str, api: str, cache_file: Path, clear_cache: bool) -> dict[str, dict]:
    import httpx
    done: dict[str, dict] = {}
    if cache_file.exists():
        for line in cache_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[str(rec["nr"])] = rec
        print(f"{len(done)} Antworten aus {cache_file} übernommen (Fortsetzung).")
    with httpx.Client(timeout=240.0) as client:
        if clear_cache and not done:
            try:
                client.post(f"{api}/cache/invalidate", json={})
                print("CAG-Cache geleert (kalte Messung).")
            except Exception as exc:  # noqa: BLE001
                print(f"Hinweis: Cache konnte nicht geleert werden: {exc}")
        with cache_file.open("a", encoding="utf-8") as fh:
            for _, row in cases.iterrows():
                nr = str(row["nr"])
                if nr in done:
                    continue
                resp = ask(client, webhook, str(row["frage"]))
                rec = {"nr": nr, "response": resp}
                done[nr] = rec
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                print(f"  #{nr:>3}  {resp.get('_latency_ms_client', 0):>6} ms  "
                      f"{(resp.get('answer') or resp.get('_error') or '')[:70]!r}")
    return done


# ── RAGAS ─────────────────────────────────────────────────────────────
def build_judge(spec: str, embed_model: str, ollama_url: str):
    provider, _, model = spec.partition(":")
    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama, OllamaEmbeddings
        except ImportError:  # z. B. im Frontend-Container: nur langchain-community
            from langchain_community.chat_models import ChatOllama
            from langchain_community.embeddings import OllamaEmbeddings
        # num_ctx 8192 reicht (11 Kontexte ≈ 3k Token) und hält qwen2.5:14b
        # vollständig auf der GPU; 16384 führte zu 22 GB und CPU-Auslagerung.
        llm = ChatOllama(model=model, base_url=ollama_url, temperature=0, num_ctx=8192, num_predict=2048)
        emb = OllamaEmbeddings(model=embed_model, base_url=ollama_url)
        workers = 1
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_ollama import OllamaEmbeddings
        llm = ChatGoogleGenerativeAI(model=model, temperature=0)
        emb = OllamaEmbeddings(model=embed_model, base_url=ollama_url)
        workers = 1   # Free-Tier: wenige Anfragen pro Minute
    else:
        raise SystemExit(f"Unbekannter Judge-Provider: {provider} (erlaubt: ollama, gemini)")
    return llm, emb, workers


def run_ragas(df: pd.DataFrame, judge: str, embed_model: str, ollama_url: str) -> pd.DataFrame:
    try:
        from ragas import EvaluationDataset, RunConfig, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (Faithfulness, LLMContextPrecisionWithReference,
                                   LLMContextRecall, ResponseRelevancy)
    except ImportError as exc:
        print(f"RAGAS nicht verfügbar ({exc}) – nur deterministische Metriken.")
        return pd.DataFrame(index=df.index)

    llm, emb, workers = build_judge(judge, embed_model, ollama_url)
    ok = df[df["contexts"].map(len) > 0]
    rows = [{
        "user_input": r.frage,
        "response": r.generierte_antwort or "",
        "retrieved_contexts": list(r.contexts),
        "reference": r.referenzantwort or "",
    } for r in ok.itertuples()]
    if not rows:
        return pd.DataFrame(index=df.index)
    print(f"RAGAS mit Judge {judge} für {len(rows)} Fälle … (lokal: ca. 1–2 min pro Fall)")
    result = evaluate(
        dataset=EvaluationDataset.from_list(rows),
        metrics=[Faithfulness(), ResponseRelevancy(strictness=1),
                 LLMContextPrecisionWithReference(), LLMContextRecall()],
        llm=LangchainLLMWrapper(llm),
        embeddings=LangchainEmbeddingsWrapper(emb),
        run_config=RunConfig(timeout=900, max_retries=3, max_workers=workers),
        raise_exceptions=False,
        show_progress=True,
    )
    res = result.to_pandas()
    mapping = {}
    for col in res.columns:
        c = col.lower()
        if c.startswith("faithfulness"):
            mapping[col] = "ragas_faithfulness"
        elif "relevancy" in c:
            mapping[col] = "ragas_answer_relevancy"
        elif "precision" in c:
            mapping[col] = "ragas_context_precision"
        elif "recall" in c:
            mapping[col] = "ragas_context_recall"
    res = res.rename(columns=mapping)[list(mapping.values())]
    res.index = ok.index
    return res.reindex(df.index)


# ── Auswertung ───────────────────────────────────────────────────────
def load_cases(path: str, sheet: str | None) -> pd.DataFrame:
    xl = pd.ExcelFile(path)
    sh = sheet or ("Evaluation Results" if "Evaluation Results" in xl.sheet_names else xl.sheet_names[0])
    df = xl.parse(sh)
    need = {"nr", "frage", "referenzantwort"}
    if not need <= set(df.columns):
        raise SystemExit(f"Testset braucht die Spalten {need}, gefunden: {list(df.columns)[:12]}")
    for col in ("artikelnummer", "nicht_beantwortbar", "fragetyp", "schwierigkeit", "produktbereich", "gruppe"):
        if col not in df.columns:
            df[col] = ""
    return df


def deterministic(df: pd.DataFrame) -> pd.DataFrame:
    df["doc_hit"] = [doc_hit(e, q) for e, q in zip(df["artikelnummer"], df["quellen"])]
    df["wert_recall"] = [wert_recall(r, a) for r, a in zip(df["referenzantwort"], df["generierte_antwort"])]
    df["beleg_quote"] = [beleg_quote(a, k) if len(k) else math.nan
                         for a, k in zip(df["generierte_antwort"], df["contexts"])]
    neg = df["nicht_beantwortbar"].map(truthy)
    df["ablehnung_ok"] = [float(ist_ablehnung(a)) if n else math.nan
                          for a, n in zip(df["generierte_antwort"], neg)]
    # v3.6: zählt auch die festen Sätze des Produkt-Guards (N2) und des
    # Leer-Kontext-Falls – sonst blieben Fehlauslösungen unsichtbar.
    df["falsche_ablehnung"] = [float(str(a).strip().lower().startswith("ich weiss es nicht")
                                     or str(a).strip().lower().startswith("ich weiß es nicht")
                                     or "keine technischen daten" in str(a).lower())
                               if not n else math.nan
                               for a, n in zip(df["generierte_antwort"], neg)]
    return df


def summary(df: pd.DataFrame, label: str) -> list[dict]:
    ans = df[~df["nicht_beantwortbar"].map(truthy)]
    rows = []

    def add(name, series, ziel1=None, ziel2=None, fmt="{:.3f}"):
        s = pd.to_numeric(series, errors="coerce").dropna()
        val = s.mean() if len(s) else math.nan
        rows.append({
            "Lauf": label, "Metrik": name, "Wert": None if math.isnan(val) else round(val, 4),
            "n": len(s),
            "Ziel Stufe 1": ziel1, "erreicht S1": None if ziel1 is None or math.isnan(val) else ("Ja" if val >= ziel1 else "Nein"),
            "Ziel Stufe 2": ziel2, "erreicht S2": None if ziel2 is None or math.isnan(val) else ("Ja" if val >= ziel2 else "Nein"),
        })

    for col, (z1, z2) in ZIELE.items():
        if col in ans:
            add(col, ans[col], z1, z2)
    if "ragas_context_recall" in ans:
        add("ragas_context_recall", ans["ragas_context_recall"])
    if "ragas_answer_relevancy" in ans:
        # v3.6: Ergänzende Auswertung. RAGAS wertet Antworten mit "nicht belegt"
        # als "noncommittal" (Relevanz 0) – auch wenn nur EIN Teilaspekt fehlt.
        # Diese Zeile zeigt die Relevanz der übrigen Antworten (kein Zielwert).
        _teil = ans["generierte_antwort"].astype(str).str.lower().str.contains(
            "nicht belegt|nicht aufgeführt|nicht aufgefuehrt|keine technischen daten", regex=True)
        add("ragas_answer_relevancy – ohne Antworten mit 'nicht belegt' (ergänzend)",
            ans.loc[~_teil, "ragas_answer_relevancy"])
    add("doc_hit (Datenblatt im Kontext)", ans["doc_hit"])
    add("wert_recall (Referenzwerte in Antwort)", ans["wert_recall"])
    add("beleg_quote (Antwortwerte im Kontext)", ans["beleg_quote"])
    add("falsche_ablehnung (beantwortbar)", ans["falsche_ablehnung"])
    add("ablehnung_ok (Negativfälle, F-13)", df["ablehnung_ok"])
    if "gruppe" in df:
        for g, metrik in (("negativ_original", "ablehnung_ok – ursprüngliche Negativfälle"),
                          ("out_of_corpus", "ablehnung_ok – Produkt nicht im Korpus")):
            sub = df[df["gruppe"] == g]
            if len(sub):
                add(metrik, sub["ablehnung_ok"])
    if "latency_ms" in df:
        lat = pd.to_numeric(df["latency_ms"], errors="coerce")
        rows.append({"Lauf": label, "Metrik": "Latenz Ø ms", "Wert": round(lat.mean(), 0), "n": lat.notna().sum()})
        anteil = round((lat <= 5000).mean(), 3)
        rows.append({"Lauf": label, "Metrik": "Anteil ≤ 5 s (NF-03)", "Wert": anteil,
                     "n": lat.notna().sum(), "Ziel Stufe 1": 1.0, "erreicht S1": "Ja" if anteil >= 1.0 else "Nein"})
    if "cache_hit" in df:   # Cache-Test: Latenz getrennt nach Treffer / kein Treffer
        hit = df["cache_hit"].fillna(False).astype(bool)
        lat_c = pd.to_numeric(df.get("latency_client_ms", df.get("latency_ms")), errors="coerce")
        rows.append({"Lauf": label, "Metrik": "Cache-Treffer (Anteil)", "Wert": round(hit.mean(), 3), "n": len(hit)})
        for maske, name in ((hit, "Latenz Ø ms – Cache-Treffer"), (~hit, "Latenz Ø ms – ohne Cache-Treffer")):
            s = lat_c[maske].dropna()
            rows.append({"Lauf": label, "Metrik": name, "Wert": round(s.mean(), 0) if len(s) else None, "n": len(s)})
    return rows


def baseline_frame(path: str) -> pd.DataFrame:
    """Alte Ergebnisdatei (Format 23.09.2026) → gleiche Spalten wie neuer Lauf."""
    b = pd.read_excel(path, sheet_name="Evaluation Results")
    b["contexts"] = b["kontext"].fillna("").astype(str).map(lambda s: [c for c in s.split("---") if c.strip()])
    b["quellen"] = b["quellen"].fillna("").astype(str)
    b["generierte_antwort"] = b["generierte_antwort"].fillna("").astype(str)
    return deterministic(b)


def _diagramme(out: Path, summ: list[dict], sheets: dict) -> None:
    """Blatt "Diagramm": RAGAS-Metriken gegen Zielwerte (Kap. 4.7.2) und Latenz je Klasse."""
    try:
        from openpyxl import load_workbook
        from openpyxl.chart import BarChart, Reference
    except ImportError:
        return
    namen = [("ragas_faithfulness", "Faithfulness"), ("ragas_context_precision", "Context Precision"),
             ("ragas_answer_relevancy", "Answer Relevancy"), ("ragas_context_recall", "Context Recall")]
    def wert(lauf_prefix, metrik):
        for r in summ:
            if str(r.get("Lauf", "")).startswith(lauf_prefix) and r.get("Metrik") == metrik:
                v = r.get("Wert")
                return float(v) if isinstance(v, (int, float)) and v == v else None
        return None
    zeilen = [(lbl, wert("Neu", m), wert("Baseline", m), (ZIELE.get(m) or (None, None))[0],
               (ZIELE.get(m) or (None, None))[1]) for m, lbl in namen]
    if all(z[1] is None for z in zeilen):
        return                                   # Lauf ohne RAGAS: kein Metrik-Diagramm
    wb = load_workbook(out)
    ws = wb.create_sheet("Diagramm", 1)
    ws.append(["Metrik", "Dieser Lauf", "Baseline 23.09. (alte Messmethode)", "Ziel Stufe 1", "Ziel Stufe 2"])
    for z in zeilen:
        ws.append(list(z))
    ch = BarChart(); ch.type = "col"; ch.grouping = "clustered"
    ch.title = "RAGAS-Metriken gegen Zielwerte (0 bis 1)"; ch.y_axis.title = "Wert"
    ch.y_axis.scaling.min = 0; ch.y_axis.scaling.max = 1
    ch.add_data(Reference(ws, min_col=2, max_col=5, min_row=1, max_row=5), titles_from_data=True)
    ch.set_categories(Reference(ws, min_col=1, min_row=2, max_row=5))
    ch.width, ch.height = 22, 11
    ch.legend.position = "b"
    ws.add_chart(ch, "G2")
    nk = sheets.get("Nach Klasse")
    if nk is not None and "latency_ms" in nk.columns:
        start = 8
        ws.cell(row=start, column=1, value="Klasse"); ws.cell(row=start, column=2, value="Latenz Ø (s)")
        for i, (klasse, ms) in enumerate(nk["latency_ms"].items(), start=1):
            ws.cell(row=start + i, column=1, value=str(klasse))
            ws.cell(row=start + i, column=2, value=round(float(ms) / 1000, 1) if ms == ms else None)
        lc = BarChart(); lc.type = "bar"; lc.title = "Latenz je Anfrageklasse (Ziel NF-03: 5 s)"
        lc.x_axis.title = "Sekunden"
        lc.add_data(Reference(ws, min_col=2, min_row=start, max_row=start + len(nk)), titles_from_data=True)
        lc.set_categories(Reference(ws, min_col=1, min_row=start + 1, max_row=start + len(nk)))
        lc.width, lc.height = 22, 9
        lc.legend = None
        ws.add_chart(lc, "G25")
    ws.column_dimensions["A"].width = 30
    wb.save(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", help="Excel mit nr, frage, referenzantwort (+ artikelnummer, nicht_beantwortbar)")
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--webhook", default="http://localhost:5678/webhook/petra-query")
    ap.add_argument("--api", default="http://localhost:8001")
    ap.add_argument("--ollama", default="http://localhost:11434")
    ap.add_argument("--judge", default="ollama:qwen2.5:14b")
    ap.add_argument("--embed-model", default="bge-m3:latest")
    ap.add_argument("--skip-ragas", action="store_true")
    ap.add_argument("--skip-run", action="store_true", help="Kein Systemlauf (z. B. nur --baseline)")
    ap.add_argument("--no-clear-cache", action="store_true")
    # s7: nur mit Workflow "Schritt 7" wirksam; ohne Angabe gilt der Endstand s6
    ap.add_argument("--temperature", type=float, default=None, help="Generator-Temperatur 0–1")
    ap.add_argument("--model", default=None, help="Generator-Modell (muss in Ollama installiert sein)")
    ap.add_argument("--top-k", type=int, default=None, help="Kontext-Abschnitte 1–10")
    ap.add_argument("--baseline", default=None, help="Alte Ergebnis-Excel für Vorher/Nachher")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--faelle", default=None,
                    help="Nur diese Fallnummern, kommagetrennt (z. B. 3,5,12) – für Teiltests")
    ap.add_argument("--nur", default=None, help="Nur diese Fallnummern, z. B. 3,27,48")
    ap.add_argument("--out", default=None)
    # v3.6 (M1): Welche Kontextsicht der Judge erhält. "bloecke" = exakt die
    # Prompt-Blöcke inkl. Quellen-Kopf (ab Workflow v3.6 im Feld "contexts"),
    # "roh" = nur Chunk-Text (Feld "contexts_roh", Stand bis v3.5). Mit
    # --answers lassen sich DIESELBEN Antworten in beiden Sichten bewerten.
    ap.add_argument("--kontext-ansicht", choices=["bloecke", "roh"], default="bloecke")
    ap.add_argument("--answers", default=None,
                    help="answers.jsonl wiederverwenden (Standard: <out>.answers.jsonl)")
    args = ap.parse_args()
    for _k, _v in (("temperature", args.temperature), ("model", args.model), ("top_k", args.top_k)):
        if _v is not None:
            EINSTELLUNGEN[_k] = _v
    if EINSTELLUNGEN:
        print("Generator-Einstellungen für diesen Lauf:", EINSTELLUNGEN)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.out or f"petra_evaluation_{stamp}.xlsx")
    summ: list[dict] = []
    sheets: dict[str, pd.DataFrame] = {}

    if args.baseline:
        b = baseline_frame(args.baseline)
        summ += summary(b, "Baseline (alt)")   # alte RAGAS-Spalten fließen mit ein
        sheets["Baseline"] = b.drop(columns=["contexts"])

    if not args.skip_run:
        if not args.testset:
            raise SystemExit("--testset fehlt")
        cases = load_cases(args.testset, args.sheet)
        if args.nur:
            wahl = {x.strip() for x in str(args.nur).split(",") if x.strip()}
            cases = cases[cases["nr"].astype(str).isin(wahl)]
            print(f"Nur Fälle: {', '.join(sorted(wahl, key=lambda x: int(x) if x.isdigit() else 0))} ({len(cases)} gefunden)")
        if args.limit:
            cases = cases.head(args.limit)
        if args.faelle:
            wahl = {int(x) for x in str(args.faelle).replace(" ", "").split(",") if x}
            cases = cases[cases["nr"].astype(int).isin(wahl)]
            print(f"Teiltest: {len(cases)} von {len(wahl)} gewählten Fällen gefunden.")
        if args.answers:
            # v3.6: --answers ist nur zum Wiederverwenden gedacht. Fehlt ein Fall
            # oder enthält er einen Fehler, NICHT still nachfragen (Cache-Zustand
            # und Workflow-Version wären dann gemischt), sondern abbrechen.
            _pfad = Path(args.answers)
            if not _pfad.exists():
                raise SystemExit(f"--answers: Datei {_pfad} nicht gefunden.")
            _vorh = {}
            for _z in _pfad.read_text(encoding="utf-8").splitlines():
                if _z.strip():
                    _rec = json.loads(_z)
                    _vorh[str(_rec["nr"])] = _rec
            _fehlt = [str(n) for n in cases["nr"] if str(n) not in _vorh]
            _fehler = [str(n) for n in cases["nr"] if str(n) in _vorh and (_vorh[str(n)].get("response") or {}).get("_error")]
            if _fehlt or _fehler:
                raise SystemExit(f"--answers unvollständig: fehlend {_fehlt[:10]}, mit Fehler {_fehler[:10]} – "
                                 "ohne --answers neu laufen lassen.")
        answers = run_system(cases, args.webhook, args.api,
                             Path(args.answers) if args.answers else out.with_suffix(".answers.jsonl"),
                             clear_cache=not args.no_clear_cache)
        rows = []
        for _, c in cases.iterrows():
            r = answers[str(c["nr"])]["response"]
            if args.kontext_ansicht == "roh" and r.get("contexts_roh"):
                ctx, ctx_quelle = r.get("contexts_roh"), "contexts_roh (Chunk-Text)"
            else:
                ctx = r.get("contexts")
                ctx_quelle = "contexts (Prompt-Blöcke mit Quellen-Kopf)" if r.get("contexts_roh") else "contexts"
            if not ctx:   # Fallback für Workflows < v3.1 (gekürzt!)
                ctx = [s.get("textauszug", "") for s in r.get("sources", [])]
                ctx_quelle = "textauszug_220 (GEKÜRZT)" if ctx else "keine"
            quellen = " | ".join(f"{s.get('dateiname')} S.{s.get('seite')} [{s.get('artikelnummer') or ''}]"
                                 for s in r.get("sources", []))
            rows.append({
                "nr": c["nr"], "gruppe": c["gruppe"], "produktbereich": c["produktbereich"], "fragetyp": c["fragetyp"],
                "schwierigkeit": c["schwierigkeit"], "artikelnummer": c["artikelnummer"],
                "nicht_beantwortbar": c["nicht_beantwortbar"], "frage": c["frage"],
                "referenzantwort": c["referenzantwort"],
                "generierte_antwort": r.get("answer", ""),
                "query_class": r.get("query_class"), "latency_ms": r.get("latency_ms") or r.get("_latency_ms_client"),
                # Cache-Test: Treffer kennzeichnet der Node "Cache-Antwort aufbereiten" (cache_hit: true);
                # latency_client_ms = Ende-zu-Ende aus Sicht des Aufrufers (inkl. n8n/HTTP).
                "cache_hit": r.get("cache_hit") is True, "latency_client_ms": r.get("_latency_ms_client"),
                "verified": r.get("verified"), "system_faithfulness": r.get("faithfulness_score"),
                "unbelegte_werte": ", ".join(r.get("unbelegte_werte") or []),
                "kontext_quelle": ctx_quelle, "kontext_anzahl": len(ctx), "contexts": ctx,
                "kontext": "\n---\n".join(ctx), "quellen": quellen, "fehler": r.get("_error"),
            })
        df = deterministic(pd.DataFrame(rows))
        if not args.skip_ragas:
            df = df.join(run_ragas(df, args.judge, args.embed_model, args.ollama))
        summ += summary(df, f"Neu ({args.judge if not args.skip_ragas else 'ohne RAGAS'})")
        sheets["Ergebnisse"] = df.drop(columns=["contexts"])
        num = [c for c in df.columns if c.startswith("ragas_")] + ["doc_hit", "wert_recall", "beleg_quote", "latency_ms"]
        sheets["Nach Klasse"] = df.groupby("query_class", dropna=False)[num].agg(
            lambda s: pd.to_numeric(s, errors="coerce").mean()).round(3).assign(
            n=df.groupby("query_class", dropna=False).size())

    # Excel verbietet Steuerzeichen (z. B. \x07 aus PDF-Extraktion) und
    # Zellen > 32767 Zeichen → vor dem Schreiben bereinigen.
    _bad = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

    def _clean(v):
        if isinstance(v, str):
            v = _bad.sub(" ", v)
            return v if len(v) <= 32000 else v[:32000] + " …[gekürzt]"
        return v

    def _clean_df(frame):
        frame = frame.copy()
        # pandas 3 speichert Text als dtype "str" (nicht object) → alle Spalten
        for c in frame.columns:
            frame[c] = frame[c].astype(object).map(_clean)
        return frame

    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        _clean_df(pd.DataFrame(summ)).to_excel(xw, sheet_name="Zusammenfassung", index=False)
        for name, frame in sheets.items():
            _clean_df(frame).to_excel(xw, sheet_name=name, index=name == "Nach Klasse")
        pd.DataFrame([{"Parameter": k, "Wert": str(v)} for k, v in vars(args).items()]).to_excel(
            xw, sheet_name="Lauf-Metadaten", index=False)
    _diagramme(out, summ, sheets)   # Blatt "Diagramm" (Metriken vs. Ziele, Latenz)
    print(pd.DataFrame(summ).to_string(index=False))
    print(f"\nGeschrieben: {out}")


if __name__ == "__main__":
    main()
