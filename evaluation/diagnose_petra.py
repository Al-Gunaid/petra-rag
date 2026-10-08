#!/usr/bin/env python3
# ============================================================
# diagnose_petra.py — Ursachenanalyse VOR dem Umbau (Tests B, C, D)
#
# Beantwortet mit Messwerten statt Vermutungen:
#   Test C  Stufen-Trace: Ist das Datenblatt der gefragten Artikelnummer
#           überhaupt im Index? Findet Dense / BM25 / TAG es? Überlebt es
#           den Cross-Encoder (Top-8)?
#   Test B  Oracle-Test: Das LLM (llama3.1:8b) bekommt die Chunks des
#           RICHTIGEN Datenblatts. Antwortet es dann richtig?
#             ja   → Fehler liegt im Retrieval  → Re-Indexierung lohnt sich
#             nein → Fehler liegt im LLM/Prompt → Modell/Prompt ändern
#   Test D  Schwellwert: Top-Rerank-Score der Negativfälle vs. beantwortbare
#           Fälle → belegter Wert für RERANK_MIN im CRAG-Filter.
#
# Braucht KEINE Re-Indexierung und KEINEN kostenpflichtigen Judge.
# Nutzt eure laufenden Dienste: ChromaDB, Ollama, ingestion-api (BM25 + Reranker).
#
# Installation (Host):  pip install chromadb httpx pandas openpyxl
# Aufruf:
#   python3 diagnose_petra.py --excel ragas_evaluation_20260923_213553.xlsx
#   python3 diagnose_petra.py --excel ... --limit 10          # schneller Probelauf
#   python3 diagnose_petra.py --excel ... --no-oracle         # nur Trace + Schwellwert (ohne LLM)
# Dauer: ca. 10–20 min für 75 Fragen (Oracle-LLM ist der langsamste Teil).
# ============================================================
from __future__ import annotations

import argparse
import math
import re
import sys
import time
from datetime import datetime

import httpx
import pandas as pd

ARTNR_RE = re.compile(r"\b\d{10}\b")
WERT_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(mm²|mm2|mm|cm|kV|mV|V|mA|kA|A|kW|mW|W|kHz|MHz|Hz|°C|kg|g|Nm|mΩ|kΩ|MΩ|Ω|%|ms|AWG)"
    r"(?![A-Za-zÄÖÜäöüß0-9²])"
)

ORACLE_PROMPT = """Du bist ein technischer Produktassistent für Weidmüller-Produkte.
Beantworte die Frage ausschließlich mit dem Kontext unten.

REGELN
1. Werte, Einheiten und Bezeichnungen zeichengenau aus dem Kontext übernehmen.
2. Nur Werte des gefragten Produkts verwenden, keine Werte ähnlicher Produkte.
3. Nichts schätzen, runden oder berechnen.
4. Steht ein gefragtes Attribut nicht im Kontext: "<Attribut> nicht belegt."
5. Antworte knapp in vollständigen Sätzen, höchstens 6 Sätze.

Kontext:
{kontext}

Frage: {frage}"""


# ── Hilfsfunktionen ───────────────────────────────────────────────────
def norm_wert(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).lower().replace(",", ".").replace("mm2", "mm²")


def werte(text: str) -> list[str]:
    clean = re.sub(r"\[Quelle:[^\]]*\]", " ", str(text or ""))
    out: list[str] = []
    for m in WERT_RE.finditer(clean):
        t = norm_wert(m.group(1) + m.group(2))
        if t not in out:
            out.append(t)
    return out


def wert_recall(referenz: str, antwort: str) -> float:
    ref = werte(referenz)
    if not ref:
        return math.nan
    a = norm_wert(antwort)
    return sum(1 for w in ref if w in a) / len(ref)


def truthy(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "ja", "yes", "wahr")


def fn_of(meta: dict) -> str:
    return str((meta or {}).get("filename") or (meta or {}).get("source") or (meta or {}).get("document") or "")


class Services:
    def __init__(self, args) -> None:
        import chromadb
        self.args = args
        self.http = httpx.Client(timeout=180.0)
        self.chroma = chromadb.HttpClient(host=args.chroma_host, port=args.chroma_port)
        self.text = self.chroma.get_collection(args.text_collection)
        self.tag = self.chroma.get_collection(args.tag_collection)

    def embed(self, text: str) -> list[float]:
        r = self.http.post(f"{self.args.ollama}/api/embed", json={"model": self.args.embed_model, "input": text})
        if r.status_code == 404:
            r = self.http.post(f"{self.args.ollama}/api/embeddings",
                               json={"model": self.args.embed_model, "prompt": text})
            r.raise_for_status()
            v = r.json()["embedding"]
        else:
            r.raise_for_status()
            v = r.json()["embeddings"][0]
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def dense(self, coll, emb, n) -> list[dict]:
        res = coll.query(query_embeddings=[emb], n_results=n, include=["documents", "metadatas", "distances"])
        return [{"chunk_id": i, "text": d or "", "metadata": m or {}, "score": 1 - float(s)}
                for i, d, m, s in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])]

    def bm25(self, q: str, n: int) -> list[dict]:
        r = self.http.get(f"{self.args.api}/retrieval/bm25", params={"q": q, "top_k": n})
        r.raise_for_status()
        return r.json().get("results", [])

    def rerank(self, q: str, chunks: list[dict], out_n: int) -> tuple[list[dict], bool]:
        if not chunks:
            return [], False
        r = self.http.post(f"{self.args.api}/retrieval/rerank",
                           json={"query": q, "chunks": chunks, "input_top_n": len(chunks), "output_top_n": out_n})
        r.raise_for_status()
        j = r.json()
        return j.get("chunks", []), bool(j.get("reranked"))

    def by_files(self, files: list[str]) -> list[dict]:
        where = {"filename": files[0]} if len(files) == 1 else {"filename": {"$in": files}}
        out = []
        for coll in (self.text, self.tag):
            res = coll.get(where=where, limit=300, include=["documents", "metadatas"])
            out += [{"chunk_id": i, "text": d or "", "metadata": m or {}}
                    for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])]
        return out

    def generate(self, prompt: str) -> str:
        r = self.http.post(f"{self.args.ollama}/api/generate", json={
            "model": self.args.llm, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.1, "num_ctx": 8192, "num_predict": 900}})
        r.raise_for_status()
        return r.json().get("response", "").strip()


# ── Diagnose je Fall ─────────────────────────────────────────────────
def diagnose_case(s: Services, row, oracle: bool) -> dict:
    frage = str(row["frage"])
    neg = truthy(row.get("nicht_beantwortbar", False))
    nummern = list(dict.fromkeys(ARTNR_RE.findall(str(row.get("artikelnummer", "")))))
    files = [f"{n}_de.pdf" for n in nummern]
    out = {"nr": row["nr"], "nicht_beantwortbar": neg, "frage": frage,
           "erwartete_datei": ", ".join(files)}

    emb = s.embed(frage)
    d = s.dense(s.text, emb, 20)
    t = s.dense(s.tag, emb, 10)
    try:
        b = s.bm25(frage, 20)
    except Exception as exc:  # noqa: BLE001
        b, out["bm25_fehler"] = [], str(exc)

    def hit(lst):
        return any(fn_of(c.get("metadata")) in files for c in lst) if files else None

    # Kandidaten wie im Workflow: Vereinigung, dann Cross-Encoder
    seen, union = set(), []
    for lst in (d, b, t):
        for c in lst:
            if c["chunk_id"] not in seen:
                seen.add(c["chunk_id"])
                union.append({"chunk_id": c["chunk_id"], "text": c.get("text", ""), "metadata": c.get("metadata", {})})
    top8, reranked = s.rerank(frage, union[:30], 8)
    scores = [c.get("rerank_score") for c in top8 if isinstance(c.get("rerank_score"), (int, float))]

    out.update({
        "dense_hit@20": hit(d), "bm25_hit@20": hit(b), "tag_hit@10": hit(t),
        "kandidaten_hit@30": hit(union[:30]), "rerank_hit@8": hit(top8),
        "reranker_aktiv": reranked,
        "top_rerank_score": max(scores) if scores else None,
        "top8_dateien": " | ".join(dict.fromkeys(fn_of(c.get("metadata")) for c in top8)),
    })

    if files:
        oracle_chunks = s.by_files(files)
        out["datenblatt_im_index"] = bool(oracle_chunks)
        out["datenblatt_chunks"] = len(oracle_chunks)
    else:
        oracle_chunks = []
        out["datenblatt_im_index"] = None

    out["wert_recall_alt"] = wert_recall(row.get("referenzantwort", ""), row.get("generierte_antwort", ""))
    if oracle and oracle_chunks and not neg:
        best, _ = s.rerank(frage, oracle_chunks[:60], 8)
        best = best or oracle_chunks[:8]
        kontext = "\n\n".join(f"[Quelle: {fn_of(c.get('metadata'))}, S. {(c.get('metadata') or {}).get('page', '?')}]\n"
                              f"{c.get('text', '')}" for c in best)
        t0 = time.time()
        antwort = s.generate(ORACLE_PROMPT.format(kontext=kontext, frage=frage))
        out["oracle_antwort"] = antwort
        out["oracle_sek"] = round(time.time() - t0, 1)
        out["wert_recall_oracle"] = wert_recall(row.get("referenzantwort", ""), antwort)
    return out


def ursache(r: dict) -> str:
    if r["nicht_beantwortbar"]:
        return "Negativfall (siehe Test D)"
    if r.get("datenblatt_im_index") is None:
        return "keine Artikelnummer im Testfall"
    if r.get("datenblatt_im_index") is False:
        return "INDEX: Datenblatt fehlt in ChromaDB"
    if not r.get("kandidaten_hit@30"):
        return "RETRIEVAL: Suche findet Datenblatt nicht"
    if not r.get("rerank_hit@8"):
        return "RERANKER: Datenblatt gefunden, aber aussortiert"
    wo = r.get("wert_recall_oracle", math.nan)
    if isinstance(wo, float) and not math.isnan(wo):
        # Referenzantworten nennen oft MEHR Werte als gefragt (z. B. Nennstrom
        # UND Spannung UND Querschnitt). Eine korrekte, knappe Antwort erreicht
        # dann nur einen Teil. Deshalb: 0 = kein einziger Referenzwert → echter
        # LLM-Fehler; 0 < x < 0.5 → nur unvollständig, von Hand prüfen.
        if wo == 0:
            return "LLM: falsch trotz richtigem Kontext"
        if wo < 0.5:
            return "LLM: unvollständig (Spalte oracle_antwort prüfen)"
    return "Retrieval ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", required=True, help="Alte Ergebnisdatei (Blatt 'Evaluation Results')")
    ap.add_argument("--chroma-host", default="localhost")
    ap.add_argument("--chroma-port", type=int, default=8000)
    ap.add_argument("--text-collection", default="petra_text_chunks_v2")
    ap.add_argument("--tag-collection", default="petra_tag_chunks")
    ap.add_argument("--ollama", default="http://localhost:11434")
    ap.add_argument("--api", default="http://localhost:8001")
    ap.add_argument("--embed-model", default="bge-m3:latest")
    ap.add_argument("--llm", default="llama3.1:8b")
    ap.add_argument("--no-oracle", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=f"diagnose_{datetime.now():%Y%m%d_%H%M%S}.xlsx")
    args = ap.parse_args()

    cases = pd.read_excel(args.excel, sheet_name="Evaluation Results")
    if args.limit:
        cases = cases.head(args.limit)
    s = Services(args)

    rows = []
    for _, row in cases.iterrows():
        try:
            r = diagnose_case(s, row, oracle=not args.no_oracle)
        except Exception as exc:  # noqa: BLE001
            r = {"nr": row["nr"], "nicht_beantwortbar": truthy(row.get("nicht_beantwortbar")),
                 "frage": row["frage"], "fehler": str(exc)}
        r["ursache"] = ursache(r) if "fehler" not in r else "Skriptfehler"
        rows.append(r)
        print(f"#{r['nr']:>3}  {r['ursache']:<48} top_rerank={r.get('top_rerank_score')}")

    df = pd.DataFrame(rows)
    pos = df[~df["nicht_beantwortbar"].astype(bool)]
    neg = df[df["nicht_beantwortbar"].astype(bool)]

    def rate(col):
        s_ = pos[col].dropna() if col in pos else pd.Series(dtype=float)
        return round(s_.astype(float).mean(), 3) if len(s_) else None

    zus = [
        ("Datenblatt im Index", rate("datenblatt_im_index")),
        ("Dense findet es (Top 20)", rate("dense_hit@20")),
        ("BM25 findet es (Top 20)", rate("bm25_hit@20")),
        ("TAG findet es (Top 10)", rate("tag_hit@10")),
        ("In den 30 Kandidaten", rate("kandidaten_hit@30")),
        ("Nach Cross-Encoder (Top 8)", rate("rerank_hit@8")),
        ("wert_recall alte Antwort", rate("wert_recall_alt")),
        ("wert_recall mit richtigem Kontext (Oracle)", rate("wert_recall_oracle")),
    ]

    # Test D: Schwellwert
    ps = pos["top_rerank_score"].dropna().astype(float) if "top_rerank_score" in pos else pd.Series(dtype=float)
    ns = neg["top_rerank_score"].dropna().astype(float) if "top_rerank_score" in neg else pd.Series(dtype=float)
    if len(ps) and len(ns):
        p10 = ps.quantile(0.10)
        zus += [("Top-Rerank-Score beantwortbar: 10 %-Quantil", round(p10, 4)),
                ("Top-Rerank-Score Negativfälle: Maximum", round(ns.max(), 4))]
        if ns.max() < p10:
            empf_d = f"Trennbar: RERANK_MIN ≈ {round((ns.max() + p10) / 2, 4)} setzen (statt 0.01)."
        else:
            empf_d = ("Nicht trennbar: Negativfälle erreichen ähnliche Scores. Eine Schwelle allein "
                      "reicht nicht; zusätzlich die Regel 'nicht belegt' im Prompt und den Belegcheck nutzen.")
    else:
        empf_d = "Zu wenig Daten für Test D."

    verteilung = df["ursache"].value_counts()
    ret = sum(v for k, v in verteilung.items() if k.startswith(("RETRIEVAL", "RERANKER", "INDEX")))
    llm = verteilung.get("LLM: falsch trotz richtigem Kontext", 0)
    llm_teil = verteilung.get("LLM: unvollständig (Spalte oracle_antwort prüfen)", 0)
    wa, wo = rate("wert_recall_alt"), rate("wert_recall_oracle")
    if wa is not None and wo is not None and wo - wa >= 0.15:
        empf_b = (f"Mit dem richtigen Datenblatt steigt wert_recall von {wa} auf {wo}. "
                  "→ Hauptursache ist das RETRIEVAL. Re-Indexierung mit Produkt-Kopf + Produkt-Lexikon umsetzen.")
    elif wo is not None and wo < 0.5:
        empf_b = (f"Auch mit richtigem Datenblatt nur wert_recall {wo}. "
                  "→ Hauptursache ist das LLM/der Prompt. Zuerst anderes Modell (z. B. qwen2.5:7b) oder Prompt testen.")
    else:
        empf_b = "Kein eindeutiges Ergebnis; Fälle einzeln im Blatt 'Diagnose' ansehen."

    empf = pd.DataFrame([
        {"Test": "B/C Ursache", "Ergebnis": f"{ret} Fälle Retrieval/Index/Reranker, {llm} Fälle LLM falsch, {llm_teil} LLM unvollständig", "Empfehlung": empf_b},
        {"Test": "D Schwellwert", "Ergebnis": f"{len(ns)} Negativfälle", "Empfehlung": empf_d},
    ])
    with pd.ExcelWriter(args.out, engine="openpyxl") as xw:
        empf.to_excel(xw, sheet_name="Empfehlung", index=False)
        pd.DataFrame(zus, columns=["Kennzahl", "Wert (beantwortbare Fälle)"]).to_excel(
            xw, sheet_name="Zusammenfassung", index=False)
        verteilung.rename_axis("Ursache").reset_index(name="Anzahl").to_excel(
            xw, sheet_name="Ursachen", index=False)
        df.to_excel(xw, sheet_name="Diagnose", index=False)

    print("\n" + pd.DataFrame(zus, columns=["Kennzahl", "Wert"]).to_string(index=False))
    print("\nUrsachen:\n" + verteilung.to_string())
    print("\nEMPFEHLUNG:\n  " + empf_b + "\n  " + empf_d)
    print(f"\nGeschrieben: {args.out}")


if __name__ == "__main__":
    main()
