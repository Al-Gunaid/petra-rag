#!/usr/bin/env python3
# ============================================================
# reindex_v3_contextual.py — Kontextuelle Re-Indexierung (v3)
#
# PROBLEM (Evaluation 23.09.2026):
#   Chunks aus Weidmüller-Datenblättern enthalten keinen Produktnamen
#   ("Bemessungsdaten ... Nennstrom 24 A"). Die Chunks der Varianten
#   WDU 2.5 / WDU 2.5 GN / WDU 2.5 WS / WDU 2.5/TC ... sind damit für
#   Dense, BM25 UND Cross-Encoder ununterscheidbar. Folge: Das Datenblatt
#   des gefragten Produkts war nur in 34 % der Testfälle im Kontext.
#
# LÖSUNG (Contextual Chunk Headers + Produkt-Lexikon):
#   1. Pass 1 liest alle vorhandenen Chunks aus ChromaDB (v2) und leitet
#      je Datenblatt ab:  Artikelnummer (Dateiname), Produktname ("Art"-
#      Feld der Bestelldaten, S. 1), Ausführung, Abschnitt je Chunk.
#   2. Pass 2 stellt jedem Chunk einen kurzen Kopf voran, z. B.
#        [Produkt: WDU 2.5 | Art.-Nr.: 1020000000 | Abschnitt: Bemessungsdaten]
#      bettet ihn NEU mit bge-m3 ein und schreibt ihn in neue
#      Collections (…_v3). Die chunk_ids bleiben unverändert.
#   3. Der BM25-Index wird aus denselben Texten als neue JSONL-Datei
#      geschrieben (identische IDs → RRF bleibt konsistent).
#   4. product_lexicon.json: Produktname → Artikelnummer(n). Das ist die
#      schlanke Umsetzung des in Kap. 4.7.2 (Stufe 2) geplanten
#      Graph-Metadaten-Layers (JSON, Python-Dict im Speicher).
#
# Die bestehenden v2-Collections werden NICHT verändert (Rollback möglich).
#
# AUFRUF (im ingestion-Container, dort wo save_to_chromadb.py liegt):
#   # 1) Probelauf: zeigt Beispiel-Köpfe + Lexikon-Statistik, schreibt nichts
#   python3 reindex_v3_contextual.py --dry-run
#   # 2) Kurzer Test mit 2000 Chunks
#   python3 reindex_v3_contextual.py --limit 2000
#   # 3) Vollständiger Lauf (fortsetzbar – bereits eingebettete IDs werden übersprungen)
#   python3 reindex_v3_contextual.py
# ============================================================
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] reindex_v3 – %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("reindex_v3")

# ── Konfiguration (ENV überschreibbar) ───────────────────────────────
SRC_TEXT = os.environ.get("REINDEX_SRC_TEXT", "petra_text_chunks_v2")
SRC_TAG = os.environ.get("REINDEX_SRC_TAG", "petra_tag_chunks")
DST_TEXT = os.environ.get("REINDEX_DST_TEXT", "petra_text_chunks_v3")
DST_TAG = os.environ.get("REINDEX_DST_TAG", "petra_tag_chunks_v3")
BM25_OUT = os.environ.get("REINDEX_BM25_OUT", "/data/bm25_index_v3.jsonl")
LEXICON_OUT = os.environ.get("PETRA_PRODUCT_LEXICON", "/data/product_lexicon.json")
PAGE_SIZE = int(os.environ.get("REINDEX_PAGE_SIZE", "2000"))
EMBED_BATCH = int(os.environ.get("REINDEX_EMBED_BATCH", "32"))
MANUFACTURER = "Weidmüller"

DATASHEET_RE = re.compile(r"^(\d{10})_de\.pdf$", re.IGNORECASE)

# Abschnittsüberschriften der Weidmüller-Datenblätter. Erkannt wird nur
# eine Überschrift, die ALLEIN auf einer Zeile steht – sonst würde jedes
# "Technische Daten" im Fließtext einen Abschnittswechsel auslösen.
SECTION_HEADINGS = [
    "Allgemeine Bestelldaten", "Produktbeschreibung", "Technische Daten",
    "Weitere technische Daten", "Abmessungen und Gewichte", "Abmessungen",
    "Temperaturen", "Umweltanforderungen", "Umweltverträglichkeit des Produkts",
    "Werkstoffdaten", "Bemessungsdaten", "Bemessungsdaten UL", "Bemessungsdaten CSA",
    "Bemessungsdaten ATEX", "Bemessungsdaten IECEx", "Ex-Bemessungsdaten",
    "Klemmbare Leiter", "Anschlussdaten", "Leiteranschluss", "Systemdaten",
    "Isolationskoordination", "Allgemeine Daten", "Allgemein", "Eingang", "Ausgang",
    "Versorgung", "Stromversorgung", "EMV", "Sicherheit", "Klassifikationen",
    "Zulassungen", "Zubehör", "Downloads", "Zeichnungen", "Wichtiger Hinweis",
    "Anschlusstechnik", "Montage", "Messbereich", "Schnittstelle", "Kommunikation",
]
_HEADING_LOOKUP = {h.casefold(): h for h in SECTION_HEADINGS}
_HEADING_LOOKUP.update({h.casefold().replace("ä", "ae").replace("ö", "oe").replace("ü", "ue"): h
                        for h in SECTION_HEADINGS})

# "Best.-Nr.\n<artnr>\nArt\n<NAME>" – Layout der Bestelldaten auf S. 1
_ART_AFTER_NR = r"Best\.-Nr\.\s*\n\s*{artnr}\s*\n\s*(?:Art|Typ)\s*\n\s*([^\n]{{2,80}})"
_ART_GENERIC = re.compile(r"(?:^|\n)\s*(?:Art|Typ)\s*\n\s*([^\n]{2,80})")
_AUSF = re.compile(r"Ausf(?:ü|ue)hrung\s*\n((?:[^\n]+\n){1,4}?)\s*Best\.-Nr\.", re.IGNORECASE)


# ── Hilfsfunktionen ───────────────────────────────────────────────────
def norm_name(name: str) -> str:
    """Normalform für Produktnamen: Großschreibung, Komma→Punkt, ohne
    Leerzeichen. 'WDU 2,5' == 'wdu2.5' == 'WDU 2.5'."""
    return re.sub(r"\s+", "", (name or "").upper().replace(",", "."))


def doc_type_of(filename: str) -> str:
    f = (filename or "").lower()
    if DATASHEET_RE.match(filename or ""):
        return "datenblatt"
    if "preisliste" in f:
        return "preisliste"
    if "katalog" in f or f.startswith("cat"):
        return "katalog"
    if "handbuch" in f or "manual" in f:
        return "handbuch"
    return "sonstiges"


def order_key(chunk_id: str, meta: dict) -> tuple:
    """Reihenfolge innerhalb einer Datei. Die IDs aus save_to_chromadb
    enden auf den globalen Index: <file>_p<page>_<typ>_<cidx>_<idx>."""
    tail = chunk_id.rsplit("_", 1)[-1]
    idx = int(tail) if tail.isdigit() else 0
    try:
        page = int(str(meta.get("page", "0")))
    except ValueError:
        page = 0
    return (page, idx)


def headings_in(text: str) -> list[tuple[int, str]]:
    """(Zeichenposition, Überschrift) aller allein stehenden Überschriften."""
    found = []
    pos = 0
    for line in (text or "").split("\n"):
        key = line.strip().casefold()
        if key in _HEADING_LOOKUP:
            found.append((pos, _HEADING_LOOKUP[key]))
        pos += len(line) + 1
    return found


def clean_line(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip(" -–")


def iter_collection(coll, limit: int | None = None) -> Iterator[tuple[str, str, dict]]:
    """Paginierter Lesezugriff ohne Embeddings (spart RAM und Zeit)."""
    offset = 0
    seen = 0
    while True:
        page = coll.get(limit=PAGE_SIZE, offset=offset, include=["documents", "metadatas"])
        ids = page.get("ids") or []
        if not ids:
            break
        for cid, doc, meta in zip(ids, page.get("documents") or [], page.get("metadatas") or []):
            yield cid, doc or "", dict(meta or {})
            seen += 1
            if limit and seen >= limit:
                return
        offset += len(ids)
        if len(ids) < PAGE_SIZE:
            break


_JUNK_PATTERNS = [
    re.compile(r"\[TABELLE Seite \d+ Nr\.\d+\]"),
    re.compile(r"(?:https?://)?www\.weidmueller\.com", re.IGNORECASE),
    re.compile(r"\bCol\d+\b"),
    re.compile(r"<br\s*/?>", re.IGNORECASE),
    re.compile(r"[|:*]+|-{2,}"),
]
_JUNK_WORDS = {"ausfuehrung", "ausführung", "art", "typ", "col"}


def is_junk(text: str) -> bool:
    """Leere Tabellen-/Text-Fragmente wie '|Zubehör|www.weidmueller.com|'.
    Mit Produkt-Kopf würden sie bei Produktfragen hoch ranken, obwohl sie
    keinerlei Information enthalten – deshalb werden sie nicht indexiert.
    Kurze, aber echte Inhalte ('|Nennstrom|24 A|') bleiben erhalten."""
    t = text or ""
    for rx in _JUNK_PATTERNS:
        t = rx.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t or t.casefold() in _HEADING_LOOKUP:
        return True
    rest = [w for w in t.split() if w.casefold() not in _JUNK_WORDS and w.casefold() not in _HEADING_LOOKUP]
    return len(re.sub(r"[^A-Za-zÄÖÜäöüß0-9]", "", "".join(rest))) < 8


def table_section(text: str, own_artnr: str, page_fallback: str) -> str:
    """Abschnitt einer Tabelle: (1) Überschrift in der ersten Tabellenzeile,
    (2) Zubehör-Tabelle (enthält fremde Artikelnummer + 'Ausfuehrung'),
    (3) erster Abschnitt derselben Seite, (4) 'Tabelle'."""
    first_rows = " ".join((text or "").split("\n")[:3])
    for cell in re.split(r"[|\n]", first_rows):
        key = re.sub(r"\*", "", cell).strip().casefold()
        if key in _HEADING_LOOKUP:
            return _HEADING_LOOKUP[key]
    fremde = [n for n in re.findall(r"\b\d{10}\b", text or "") if n != own_artnr]
    if fremde and re.search(r"Ausf(?:ü|ue)hrung", text or ""):
        return "Zubehör"
    return page_fallback or "Tabelle"


# ── Pass 1: Produkt- und Abschnittswissen sammeln ──────────────────────
class Knowledge:
    def __init__(self) -> None:
        self.page1_text: dict[str, list[tuple[tuple, str]]] = defaultdict(list)
        self.chunk_order: dict[str, list[tuple[tuple, str, list]]] = defaultdict(list)
        self.products: dict[str, dict[str, str]] = {}   # filename -> {artikelnummer, produkt, ausfuehrung}
        self.section_of: dict[str, str] = {}             # chunk_id -> Abschnitt
        self.page_section: dict[tuple, str] = {}         # (datei, seite) -> erster Abschnitt der Seite
        self.junk = 0

    def observe(self, cid: str, text: str, meta: dict, is_table: bool = False) -> None:
        fn = str(meta.get("filename") or meta.get("source") or "")
        key = order_key(cid, meta)
        if DATASHEET_RE.match(fn) and key[0] <= 1:
            self.page1_text[fn].append((key, text))
        if is_table:
            # Tabellen haben eine eigene Indexzählung (eigene Collection) und
            # dürfen die Abschnitts-Vererbung der Fließtext-Chunks nicht stören.
            if is_junk(text):
                self.junk += 1
            return
        self.chunk_order[fn].append((key, cid, headings_in(text)))

    def finalize(self) -> None:
        # (a) Produktidentität je Datenblatt
        for fn, parts in self.page1_text.items():
            artnr = DATASHEET_RE.match(fn).group(1)
            text = "\n".join(t for _, t in sorted(parts))
            m = re.search(_ART_AFTER_NR.format(artnr=re.escape(artnr)), text)
            produkt = clean_line(m.group(1)) if m else ""
            if not produkt:
                m2 = _ART_GENERIC.search(text)
                produkt = clean_line(m2.group(1)) if m2 else ""
            if produkt.isdigit():          # "Art" direkt gefolgt von Nummer → unbrauchbar
                produkt = ""
            m3 = _AUSF.search(text)
            ausf = clean_line(m3.group(1).replace("-\n", "").replace("\n", " ")) if m3 else ""
            self.products[fn] = {"artikelnummer": artnr, "produkt": produkt, "ausfuehrung": ausf[:120]}
        # Datenblätter ohne Seite-1-Chunk: wenigstens die Artikelnummer
        for fn in self.chunk_order:
            m = DATASHEET_RE.match(fn)
            if m and fn not in self.products:
                self.products[fn] = {"artikelnummer": m.group(1), "produkt": "", "ausfuehrung": ""}

        # (b) Abschnitt je Chunk: Überschrift am Chunk-Anfang, sonst die
        #     zuletzt gesehene Überschrift desselben Dokuments (Vererbung).
        for fn, rows in self.chunk_order.items():
            current = ""
            for _, cid, heads in sorted(rows, key=lambda r: r[0]):
                if heads and heads[0][0] < 60:
                    current = heads[0][1]
                self.section_of[cid] = current
                self.page_section.setdefault((fn, _[0]), current)
                if heads:
                    current = heads[-1][1]
        self.page1_text.clear()
        self.chunk_order.clear()

    def lexicon(self) -> dict[str, Any]:
        by_name: dict[str, list[dict]] = defaultdict(list)
        by_artnr: dict[str, dict] = {}
        for fn, p in self.products.items():
            entry = {**p, "document": fn}
            by_artnr[p["artikelnummer"]] = entry
            if p["produkt"]:
                by_name[norm_name(p["produkt"])].append(entry)
        return {
            "version": 1,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "count_products": len(by_artnr),
            "count_names": len(by_name),
            "by_name": dict(by_name),
            "by_artnr": by_artnr,
        }


# ── Kopfzeile + Metadaten ─────────────────────────────────────────────
def enrich(cid: str, text: str, meta: dict, know: Knowledge, is_table: bool) -> tuple[str, dict]:
    fn = str(meta.get("filename") or meta.get("source") or "unbekannt")
    prod = know.products.get(fn, {})
    if is_table:
        try:
            page = int(str(meta.get("page", "0")))
        except ValueError:
            page = 0
        abschnitt = table_section(text, prod.get("artikelnummer", ""),
                                  know.page_section.get((fn, page), ""))
    else:
        abschnitt = know.section_of.get(cid, "")

    parts: list[str] = []
    if prod.get("produkt"):
        parts.append(f"Produkt: {prod['produkt']}")
    if prod.get("artikelnummer"):
        parts.append(f"Art.-Nr.: {prod['artikelnummer']}")
    if prod.get("ausfuehrung") and abschnitt in ("", "Allgemeine Bestelldaten", "Produktbeschreibung"):
        parts.append(prod["ausfuehrung"][:80])
    if not prod:
        parts.append(f"Dokument: {fn}")
        parts.append(f"S. {meta.get('page', '?')}")
    if abschnitt:
        parts.append(f"Abschnitt: {abschnitt}")
    header = "[" + " | ".join(parts) + "]"
    new_text = f"{header}\n{text.strip()}"

    new_meta = {k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))}
    new_meta.update({
        "document": fn,
        "filename": fn,
        "source_type": "table" if is_table else str(meta.get("type", "text")),
        "doc_typ": doc_type_of(fn),
        "artikelnummer": prod.get("artikelnummer", ""),
        "produkt": prod.get("produkt", ""),
        "ausfuehrung": prod.get("ausfuehrung", ""),
        "abschnitt": abschnitt,
        "manufacturer": MANUFACTURER,
        "chunk_id": cid,
        "index_version": "v3",
    })
    return new_text, new_meta


# ── Einbettung ───────────────────────────────────────────────────────
def load_model():
    from sentence_transformers import SentenceTransformer
    import torch
    name = os.environ.get("MODEL_NAME", "BAAI/bge-m3")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(name, device=device)
    model.max_seq_length = 512
    if device == "cuda":
        model.half()   # halbiert VRAM, Qualität praktisch identisch
    log.info("Embedding-Modell %s auf %s geladen.", name, device)
    return model


def embed(model, texts: list[str]) -> list[list[float]]:
    vecs = model.encode(texts, batch_size=EMBED_BATCH, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False)
    return vecs.tolist()


# ── Hauptablauf ──────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Nur analysieren, nichts schreiben.")
    ap.add_argument("--limit", type=int, default=None, help="Nur die ersten N Chunks je Collection.")
    ap.add_argument("--skip-embedding", action="store_true",
                    help="Nur Lexikon + BM25-Datei schreiben (keine Chroma-Collection).")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from save_to_chromadb import get_chroma_client, get_or_create_collection

    client = get_chroma_client()
    src = [(client.get_collection(SRC_TEXT), DST_TEXT, False),
           (client.get_collection(SRC_TAG), DST_TAG, True)]

    # Pass 1
    know = Knowledge()
    t0 = time.time()
    for coll, _, is_tab in src:
        n = 0
        for cid, text, meta in iter_collection(coll, args.limit):
            know.observe(cid, text, meta, is_tab)
            n += 1
            if n % 50_000 == 0:
                log.info("Pass 1 %s: %d Chunks gelesen", coll.name, n)
        log.info("Pass 1 %s fertig: %d Chunks.", coll.name, n)
    know.finalize()
    lex = know.lexicon()
    named = sum(1 for p in know.products.values() if p["produkt"])
    log.info("Pass 1 fertig in %.0fs: %d Datenblätter, davon %d mit Produktname (%.1f %%), %d Namen im Lexikon.",
             time.time() - t0, len(know.products), named,
             100 * named / max(len(know.products), 1), lex["count_names"])

    if args.dry_run:
        probe = os.environ.get("REINDEX_PROBE_FILE", "1020000000_de.pdf")
        print(f"\n=== Beispiel-Köpfe für {probe} (Dry-Run) ===")
        for coll, _, is_table in src:
            res = coll.get(where={"filename": probe}, limit=400, include=["documents", "metadatas"])
            rows = sorted(zip(res["ids"], res["documents"], res["metadatas"]),
                          key=lambda r: order_key(r[0], r[2] or {}))
            for i, (cid, text, meta) in enumerate(rows):
                if is_junk(text or ""):
                    continue
                new_text, _m = enrich(cid, text or "", dict(meta or {}), know, is_table)
                if i % 6 == 0:
                    print(f"\n{cid}\n  {new_text[:220]!r}")
        print("\n=== Lexikon-Test ===")
        for name in ("WDU 2.5", "WDU 2.5 GN", "A2C 2.5", "PRO ECO3 240W 24V 10A II", "UR20-FBC-PN-IRT-V2"):
            hits = lex["by_name"].get(norm_name(name), [])
            print(f"  {name:28s} -> {[e['artikelnummer'] for e in hits] or 'NICHT GEFUNDEN'}")
        print("\n=== ZUSAMMENFASSUNG ===")
        print(f"  Datenblätter:            {len(know.products)}")
        print(f"  davon mit Produktname:   {named} ({100 * named / max(len(know.products), 1):.1f} %)")
        print(f"  Namen im Lexikon:        {lex['count_names']}")
        print(f"  leere Tabellen (werden übersprungen): {know.junk}")
        return

    Path(LEXICON_OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(LEXICON_OUT, "w", encoding="utf-8") as fh:
        json.dump(lex, fh, ensure_ascii=False)
    log.info("Produkt-Lexikon geschrieben: %s", LEXICON_OUT)

    # Pass 2
    model = None if args.skip_embedding else load_model()
    Path(BM25_OUT).parent.mkdir(parents=True, exist_ok=True)
    bm25_tmp = BM25_OUT + ".tmp"
    written = embedded = skipped = 0
    t1 = time.time()
    with open(bm25_tmp, "w", encoding="utf-8") as bm25_fh:
        for coll, dst_name, is_table in src:
            dst = None if args.skip_embedding else get_or_create_collection(client, dst_name)
            buf_ids: list[str] = []
            buf_txt: list[str] = []
            buf_meta: list[dict] = []

            def flush() -> None:
                nonlocal embedded, skipped
                if not buf_ids or dst is None:
                    buf_ids.clear(); buf_txt.clear(); buf_meta.clear()
                    return
                existing = set(dst.get(ids=list(buf_ids), include=[]).get("ids") or [])
                todo = [i for i, cid in enumerate(buf_ids) if cid not in existing]
                skipped += len(buf_ids) - len(todo)
                if todo:
                    ids = [buf_ids[i] for i in todo]
                    txt = [buf_txt[i] for i in todo]
                    met = [buf_meta[i] for i in todo]
                    dst.upsert(ids=ids, documents=txt, metadatas=met, embeddings=embed(model, txt))
                    embedded += len(ids)
                buf_ids.clear(); buf_txt.clear(); buf_meta.clear()

            for cid, text, meta in iter_collection(coll, args.limit):
                if not text.strip() or is_junk(text):
                    continue
                new_text, new_meta = enrich(cid, text, meta, know, is_table)
                bm25_fh.write(json.dumps({"chunk_id": cid, "text": new_text, "metadata": new_meta},
                                         ensure_ascii=False) + "\n")
                written += 1
                buf_ids.append(cid); buf_txt.append(new_text); buf_meta.append(new_meta)
                if len(buf_ids) >= 512:
                    flush()
                if written % 20_000 == 0:
                    rate = embedded / max(time.time() - t1, 1)
                    log.info("Pass 2: %d geschrieben, %d eingebettet, %d übersprungen (%.0f Chunks/s)",
                             written, embedded, skipped, rate)
            flush()
    os.replace(bm25_tmp, BM25_OUT)
    log.info("FERTIG in %.0f min: %d Chunks, %d neu eingebettet, %d bereits vorhanden.",
             (time.time() - t1) / 60, written, embedded, skipped)
    log.info("BM25-Datei: %s  (aktivieren: alte bm25_index.jsonl sichern, neue umbenennen, API neu starten)", BM25_OUT)


if __name__ == "__main__":
    main()