#!/usr/bin/env python3
# ============================================================
# v3_sync.py — neue Importe auch in den kontextuellen v3-Index schreiben
#
# PROBLEM (04.10.2026): Seit Schritt 1 liest der Workflow die Collections
#   petra_text_chunks_v3 / petra_tag_chunks_v3 (Chunks mit Kontextkopf
#   „[Produkt | Art.-Nr. | Abschnitt]“), das Produkt-Retrieval (s2) liest das
#   Produkt-Lexikon. Beides entstand EINMALIG mit reindex_v3_contextual.py.
#   save_to_chromadb.py schreibt neue PDFs aber nur in die v2-Collections und
#   ohne Kopf in den BM25-Index → neue Datenblätter fehlen im Dense-, TAG- und
#   Produktpfad.
#
# LÖSUNG: Nach jedem Speichern wird dieselbe Anreicherung wie bei der
#   Re-Indexierung für GENAU DIESE Datei ausgeführt (gleiche Funktionen aus
#   reindex_v3_contextual.py, also identische Köpfe und Abschnitte):
#     1. Chunks der Datei aus den v2-Collections lesen
#     2. Produktname, Artikelnummer, Abschnitte ableiten, Kopf voranstellen
#        (leere Tabellen-Fragmente werden wie beim Reindex übersprungen)
#     3. alte v3-Einträge der Datei löschen, neu einbetten (bge-m3), upserten
#     4. Produkt-Lexikon um das Datenblatt ergänzen (wird vom Backend bei
#        Dateiänderung automatisch neu geladen)
#     5. Texte mit Kopf an den BM25-Index zurückgeben (save_to_chromadb.py
#        schreibt sie statt der Texte ohne Kopf)
#
# NACHHOLEN für schon importierte Dateien (im Container petra_ingestion):
#   python3 v3_sync.py --fehlende            # alle Dateien, die in v3 fehlen
#   python3 v3_sync.py 1234567890_de.pdf …   # bestimmte Dateien
#   danach: docker restart petra_ingestion   (BM25-Index im Speicher neu laden)
# ============================================================
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("v3_sync")

DST_TEXT = os.environ.get("PETRA_COLLECTION_TEXT_V3", "petra_text_chunks_v3")
DST_TAG = os.environ.get("PETRA_COLLECTION_TAG_V3", "petra_tag_chunks_v3")
LEXICON = Path(os.environ.get("PETRA_PRODUCT_LEXICON", "/data/product_lexicon.json"))
SEITE = 2000      # Lesen je Abruf
UPSERT = 256      # Einbetten/Schreiben je Abruf


def _alle(coll, where: dict | None = None):
    """Paginierter Lesezugriff (Dokumente + Metadaten, ohne Embeddings)."""
    offset = 0
    while True:
        kw = {"limit": SEITE, "offset": offset, "include": ["documents", "metadatas"]}
        if where:
            kw["where"] = where
        page = coll.get(**kw)
        ids = page.get("ids") or []
        if not ids:
            return
        for cid, doc, meta in zip(ids, page.get("documents") or [], page.get("metadatas") or []):
            yield cid, doc or "", dict(meta or {})
        offset += len(ids)
        if len(ids) < SEITE:
            return


def _lexikon_ergaenzen(produkte: dict[str, dict[str, str]], norm_name) -> int:
    """Trägt die Datenblätter dieser Datei ins Lexikon ein (by_artnr ersetzen, by_name neu aufbauen)."""
    if not produkte:
        return 0
    try:
        lex = json.loads(LEXICON.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        lex = {"version": 1, "by_name": {}, "by_artnr": {}}
    by_artnr = lex.get("by_artnr") or {}
    for fn, p in produkte.items():
        by_artnr[p["artikelnummer"]] = {**p, "document": fn}
    by_name: dict[str, list[dict]] = {}
    for e in by_artnr.values():
        if e.get("produkt"):
            by_name.setdefault(norm_name(e["produkt"]), []).append(e)
    lex.update({"by_artnr": by_artnr, "by_name": by_name, "count_products": len(by_artnr),
                "count_names": len(by_name), "updated": time.strftime("%Y-%m-%dT%H:%M:%S")})
    LEXICON.parent.mkdir(parents=True, exist_ok=True)
    tmp = LEXICON.with_suffix(".tmp")
    tmp.write_text(json.dumps(lex, ensure_ascii=False), encoding="utf-8")
    tmp.replace(LEXICON)
    return len(produkte)


def sync_datei(filename: str, model=None, device: str | None = None) -> dict[str, Any]:
    """Schreibt die Chunks einer Datei mit Kontextkopf in die v3-Collections.

    Rückgabe: ids/docs/metas (für den BM25-Index) und Zähler.
    """
    from reindex_v3_contextual import Knowledge, enrich, is_junk, norm_name
    from save_to_chromadb import (COLLECTION_TAG, COLLECTION_TEXT, compute_embeddings,
                                  get_chroma_client, get_embedding_model, get_or_create_collection)

    if model is None:
        model, device = get_embedding_model()
    client = get_chroma_client()
    quellen = [(client.get_collection(COLLECTION_TEXT), DST_TEXT, False),
               (client.get_collection(COLLECTION_TAG), DST_TAG, True)]

    # Pass 1: alle Chunks der Datei lesen, Produkt- und Abschnittswissen ableiten
    rohdaten: list[tuple[str, str, dict, bool, str]] = []
    know = Knowledge()
    for coll, ziel, is_table in quellen:
        for cid, text, meta in _alle(coll, {"filename": filename}):
            know.observe(cid, text, meta, is_table)
            rohdaten.append((cid, text, meta, is_table, ziel))
    know.finalize()

    # Pass 2: Kopf voranstellen, einbetten, schreiben
    ergebnis: dict[str, Any] = {"ids": [], "docs": [], "metas": [], "v3_text": 0, "v3_tag": 0,
                                "uebersprungen": 0, "produkt": know.products.get(filename, {}).get("produkt", "")}
    for ziel in (DST_TEXT, DST_TAG):
        dst = get_or_create_collection(client, ziel)
        try:
            dst.delete(where={"filename": filename})     # Neuimport: alte Fassung entfernen
        except Exception as exc:  # noqa: BLE001
            log.warning("v3: Löschen alter Einträge von %s in %s fehlgeschlagen: %s", filename, ziel, exc)
        zeilen = [r for r in rohdaten if r[4] == ziel]
        puffer: list[tuple[str, str, dict]] = []

        def schreiben() -> None:
            if not puffer:
                return
            ids = [p[0] for p in puffer]
            txt = [p[1] for p in puffer]
            met = [p[2] for p in puffer]
            dst.upsert(ids=ids, documents=txt, metadatas=met, embeddings=compute_embeddings(model, txt, device))
            ergebnis["ids"].extend(ids)
            ergebnis["docs"].extend(txt)
            ergebnis["metas"].extend(met)
            ergebnis["v3_tag" if ziel == DST_TAG else "v3_text"] += len(ids)
            puffer.clear()

        for cid, text, meta, is_table, _ in zeilen:
            if not text.strip() or is_junk(text):
                ergebnis["uebersprungen"] += 1
                continue
            neu_text, neu_meta = enrich(cid, text, meta, know, is_table)
            puffer.append((cid, neu_text, neu_meta))
            if len(puffer) >= UPSERT:
                schreiben()
        schreiben()

    ergebnis["lexikon"] = _lexikon_ergaenzen(know.products, norm_name)
    log.info("v3-Sync %s: %d Text, %d Tabellen, %d übersprungen, Produkt '%s'",
             filename, ergebnis["v3_text"], ergebnis["v3_tag"], ergebnis["uebersprungen"], ergebnis["produkt"])
    return ergebnis


def _dateien(coll) -> set[str]:
    namen: set[str] = set()
    offset = 0
    while True:
        page = coll.get(limit=5000, offset=offset, include=["metadatas"])
        metas = page.get("metadatas") or []
        if not metas:
            return namen
        namen.update(str(m.get("filename") or m.get("source") or "") for m in metas if m)
        offset += len(metas)
        if len(metas) < 5000:
            return namen


def main() -> None:
    logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] v3_sync – %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("dateien", nargs="*", help="Dateinamen (wie in den Metadaten, z. B. 1020000000_de.pdf)")
    ap.add_argument("--fehlende", action="store_true", help="alle Dateien nachholen, die in v3 fehlen")
    ap.add_argument("--nur-liste", action="store_true", help="nur anzeigen, was fehlt")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from save_to_chromadb import (BM25_INDEX_PATH, COLLECTION_TAG, COLLECTION_TEXT, get_chroma_client,
                                  get_or_create_collection)

    dateien = list(a.dateien)
    if a.fehlende:
        c = get_chroma_client()
        alt = _dateien(c.get_collection(COLLECTION_TEXT)) | _dateien(c.get_collection(COLLECTION_TAG))
        neu = _dateien(get_or_create_collection(c, DST_TEXT)) | _dateien(get_or_create_collection(c, DST_TAG))
        fehlend = sorted(x for x in alt - neu if x)
        print(f"In v2: {len(alt)} Dateien, in v3: {len(neu)}, fehlend in v3: {len(fehlend)}")
        for x in fehlend[:50]:
            print("  ", x)
        if a.nur_liste:
            return
        dateien += fehlend
    if not dateien:
        print("Nichts zu tun.")
        return
    from backend.petra_hybrid.ingestion_hook import index_chunks_hybrid
    ids, docs, metas = [], [], []
    for i, fn in enumerate(dateien, 1):
        r = sync_datei(fn)
        ids += r["ids"]; docs += r["docs"]; metas += r["metas"]
        print(f"[{i}/{len(dateien)}] {fn}: {r['v3_text']} Text, {r['v3_tag']} Tabellen, Produkt '{r['produkt']}'")
    if ids:
        n = index_chunks_hybrid(ids, docs, metas, bm25_index_path=BM25_INDEX_PATH)
        print(f"BM25-Index ({BM25_INDEX_PATH}): {n} Chunks mit Kontextkopf übernommen.")
    print("Fertig. Jetzt: docker restart petra_ingestion (BM25-Index im Speicher neu laden).")


if __name__ == "__main__":
    main()
