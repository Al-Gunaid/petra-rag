#!/usr/bin/env python3
# ============================================================
# backfill_bm25_from_chroma.py — Nachtrag fehlender BM25-Chunks aus ChromaDB
#
# Einmaliges Wartungsskript: Liest beide Collections paginiert und
# registriert alle Chunks mit ihren ChromaDB-IDs im BM25-Index.
# Bereits vorhandene IDs werden ersetzt (Upsert).
#
# BM25Index.add_or_replace() baut bei jedem Aufruf den gesamten Index
# neu auf und schreibt die Datei vollständig. Daher wird in großen
# Batches (BATCH_SIZE) geschrieben statt je Datei oder Chunk.
#
# Ausführung im Ingestion-Container:
#   docker exec -it petra_ingestion python3 \
#       /app/src/pipeline/backfill_bm25_from_chroma.py
# ============================================================
from __future__ import annotations

import logging
import sys
import time

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] backfill_bm25 – %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

# Grosse Batches: reduziert die Anzahl teurer Voll-Rebuilds.
# Bei ~640k Chunks gesamt ergibt das ca. 6-7 Rebuild-Durchlaeufe
# statt 26821 (einer pro Datei) oder gar 639411 (einer pro Chunk).
BATCH_SIZE = 100_000
CHROMA_PAGE_SIZE = 5_000


def backfill_collection(collection, collection_name: str, index_chunks_hybrid) -> tuple[int, int]:
    """Überträgt eine Collection paginiert in den BM25-Index.

    Leere Dokumente werden übersprungen.

    Returns:
        (gelesene Chunks, neu im BM25-Index)"""
    total = collection.count()
    log.info("Collection '%s': %d Chunks in ChromaDB, starte Nachtrag...", collection_name, total)

    pending_ids: list[str] = []
    pending_texts: list[str] = []
    pending_metas: list[dict] = []

    read_total = 0
    new_total = 0
    offset = 0

    def flush() -> None:
        """Schreibt die gesammelten Chunks in den BM25-Index und leert den Puffer."""
        nonlocal pending_ids, pending_texts, pending_metas, new_total
        if not pending_ids:
            return
        t0 = time.time()
        n_new = index_chunks_hybrid(pending_ids, pending_texts, pending_metas)
        new_total += n_new
        log.info(
            "Batch geschrieben: %d Chunks uebergeben (%d davon neu) in %.1fs.",
            len(pending_ids), n_new, time.time() - t0,
        )
        pending_ids = []
        pending_texts = []
        pending_metas = []

    while True:
        page = collection.get(limit=CHROMA_PAGE_SIZE, offset=offset, include=["documents", "metadatas"])
        ids = page.get("ids", [])
        if not ids:
            break

        docs = page.get("documents", [])
        metas = page.get("metadatas", [])

        for cid, doc, meta in zip(ids, docs, metas):
            if not doc or not str(doc).strip():
                continue
            pending_ids.append(cid)
            pending_texts.append(doc)
            pending_metas.append(meta or {})

        read_total += len(ids)
        offset += len(ids)

        if len(pending_ids) >= BATCH_SIZE:
            flush()
        # Letzte Seite erreicht
        if len(ids) < CHROMA_PAGE_SIZE:
            break
        # Fortschritt etwa alle 50.000 Chunks
        if read_total % 50_000 < CHROMA_PAGE_SIZE:
            log.info("Fortschritt '%s': %d/%d gelesen.", collection_name, read_total, total)

    flush()
    log.info("Collection '%s' fertig: %d gelesen, %d davon neu im BM25-Index.", collection_name, read_total, new_total)
    return read_total, new_total


def main() -> None:
    """Überträgt Text- und Tabellen-Collection in den BM25-Index."""
    # Import erst zur Laufzeit (benötigt PYTHONPATH des Ingestion-Containers).
    from backend.petra_hybrid.ingestion_hook import index_chunks_hybrid
    from save_to_chromadb import (
        COLLECTION_TAG,
        COLLECTION_TEXT,
        get_chroma_client,
        get_or_create_collection,
    )

    client = get_chroma_client()
    text_collection = get_or_create_collection(client, COLLECTION_TEXT)
    tag_collection = get_or_create_collection(client, COLLECTION_TAG)

    total_read = 0
    total_new = 0

    for name, coll in [(COLLECTION_TEXT, text_collection), (COLLECTION_TAG, tag_collection)]:
        r, n = backfill_collection(coll, name, index_chunks_hybrid)
        total_read += r
        total_new += n

    log.info(
        "FERTIG. Insgesamt %d Chunks aus ChromaDB gelesen, %d neue Chunks in den BM25-Index nachgetragen.",
        total_read, total_new,
    )


if __name__ == "__main__":
    main()
