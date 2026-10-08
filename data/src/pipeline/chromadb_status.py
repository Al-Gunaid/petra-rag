#!/usr/bin/env python3
"""chromadb_status.py — Chunk-Statistik je Quelldatei aus ChromaDB.

Liest Text- und Tabellen-Collection seitenweise (nur Metadaten) und
aggregiert Chunks, Typen und Seiten je Datei. Aufruf über GET /status und
GET /export-report (api_server.py).

Ausgabe: JSON über stdout; Logging über stderr.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from collections import defaultdict
from typing import Any

import chromadb

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] chromadb_status – %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

CHROMA_HOST = os.environ.get("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8000"))
COLLECTION_TEXT = os.environ.get("CHROMA_COLLECTION_TEXT", "petra_text_chunks")
COLLECTION_TAG = os.environ.get("CHROMA_COLLECTION_TAG", "petra_tag_chunks")
BATCH_SIZE = int(os.environ.get("STATUS_BATCH_SIZE", "1000"))


def get_status() -> dict[str, Any]:
    """
    Aggregiert die Chunk-Statistik je PDF über beide Collections.

    Paginierung mit limit/offset vermeidet Timeouts und den
    SQLite-Fehler "too many SQL variables" bei großen Collections.

    Returns:
        Dict mit gesamt_chunks, gesamt_dateien, dateien (je Datei:
        datei, chunks_gesamt, text_chunks, tabellen, ocr_chunks, seiten,
        status), collections und batch_size.

    Raises:
        Exception: Lesefehler einer Collection werden protokolliert und
            weitergereicht.
    """
    client = chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT,
    )

    batch_size = int(os.environ.get("STATUS_BATCH_SIZE", "500"))

    stats: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "total": 0,
        "text": 0,
        "table": 0,
        "ocr": 0,
        "pages": set(),
    })

    total_chunks = 0
    collection_counts: dict[str, int] = {}

    for collection_name in [COLLECTION_TEXT, COLLECTION_TAG]:
        try:
            collection = client.get_collection(collection_name)
            count = collection.count()

            collection_counts[collection_name] = count
            total_chunks += count

            log.info(
                "Lese Collection '%s': %d Chunks",
                collection_name,
                count,
            )

            for offset in range(0, count, batch_size):
                log.info(
                    "Collection '%s': lese %d bis %d",
                    collection_name,
                    offset,
                    min(offset + batch_size, count),
                )

                result = collection.get(
                    include=["metadatas"],
                    limit=batch_size,
                    offset=offset,
                )

                metadatas = result.get("metadatas") or []

                for meta in metadatas:
                    filename = (
                        meta.get("filename")
                        or meta.get("source")
                        or "unknown"
                    )

                    typ = str(meta.get("type", "text"))
                    page = str(meta.get("page", "0"))

                    entry = stats[filename]
                    entry["total"] += 1
                    entry["pages"].add(page)

                    if typ == "table":
                        entry["table"] += 1
                    elif typ == "ocr":
                        entry["ocr"] += 1
                    else:
                        entry["text"] += 1

        except Exception as exc:
            log.exception(
                "Fehler beim Lesen der Collection '%s': %s",
                collection_name,
                exc,
            )
            raise

    files_report: list[dict[str, Any]] = []

    for filename, values in sorted(stats.items()):
        files_report.append({
            "datei": filename,
            "chunks_gesamt": values["total"],
            "text_chunks": values["text"],
            "tabellen": values["table"],
            "ocr_chunks": values["ocr"],
            "seiten": len(values["pages"]),
            "status": "✅ OK" if values["total"] > 0 else "❌ Leer",
        })

    log.info(
        "Status fertig: %d Chunks | %d Dateien",
        total_chunks,
        len(files_report),
    )

    return {
        "gesamt_chunks": total_chunks,
        "gesamt_dateien": len(files_report),
        "dateien": files_report,
        "collections": collection_counts,
        "batch_size": batch_size,
    }


def main() -> None:
    print(json.dumps(get_status(), ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log.critical("Unerwarteter Fehler: %s", exc, exc_info=True)
        sys.exit(1)