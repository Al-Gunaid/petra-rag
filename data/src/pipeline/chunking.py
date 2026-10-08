#!/usr/bin/env python3
# ============================================================
# chunking.py — Node 4: Chunking von Fließtext (LangChain)
#
# Teilt Fließtext in Chunks zu 512 Zeichen mit 50 Zeichen Overlap.
# Tabellen- und OCR-Blöcke bleiben ungeteilt (TAG-Prinzip: die
# Zeilen-Spalten-Zuordnung ginge sonst verloren).
#
# Verwendung: Import durch api_server.py (POST /chunk) oder CLI.
#   Eingabe: JSON über stdin {"chunks": [...], "filename": "..."}
#   Ausgabe: JSON über stdout; Logging über stderr.
#
# Warum 512 Token, 50 Overlap: Spez-Vorgabe
#   Zu klein = Kontext verloren; zu groß = LLM überlastet.
#   50 Overlap verhindert Informationsverlust an Chunk-Grenzen. 
# ============================================================

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Logging auf stderr ────────────────────────────────────────
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] chunking – %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Chunking-Parameter (laut Spezifikation FA-ING-06) ────────
CHUNK_SIZE:    int = 512  # Maximale Chunk-Größe in Zeichen
CHUNK_OVERLAP: int = 50   # Überlappung zwischen Chunks

# Chunk-Typen die nicht gesplittet werden dürfen (TAG + OCR)
NO_SPLIT_TYPES: frozenset[str] = frozenset({"table", "ocr"})


def build_splitter() -> RecursiveCharacterTextSplitter:
    """
    Erstellt den LangChain TextSplitter.

    Recursive bedeutet: versucht bei Absätzen zu trennen, dann
    bei Sätzen, dann Wörtern — nie mitten in einem Wort.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )


def process_chunks(
    blocks: list[dict[str, Any]],
    splitter: RecursiveCharacterTextSplitter,
) -> list[dict[str, Any]]:
    """
    Verarbeitet alle Blöcke: splittet Text, übernimmt Tabellen/OCR unverändert.

    Args:
        blocks:   Rohe Chunks aus extract_text / ocr_fallback.
        splitter: Konfigurierter LangChain TextSplitter.

    Returns:
        Finale Chunk-Liste mit chunk_index für Sub-Chunks.
    """
    final: list[dict[str, Any]] = []
    stats = {"text_split": 0, "preserved": 0, "sub_chunks": 0}

    for block in blocks:
        chunk_type = block.get("type", "text")

        # Tabellen und OCR werden direkt übernommen (TAG-Prinzip)
        if chunk_type in NO_SPLIT_TYPES:
            final.append({**block, "chunk_index": 0})
            stats["preserved"] += 1
            continue

        # Fliesstext aufteilen
        text = block.get("text", "")
        if not text.strip():
            log.debug("Leerer Block übersprungen (Seite %s)", block.get("page"))
            continue

        splits = splitter.split_text(text)
        for i, split_text in enumerate(splits):
            final.append({
                "text":        split_text,
                "page":        block["page"],
                "source":      block["source"],
                "type":        "text",
                "chunk_index": i,
            })
        stats["text_split"] += 1
        stats["sub_chunks"] += len(splits)

    log.info(
        "Chunking: %d Text-Blöcke gesplittet (%d Sub-Chunks) | "
        "%d Blöcke unverändert | %d Chunks total",
        stats["text_split"], stats["sub_chunks"],
        stats["preserved"], len(final),
    )
    return final


def main() -> None:
    # JSON von stdin lesen
    raw = sys.stdin.read()

    if not raw.strip():
        log.error("Keine JSON-Daten über stdin erhalten")
        sys.exit(1)

    try:
        data = json.loads(raw)


        if isinstance(data, str):
            data = json.loads(data) # doppelt kodiertes JSON
            
    except json.JSONDecodeError as exc:
        log.error("Ungültiger JSON-Input: %s", exc)
        sys.exit(1)

    blocks: list[dict[str, Any]] = data.get("chunks", [])
    filename: str = data.get("filename", "unknown")

    if not blocks:
        log.warning("Keine Chunks in der Eingabe für '%s'", filename)

    splitter = build_splitter()
    final = process_chunks(blocks, splitter)

    print(json.dumps({
        "filename": filename,
        "total_chunks": len(final),
        "chunks": final,
    }, ensure_ascii=False))




if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log.critical("Unerwarteter Fehler: %s", exc, exc_info=True)
        sys.exit(1)
