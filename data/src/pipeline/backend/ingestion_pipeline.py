"""ingestion_pipeline — Fassade der Ingestion-Schicht (NF-10).

UC-05 (PDF importieren), UC-06 (Dokumente verarbeiten), UC-10 (Embeddings
speichern). PDF-Parsing und OCR liegen in den Pipeline-Skripten
(extract_text.py, ocr_fallback.py). Dieses Modul stellt die Funktionen
für konsistente Chunk-IDs zwischen BM25-Index und ChromaDB bereit.
"""
from backend.petra_hybrid.ingestion_hook import (
    build_chunk_ids, bump_corpus_version, index_chunks_hybrid,
    index_table_chunks_hybrid, verify_index_consistency,
)
from backend.petra_hybrid.table_serializer import TableChunk, serialize_table

__all__ = [
    "build_chunk_ids", "index_chunks_hybrid", "index_table_chunks_hybrid",
    "bump_corpus_version", "verify_index_consistency", "serialize_table",
    "TableChunk",
]
