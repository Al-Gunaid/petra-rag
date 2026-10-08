# ============================================================
# petra_hybrid/ingestion_hook.py — Anbindung der Ingestion (UC-06)
#
# Stellt sicher, dass BM25-Index und ChromaDB denselben Dokumentbestand
# mit identischen chunk_ids führen. Ohne ID-Gleichheit kann RRF keine
# Ränge kombinieren ([3a]).
#
# Verwendung in der Ingestion:
#
#   from backend.petra_hybrid.ingestion_hook import (
#       build_chunk_ids, index_chunks_hybrid, index_table_chunks_hybrid,
#   )
#   from backend.petra_hybrid.table_serializer import serialize_tables_from_pymupdf
#
#   chunk_ids = build_chunk_ids(filename, pages)
#   chroma_collection.add(ids=chunk_ids, documents=texts, metadatas=metas, embeddings=embeddings)
#   index_chunks_hybrid(chunk_ids, texts, metas)
#
#   table_chunks = serialize_tables_from_pymupdf(page.find_tables().tables, document=filename, page=page_no)
#   index_table_chunks_hybrid(table_chunks, chroma_collection.add, embedding_fn)
#
#   bump_corpus_version(<Hash über den Bestand>)   # invalidiert den CAG-Cache
# ============================================================
from __future__ import annotations

import logging

from backend.petra_hybrid.bm25_index import BM25Chunk, get_bm25_index
from backend.petra_hybrid.cache import get_cache
from backend.petra_hybrid.chunk_id import make_chunk_id
from backend.petra_hybrid.table_serializer import TableChunk

logger = logging.getLogger("petra.ingestion_hook")


def build_chunk_ids(
    document: str,
    pages: list[int | None],
    source_type: str = "text",
) -> list[str]:
    """Kanonische Chunk-IDs eines Dokuments (identisch für ChromaDB und BM25).

    Args:
        document: Dokumentname.
        pages: Seitenzahl je Chunk; der Listenindex ist die Position im Dokument.
        source_type: "text" oder "table".
    """
    return [
        make_chunk_id(document, page, index, source_type)
        for index, page in enumerate(pages)
    ]


def bump_corpus_version(version: str) -> int:
    """Setzt eine neue Korpusversion und verwirft damit den CAG-Cache.

    Nach jedem Import aufrufen (Kap. 4.6.2, Failure Case 4). Geeignet ist
    ein Hash über alle Dateinamen und Inhaltshashes.

    Returns:
        Anzahl verworfener Cache-Einträge.
    """
    dropped = get_cache().set_corpus_version(version)
    if dropped:
        logger.info("CAG-Cache invalidiert: %d Einträge nach Korpusänderung verworfen.", dropped)
    return dropped


def index_chunks_hybrid(
    chunk_ids: list[str],
    texts: list[str],
    metadatas: list[dict],
    bm25_index_path: str = "data/bm25_index.jsonl",
) -> int:
    """Registriert die in ChromaDB geschriebenen Text-Chunks im BM25-Index.

    Muss mit denselben chunk_ids wie ChromaDB `add()` aufgerufen werden.

    Raises:
        ValueError: bei unterschiedlich langen Listen.

    Returns:
        Anzahl neu hinzugefügter Chunks.
    """
    if not (len(chunk_ids) == len(texts) == len(metadatas)):
        raise ValueError("chunk_ids, texts und metadatas müssen gleich lang sein.")
    index = get_bm25_index(bm25_index_path)
    n_new = index.add_or_replace(
        BM25Chunk(chunk_id=cid, text=t, metadata=m) for cid, t, m in zip(chunk_ids, texts, metadatas)
    )
    logger.info("BM25-Index aktualisiert: %d neue Chunks (gesamt aufgerufen: %d).", n_new, len(chunk_ids))
    return n_new


def index_table_chunks_hybrid(
    table_chunks: list[TableChunk],
    chroma_add_fn,
    embedding_fn,
    bm25_index_path: str = "data/bm25_index.jsonl",
) -> int:
    """Schreibt Tabellen-Chunks (Partial TAG) mit identischen IDs in ChromaDB und BM25.

    Args:
        table_chunks: Ausgabe von serialize_table(s).
        chroma_add_fn: Callable(ids, documents, metadatas, embeddings),
            z. B. ``collection.add``.
        embedding_fn: Callable(list[str]) -> list[Vektor]; muss dasselbe
            Modell wie für Text-Chunks und Anfragen verwenden.

    Returns:
        Anzahl neu im BM25-Index hinzugefügter Chunks.
    """
    if not table_chunks:
        return 0

    ids = [c.chunk_id for c in table_chunks]
    texts = [c.text for c in table_chunks]
    metas = [c.metadata for c in table_chunks]
    embeddings = embedding_fn(texts)

    chroma_add_fn(ids=ids, documents=texts, metadatas=metas, embeddings=embeddings)
    index = get_bm25_index(bm25_index_path)
    n_new = index.add_or_replace(BM25Chunk(chunk_id=cid, text=t, metadata=m) for cid, t, m in zip(ids, texts, metas))
    logger.info("Partial-TAG: %d Tabellenzeilen in ChromaDB + BM25 indiziert.", len(table_chunks))
    return n_new


def verify_index_consistency(
    chroma_ids: list[str],
    bm25_index_path: str = "data/bm25_index.jsonl",
) -> dict:
    """Prüft nach der Ingestion, ob ChromaDB und BM25 dieselben IDs führen.

    Gehört in die Ingestion: Im Query-Pfad zeigt sich eine Inkonsistenz
    nur als schleichend schlechtere Antwortqualität.

    Returns:
        Bericht mit Anzahlen, Überschneidung, consistency_ratio und
        ok (ratio >= 0.99).
    """
    index = get_bm25_index(bm25_index_path)
    bm25_ids = set(index._chunk_id_to_pos)  # noqa: SLF001 - Diagnosezugriff
    chroma = set(chroma_ids)
    overlap = len(chroma & bm25_ids)
    smaller = min(len(chroma), len(bm25_ids)) or 1
    ratio = overlap / smaller
    report = {
        "chroma_ids": len(chroma),
        "bm25_ids": len(bm25_ids),
        "overlap": overlap,
        "consistency_ratio": round(ratio, 4),
        "ok": ratio >= 0.99,
    }
    if not report["ok"]:
        logger.error(
            "ID-INKONSISTENZ: nur %.1f%% der Chunk-IDs stimmen zwischen ChromaDB und "
            "BM25 überein. RRF kann keine Ränge kombinieren. Beide Indizes müssen mit "
            "build_chunk_ids() befüllt werden.", ratio * 100,
        )
    return report
