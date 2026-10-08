"""retrieval_engine — Fassade der Retrieval-Schicht (Modulstruktur nach NF-10).

Die Implementierung liegt im Paket ``petra_hybrid``; dieses Modul stellt
die nach NF-10 geforderte Schnittstelle bereit.

Umfang: BM25 Sparse Retrieval [2a], Dense-Score-Normalisierung,
RRF-Fusion [3a], CRAG-Score-Filter [3b], Cross-Encoder-Reranking [3c],
Chunk-IDs und getrenntes Retrieval je Produkt (UC-02).
"""
from backend.petra_hybrid.bm25_index import BM25Chunk, BM25Index, get_bm25_index
from backend.petra_hybrid.chunk_id import (
    id_consistency_ratio, make_chunk_id, make_content_fallback_id,
)
from backend.petra_hybrid.crag_filter import DEFAULT_THRESHOLD, apply_crag_filter
from backend.petra_hybrid.difference_matrix import build_difference_matrix, to_markdown
from backend.petra_hybrid.reranker import rerank
from backend.petra_hybrid.rrf import fuse, reciprocal_rank_fusion, to_dicts
from backend.petra_hybrid.scoring import (
    compute_confidence, dense_score_to_similarity, normalize_bm25_scores,
)

__all__ = [
    "BM25Chunk", "BM25Index", "get_bm25_index", "make_chunk_id",
    "make_content_fallback_id", "id_consistency_ratio", "fuse",
    "reciprocal_rank_fusion", "to_dicts", "apply_crag_filter",
    "DEFAULT_THRESHOLD", "rerank", "compute_confidence",
    "dense_score_to_similarity", "normalize_bm25_scores",
    "build_difference_matrix", "to_markdown", "dual_retrieval_bm25",
]


def dual_retrieval_bm25(
    products: list[str],
    base_query: str = "",
    top_k_per_product: int = 5,
    index_path: str = "data/bm25_index.jsonl",
) -> dict[str, list[dict]]:
    """Getrennte BM25-Suche je Produkt (UC-02).

    Ein festes Kontingent je Produkt verhindert, dass das besser
    dokumentierte Produkt das andere aus den Top-k verdrängt.
    Nur der lexikalische Pfad wird aufgeteilt: Produktcodes matchen dort
    exakt, und ein zusätzliches Query-Embedding je Produkt würde die
    Latenz verdoppeln. Mehr als zwei Produkte werden auf zwei begrenzt
    (UC-02 A4).

    Returns:
        Treffer je Produkt (Produktname -> Liste von Chunk-Dicts).
    """
    index = get_bm25_index(index_path)
    begrenzt = list(dict.fromkeys(products))[:2]

    ergebnisse: dict[str, list[dict]] = {}
    for produkt in begrenzt:
        query = f"{produkt} {base_query}".strip()
        treffer = index.search(query, top_k=top_k_per_product)
        # Nur Chunks, die den Produktcode enthalten (Vergleich ohne
        # Leerzeichen), damit keine fremden Abschnitte in die Spalte geraten.
        gefiltert = [
            t for t in treffer
            if produkt.lower().replace(" ", "") in (t.get("text") or "").lower().replace(" ", "")
        ]
        ergebnisse[produkt] = gefiltert or treffer
    return ergebnisse
