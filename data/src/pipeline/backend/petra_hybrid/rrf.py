# ============================================================
# petra_hybrid/rrf.py — Reciprocal Rank Fusion 
#
# RRF(d) = Σ_i  1 / (k + rank_i(d))
#
# ARCHITEKTURBEZUG: Schicht [3a],[3a].
#
# Unabhängig von der Score-Skalierung der Quellen (BM25 [0, ∞) vs.
# Kosinus [0, 1]). Der RRF-Score dient nur dem Ranking; für den
# CRAG-Filter wird zusätzlich `bm25_normalized` mitgeführt (scoring.py).
#
# ID-Konsistenzprüfung: Gemessen wird der Anteil gemeinsamer chunk_ids
# zwischen Pfaden, die denselben Suchraum durchsuchen (COMPARABLE_SOURCES).
# Ohne gemeinsame IDs reiht RRF die Listen nur aneinander. Der TAG-Pfad
# ist ausgenommen, da er eine eigene Collection (petra_tag_chunks) liest.
#
# Zusätzlich wird die Überschneidung auf Dokumentebene gemessen:
#   - gleiche Dokumente, andere Chunks  -> Hinweis auf zu feines Chunking
#   - verschiedene Dokumente            -> lexikalische und semantische
#                                          Suche reagieren auf andere Signale
# ============================================================
from __future__ import annotations

import re
from dataclasses import dataclass, field

from backend.petra_hybrid.chunk_id import make_content_fallback_id
from backend.petra_hybrid.scoring import normalize_bm25_scores


# Pfade mit gemeinsamem Suchraum. BM25 indexiert Text- und Tabellen-Chunks
# gemeinsam und ist daher mit Dense vergleichbar.
COMPARABLE_SOURCES = ("dense", "bm25")

# Dateiname aus der chunk_id: "1020000000_de.pdf_p2_text_3_5" -> "1020000000_de.pdf"
_DOC_FROM_ID = re.compile(r"^(.*?)_p\d+_")


@dataclass
class RankedItem:
    """Ein fusionierter Chunk mit Scores je Retrieval-Quelle."""
    chunk_id: str
    metadata: dict = field(default_factory=dict)
    text: str = ""
    source_scores: dict[str, float] = field(default_factory=dict)
    bm25_normalized: float | None = None
    sources: list[str] = field(default_factory=list)


@dataclass
class FusionResult:
    """Fusionsergebnis mit Konsistenzkennzahlen und Warnungen.

    Attributes:
        id_consistency: Anteil gemeinsamer chunk_ids der vergleichbaren
            Pfade. 1.0, wenn weniger als zwei Pfade Treffer lieferten
            (siehe ``consistency_measurable``).
        consistency_measurable: True, wenn zwei vergleichbare Pfade
            Treffer hatten.
        doc_overlap: Anzahl gemeinsamer Dokumente. > 0 bei
            id_consistency == 0 bedeutet: gleiche Dokumente, andere Abschnitte.
    """
    items: list[RankedItem]
    id_consistency: float
    total_candidates: int
    consistency_measurable: bool = False
    doc_overlap: int = 0
    warnings: list[str] = field(default_factory=list)


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[dict]],
    k: int,
    top_k: int,
) -> list[RankedItem]:
    """Rückwärtskompatible Fassade — liefert nur die fusionierte Liste."""
    return fuse(ranked_lists, k=k, top_k=top_k).items


def fuse(
    ranked_lists: dict[str, list[dict]],
    k: int,
    top_k: int,
) -> FusionResult:
    """Fusioniert mehrere Ranking-Listen per RRF.

    Args:
        ranked_lists: Quellenname ("dense"/"bm25"/"tag") -> Trefferliste.
            Treffer: {"chunk_id", "rank", "score"}, optional {"metadata",
            "text"}; `rank` beginnt bei 1.
        k: RRF-Konstante (Standard 60).
        top_k: Maximale Anzahl fusionierter Kandidaten.
    """
    normalized_lists = {
        name: _ensure_ids(items or []) for name, items in (ranked_lists or {}).items()
    }
    bm25_norm = normalize_bm25_scores(normalized_lists.get("bm25", []))

    fused: dict[str, RankedItem] = {}
    rrf_scores: dict[str, float] = {}

    for source_name, items in normalized_lists.items():
        for position, item in enumerate(items):
            cid = item["chunk_id"]
            rank = int(item.get("rank") or (position + 1))
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)

            if cid not in fused:
                fused[cid] = RankedItem(
                    chunk_id=cid,
                    metadata=item.get("metadata") or {},
                    text=item.get("text") or "",
                )
            entry = fused[cid]
            entry.source_scores[source_name] = item.get("score", 0.0)
            if source_name not in entry.sources:
                entry.sources.append(source_name)
            if source_name == "bm25" and cid in bm25_norm:
                entry.bm25_normalized = round(bm25_norm[cid], 4)
            if not entry.text and item.get("text"):
                entry.text = item["text"]
            if not entry.metadata and item.get("metadata"):
                entry.metadata = item["metadata"]

    ordered = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)
    result: list[RankedItem] = []
    for cid in ordered[:top_k]:
        item = fused[cid]
        item.metadata = {**item.metadata, "rrf_score": round(rrf_scores[cid], 6)}
        result.append(item)

    consistency, measurable, doc_overlap, warnings = _consistency(normalized_lists)
    return FusionResult(
        items=result,
        id_consistency=consistency,
        total_candidates=len(rrf_scores),
        consistency_measurable=measurable,
        doc_overlap=doc_overlap,
        warnings=warnings,
    )


def _ensure_ids(items: list[dict]) -> list[dict]:
    """Ergänzt fehlende chunk_ids durch einen Inhalts-Hash und setzt fehlende Ränge.

    Quellenspezifische Platzhalter ("dense-0") würden nie übereinstimmen
    und die Deduplizierung verhindern.
    """
    out = []
    for position, item in enumerate(items):
        clone = dict(item)
        cid = clone.get("chunk_id") or (clone.get("metadata") or {}).get("chunk_id")
        if not cid:
            source_type = (clone.get("metadata") or {}).get("source_type", "text")
            cid = make_content_fallback_id(clone.get("text", ""), source_type)
        clone["chunk_id"] = cid
        clone.setdefault("rank", position + 1)
        out.append(clone)
    return out


def _document_of(item: dict) -> str:
    """
    Quelldokument eines Treffers aus den Metadaten, ersatzweise aus der chunk_id.
    """
    meta = item.get("metadata") or {}
    for key in ("filename", "document", "source"):
        value = meta.get(key)
        if value:
            return str(value)
    match = _DOC_FROM_ID.match(str(item.get("chunk_id") or ""))
    return match.group(1) if match else ""


def _consistency(
    lists: dict[str, list[dict]],
) -> tuple[float, bool, int, list[str]]:
    """Misst die Überschneidung zwischen vergleichbaren Pfaden.

    Returns:
        (id_consistency, measurable, doc_overlap, warnings)
    """
    comparable = {
        name: items
        for name, items in lists.items()
        if name in COMPARABLE_SOURCES and items
    }
    if len(comparable) < 2:
        # Nur ein Pfad aktiv: 1.0, damit ein Ausfall nicht als
        # Qualitätsproblem erscheint; `measurable` hält den Unterschied fest.
        return 1.0, False, 0, []

    names = list(comparable)
    id_sets = {n: {i["chunk_id"] for i in comparable[n]} for n in names}
    doc_sets = {n: {_document_of(i) for i in comparable[n]} - {""} for n in names}

    # Paarweise Überschneidung, bezogen auf die kleinere Menge.
    overlaps = []
    doc_overlaps = []
    for a_idx in range(len(names)):
        for b_idx in range(a_idx + 1, len(names)):
            a, b = id_sets[names[a_idx]], id_sets[names[b_idx]]
            smaller = min(len(a), len(b)) or 1
            overlaps.append(len(a & b) / smaller)
            doc_overlaps.append(
                len(doc_sets[names[a_idx]] & doc_sets[names[b_idx]])
            )

    ratio = max(overlaps) if overlaps else 0.0
    doc_overlap = max(doc_overlaps) if doc_overlaps else 0

    warnings: list[str] = []
    if ratio == 0.0:
        warnings.append(
            f"RRF-Überschneidung 0 %: {' und '.join(names)} liefern keine "
            f"gemeinsamen chunk_ids; RRF reiht die Listen aneinander statt "
            f"Ränge zu kombinieren. Gemeinsame Dokumente: {doc_overlap}. "
            f"Die chunk_ids selbst sind konsistent (geprüft: 100 % Deckung "
            f"BM25-Index gegen ChromaDB) — bei gemeinsamen Dokumenten "
            f"deutet dies auf zu feines Chunking hin, ohne gemeinsame "
            f"Dokumente darauf, dass lexikalische und semantische Suche "
            f"auf unterschiedliche Signale reagieren."
        )
    return round(ratio, 4), True, doc_overlap, warnings


def to_dicts(items: list[RankedItem]) -> list[dict]:
    """Serialisiert RankedItems für JSON-Responses und n8n."""
    return [
        {
            "chunk_id": it.chunk_id,
            "text": it.text,
            "metadata": it.metadata,
            "rrf_score": it.metadata.get("rrf_score"),
            "source_scores": it.source_scores,
            "bm25_normalized": it.bm25_normalized,
            "sources": it.sources,
        }
        for it in items
    ]