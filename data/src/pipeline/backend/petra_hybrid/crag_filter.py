# ============================================================
# petra_hybrid/crag_filter.py — CRAG-Score-Filter
#
# RAG-Architektur: [3b], Kombination D.
#
# Schwellwertfilter mit genau einer Query-Rewrite-Runde, kein vollständiges
# CRAG: Ein trainierter Retrieval Evaluator (Yan et al., arXiv:2401.15884)
# erfordert einen annotierten Domänendatensatz.
#
# Geprüft wird die quellenübergreifende Confidence aus scoring.py, nicht der
# RRF-Score (max. ≈ 0.049 bei k=60, Schwelle 0.65 unerreichbar). Damit
# können auch Chunks passieren, die nur BM25 gefunden hat.
# Abweichung von Kap. 4.2 [3b]: siehe docs/ARCHITEKTUR_ABGLEICH.md, A-04.
# ============================================================
from __future__ import annotations

from dataclasses import dataclass, field

from backend.petra_hybrid.scoring import compute_confidence

DEFAULT_THRESHOLD = 0.65
MAX_ROUNDS = 2

STATUS_PASS = "pass"
STATUS_NEEDS_REWRITE = "needs_rewrite"
STATUS_INSUFFICIENT = "insufficient_context"

# Obergrenze des weitergereichten Kontexts: Token-Budget 7.168 (Kap. 4.2 [4a])
# und "Lost in the Middle" (Liu et al. 2023, Kap. 4.6.1).
MAX_PASSED_CHUNKS = 5


@dataclass
class CragResult:
    """Ergebnis des CRAG-Score-Filters inkl. Statusentscheidung."""
    status: str
    context_sufficient: bool
    passed_chunks: list[dict] = field(default_factory=list)
    top_score: float | None = None
    reason: str = ""
    retrieval_round: int = 1
    metric: str = "max_source_confidence"

    def to_dict(self) -> dict:
        """JSON-serialisierbare Darstellung fuer n8n und FastAPI."""
        return {
            "crag_status": self.status,
            "context_sufficient": self.context_sufficient,
            "passed_chunks": self.passed_chunks,
            "top_score": self.top_score,
            "crag_reason": self.reason,
            "retrieval_round": self.retrieval_round,
            "crag_metric": self.metric,
        }


def apply_crag_filter(
    fused_chunks: list[dict],
    threshold: float = DEFAULT_THRESHOLD,
    retrieval_round: int = 1,
    dense_space: str = "auto",
    max_passed: int = MAX_PASSED_CHUNKS,
) -> CragResult:
    """Bewertet die Retrieval-Qualität und entscheidet über den weiteren Pfad.

    Args:
        fused_chunks: Ausgabe der RRF-Fusion mit ``source_scores`` und
            optional ``bm25_normalized``.
        threshold: Mindest-Confidence (Standard 0.65, UC-07).
        retrieval_round: 1 = Erstversuch, 2 = nach Query-Rewriting.
        dense_space: Score-Raum von ChromaDB (siehe scoring.py).
        max_passed: Maximale Anzahl weitergereichter Chunks.

    Returns:
        CragResult mit Status ``pass``, ``needs_rewrite`` oder
        ``insufficient_context``. Jeder Chunk erhält das Feld ``confidence``.
    """
    scored: list[tuple[float, dict]] = []
    for chunk in fused_chunks or []:
        confidence = compute_confidence(
            chunk.get("source_scores", {}) or {},
            bm25_normalized=chunk.get("bm25_normalized"),
            dense_space=dense_space,
        )
        enriched = dict(chunk)
        enriched["confidence"] = round(confidence, 4)
        scored.append((confidence, enriched))

    if not scored:
        return CragResult(
            status=STATUS_INSUFFICIENT,
            context_sufficient=False,
            top_score=None,
            reason="Retrieval lieferte keine Kandidaten.",
            retrieval_round=retrieval_round,
        )

    top_score = max(s for s, _ in scored)

    if top_score >= threshold:
        # Reihenfolge bleibt die RRF-Rangfolge; gefiltert wird über die Confidence.
       
        passed = [c for s, c in scored if s >= threshold][:max_passed]
        return CragResult(
            status=STATUS_PASS,
            context_sufficient=True,
            passed_chunks=passed,
            top_score=round(top_score, 4),
            reason=f"Top-Confidence {top_score:.3f} >= Schwellwert {threshold}.",
            retrieval_round=retrieval_round,
        )

    if retrieval_round < MAX_ROUNDS:
        return CragResult(
            status=STATUS_NEEDS_REWRITE,
            context_sufficient=False,
            top_score=round(top_score, 4),
            reason=(
                f"Top-Confidence {top_score:.3f} < Schwellwert {threshold} in Runde "
                f"{retrieval_round}. Query-Rewriting und zweite Retrieval-Runde."
            ),
            retrieval_round=retrieval_round,
        )

    return CragResult(
        status=STATUS_INSUFFICIENT,
        context_sufficient=False,
        top_score=round(top_score, 4),
        reason=(
            f"Top-Confidence {top_score:.3f} < Schwellwert {threshold} auch nach "
            f"{MAX_ROUNDS} Retrieval-Runden. Kontrollierte 'Ich weiß es nicht'-Antwort."
        ),
        retrieval_round=retrieval_round,
    )
