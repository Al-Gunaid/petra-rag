# ============================================================
# petra_hybrid/reranker.py — Cross-Encoder-Reranking
#
# Ein Cross-Encoder bewertet Anfrage und Chunk gemeinsam und unterscheidet
# ähnliche Produktcodes (WD2.5 vs. WD2.6) besser als die
# Kosinus-Ähnlichkeit eines Bi-Encoders. Er entscheidet zwischen
# widersprüchlichen BM25- und Dense-Kandidaten.
#
# Standardmäßig aktiv (PETRA_RERANKER_ENABLED). Modell über PETRA_RERANKER_MODEL;
# Abweichung A-07.
#
# Fehlt sentence-transformers oder das Modell, bleibt die
# RRF-Reihenfolge erhalten (`reranked=False`); die Pipeline läuft weiter.
# ============================================================
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger("petra.reranker")

DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"

# Prozessweiter Modell-Cache; ein Ladefehler wird gemerkt.
_model = None
_model_load_failed = False


@dataclass
class RerankResult:
    """Ergebnis des Rerankings; ``reranked=False`` bei unveränderter RRF-Reihenfolge."""
    chunks: list[dict]
    reranked: bool
    model: str | None = None
    duration_ms: int = 0
    note: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """JSON-serialisierbare Darstellung fuer n8n und FastAPI."""
        return {
            "chunks": self.chunks,
            "reranked": self.reranked,
            "reranker_model": self.model,
            "reranker_duration_ms": self.duration_ms,
            "reranker_note": self.note,
        }


def is_enabled() -> bool:
    """True, sofern PETRA_RERANKER_ENABLED nicht "false", "0" oder "no" ist."""
    return os.getenv("PETRA_RERANKER_ENABLED", "true").lower() not in ("false", "0", "no")


def _load_model():
    """Lädt den CrossEncoder einmalig (Lazy Load).

    Ein Ladefehler wird protokolliert und gemerkt; weitere Ladeversuche je
    Anfrage unterbleiben.

    Returns:
        CrossEncoder-Instanz oder None.
    """
    global _model, _model_load_failed
    if _model is not None or _model_load_failed:
        return _model
    try:
        from sentence_transformers import CrossEncoder  # type: ignore

        model_name = os.getenv("PETRA_RERANKER_MODEL", DEFAULT_MODEL)
        _model = CrossEncoder(model_name, max_length=512)
        logger.info("bge-reranker geladen: %s", model_name)
    except Exception as exc:  # noqa: BLE001 - bewusst breit: Import, Netz, VRAM
        _model_load_failed = True
        logger.warning(
            "bge-reranker nicht verfügbar (%s). Pipeline läuft mit RRF-Reihenfolge "
            "weiter (degradiert, nicht fehlerhaft).", exc,
        )
    return _model


def rerank(
    query: str,
    chunks: list[dict],
    input_top_n: int,
    output_top_n: int,
) -> RerankResult:
    """Bewertet die ersten `input_top_n` Kandidaten neu und liefert die besten `output_top_n`.

    Jeder Chunk erhält ``rerank_score``; ``rrf_score`` und ``confidence``
    bleiben zur Nachvollziehbarkeit erhalten (P6).
    """
    import time

    candidates = list(chunks or [])[:input_top_n]

    if not candidates:
        return RerankResult(chunks=[], reranked=False, note="Keine Kandidaten.")

    if not is_enabled():
        return RerankResult(
            chunks=candidates[:output_top_n],
            reranked=False,
            note="Reranking per PETRA_RERANKER_ENABLED deaktiviert.",
        )

    model = _load_model()
    if model is None:
        return RerankResult(
            chunks=candidates[:output_top_n],
            reranked=False,
            note="bge-reranker-Modell nicht verfügbar — RRF-Reihenfolge beibehalten.",
        )

    started = time.time()
    try:
        pairs = [(query, c.get("text", "") or "") for c in candidates]
        scores = model.predict(pairs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cross-Encoder-Inferenz fehlgeschlagen: %s", exc)
        return RerankResult(
            chunks=candidates[:output_top_n],
            reranked=False,
            note=f"Reranking-Fehler, RRF-Reihenfolge beibehalten: {exc}",
        )

    enriched = []
    for chunk, score in zip(candidates, scores):
        clone = dict(chunk)
        clone["rerank_score"] = round(float(score), 4)
        enriched.append(clone)
    enriched.sort(key=lambda c: c["rerank_score"], reverse=True)

    duration = int((time.time() - started) * 1000)
    return RerankResult(
        chunks=enriched[:output_top_n],
        reranked=True,
        model=os.getenv("PETRA_RERANKER_MODEL", DEFAULT_MODEL),
        duration_ms=duration,
        note=f"Top-{len(candidates)} neu bewertet, Top-{min(output_top_n, len(enriched))} weitergegeben.",
    )
