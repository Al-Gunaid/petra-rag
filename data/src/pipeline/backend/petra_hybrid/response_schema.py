# ============================================================
# petra_hybrid/response_schema.py — Strukturiertes Antwortschema
#
# Zielschema für FastAPI (query_router_v3.py) und den n8n-Node
# "Output Parser (Antwort + Quellen)". Quellen werden aus den
# Chunk-Metadaten erzeugt, nicht vom LLM (build_sources_from_chunks).
# ============================================================
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Source:
    """Eine Quellenangabe gemaess UC-09 / F-11 (Pflichtfelder der Anzeige)."""
    dateiname: str
    seite: int | None
    chunk_id: str
    hersteller: str
    produktname: str | None
    artikelnummer: str | None
    dokumenttyp: str
    score: float | None
    textauszug: str
    # Pflichtfelder laut UC-09 HS 4 / F-11 (Metadatenvollständigkeit >= 95 %);
    # fehlende Werte bleiben als None sichtbar (UC-09 A1)..
    datum: str | None = None
    relevanz_prozent: int | None = None


@dataclass
class StructuredAnswer:
    """Antwortschema des /query-Endpoints."""
    answer: str
    query_class: str
    context_sufficient: bool
    sources: list[Source] = field(default_factory=list)
    retrieved_documents: int = 0
    latency_ms: int = 0
    model: str = "llama3.1:8b"
    retrieval_method: str = "hybrid_bm25_dense_rrf"
    verified: bool = False
    faithfulness_score: float | None = None
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisierbare Darstellung fuer n8n und FastAPI."""
        return asdict(self)


def _textauszug(text: str, max_len: int = 220) -> str:
    text = (text or "").strip()
    return text if len(text) <= max_len else text[: max_len - 1].rstrip() + "…"


def _relevanz_prozent(chunk: dict) -> int | None:
    """Relevanz in Prozent (UC-09) aus confidence, rerank_score oder score.

    Der RRF-Score ist ungeeignet: Sein Maximum liegt bei ≈ 0,049 und
    würde als "5 %" angezeigt.
    """
    for key in ("confidence", "rerank_score", "score"):
        value = chunk.get(key)
        if isinstance(value, (int, float)):
            return max(0, min(100, int(round(float(value) * 100))))
    return None


def build_sources_from_chunks(chunks: list[dict]) -> list[Source]:
    """Erzeugt die Quellenliste aus Chunk-Metadaten, dedupliziert über chunk_id.
    Fehlende Metadaten werden als "unbekannt" geführt statt ausgelassen,
    damit Lücken sichtbar bleiben (Nachvollziehbarkeit, NF-06)."""
    sources: list[Source] = []
    seen: set[str] = set()
    for chunk in chunks:
        meta = chunk.get("metadata", {}) or {}
        cid = chunk.get("chunk_id", meta.get("chunk_id", "unbekannt"))
        if cid in seen:
            continue
        seen.add(cid)
        score = chunk.get("rrf_score") or meta.get("rrf_score") or chunk.get("score")
        sources.append(
            Source(
                dateiname=(meta.get("document") or meta.get("dateiname") or meta.get("filename") or meta.get("source") or "unbekannt"),
                seite=meta.get("page", meta.get("seite")),
                chunk_id=cid,
                hersteller=meta.get("manufacturer", meta.get("hersteller", "unbekannt")),
                produktname=meta.get("product", meta.get("produktname")),
                artikelnummer=meta.get("artikelnummer", meta.get("product_code")),
                dokumenttyp=meta.get("source_type", meta.get("dokumenttyp", "text")),
                score=round(float(score), 4) if isinstance(score, (int, float)) else None,
                textauszug=_textauszug(chunk.get("text", "")),
                datum=meta.get("date", meta.get("datum")),
                relevanz_prozent=_relevanz_prozent(chunk),
            )
        )
    return sources


def build_no_context_answer(query_class: str, latency_ms: int, fallback_reason: str) -> StructuredAnswer:
    """Kontrollierte Antwort bei unzureichender Evidenz (CRAG-Filter FAIL
    nach max. Runden). Quellenliste bleibt bewusst leer."""
    return StructuredAnswer(
        answer="Ich weiß es nicht.",
        query_class=query_class,
        context_sufficient=False,
        sources=[],
        retrieved_documents=0,
        latency_ms=latency_ms,
        verified=False,
        fallback_reason=fallback_reason,
    )
