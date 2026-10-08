# ============================================================
# query_router_v3.py — FastAPI-Router der PETRA-RAG-Query-Pipeline
#
# Die Orchestrierung läuft in n8n. Python stellt nur Bausteine bereit,
# für die es keine n8n-Nodes gibt, sowie POST /query als Delegation an
# den n8n-Webhook.
#
# Endpoints (Architekturbaustein in Klammern, Kap. 4.2):
#   POST /query                  Delegation an n8n
#   GET  /retrieval/bm25         BM25 Sparse Retrieval [2a]
#   POST /retrieval/fuse         RRF [3a] inkl. ID-Konsistenzprüfung
#   POST /retrieval/crag-filter  CRAG-Score-Filter [3b]
#   POST /retrieval/rerank       Cross-Encoder [3c]
#   POST /retrieval/prompt       Prompt mit Token-Budget [4a]
#   POST /retrieval/sources      Quellenliste aus Metadaten [5b]
#   POST /retrieval/dual|difference-matrix|prices|language
#   POST /cache/lookup|store|invalidate, GET /cache/stats   CAG [1c]
#   POST /agent/price            Preis-Agent [6a]
#   POST /telemetry/log, GET /telemetry/summary             [5c]
#   GET  /ops/health/deep        Ollama, ChromaDB, n8n, BM25, Modellkonfiguration
#
# Einbindung: api_server.py (app.include_router für alle Router).
# ============================================================
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from backend.petra_hybrid.agent import run_price_agent, web_agent_enabled
from backend.petra_hybrid.bm25_index import get_bm25_index
from backend.petra_hybrid.cache import get_cache, is_enabled as cache_enabled
from backend.petra_hybrid.crag_filter import DEFAULT_THRESHOLD, apply_crag_filter
from backend.petra_hybrid.difference_matrix import build_difference_matrix, to_markdown
from backend.petra_hybrid.language import detect_language, unsupported_language_message
from backend.petra_hybrid.price_extraction import extract_prices
from backend.petra_hybrid.prompt_builder import build_prompt
from backend.petra_hybrid.reranker import is_enabled as reranker_enabled, rerank
from backend.petra_hybrid.response_schema import build_sources_from_chunks
from backend.petra_hybrid.rrf import fuse, to_dicts
from backend.retrieval_engine import dual_retrieval_bm25
from backend.petra_hybrid import telemetry

logger = logging.getLogger("petra.query_router_v3")

N8N_WEBHOOK_URL = os.getenv(
    "PETRA_N8N_QUERY_WEBHOOK_URL",
    "http://n8n:5678/webhook/petra-query",
)
N8N_TIMEOUT = float(os.getenv("PETRA_N8N_TIMEOUT", "150"))
BM25_INDEX_PATH = os.getenv("PETRA_BM25_INDEX_PATH", "data/bm25_index.jsonl")
DENSE_SCORE_SPACE = os.getenv("PETRA_CHROMA_SCORE_SPACE", "auto")
OLLAMA_URL = os.getenv("PETRA_OLLAMA_URL", "http://ollama:11434")
CHROMA_URL = os.getenv("PETRA_CHROMA_URL", "http://chromadb:8000")
EMBEDDING_MODEL = os.getenv("PETRA_EMBEDDING_MODEL", "bge-m3:latest")
LLM_MODEL = os.getenv("PETRA_LLM_MODEL", "llama3.1:8b")

router = APIRouter(prefix="/query", tags=["query"])
_query_lock = asyncio.Lock()    # höchstens eine Anfrage gleichzeitig an n8n
retrieval_router = APIRouter(prefix="/retrieval", tags=["retrieval"])
cache_router = APIRouter(prefix="/cache", tags=["cache"])
agent_router = APIRouter(prefix="/agent", tags=["agent"])
telemetry_router = APIRouter(prefix="/telemetry", tags=["telemetry"])
ops_router = APIRouter(prefix="/ops", tags=["ops"])


# ── Modelle ──────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    language: str = "Deutsch"
    # s7: Einstellungen – None = Workflow-Standard (6 Kontext-Chunks, wie s6)
    top_k: int | None = Field(default=None, ge=1, le=10)
    score_threshold: float = Field(default=DEFAULT_THRESHOLD, ge=0.0, le=1.0)
    request_id: str | None = None
    # s7: Einstellungen des Frontends – None = Stand des Workflows
    temperature: float | None = Field(default=None, ge=0.0, le=1.0)
    model: str | None = Field(default=None, max_length=64, pattern=r"^[A-Za-z0-9._:/-]+$")
    web_agent_enabled: bool | None = None


class SourceModel(BaseModel):
    dateiname: str
    seite: int | None = None
    chunk_id: str
    hersteller: str = "unbekannt"
    produktname: str | None = None
    artikelnummer: str | None = None
    dokumenttyp: str = "text"
    score: float | None = None
    textauszug: str = ""


class QueryResponse(BaseModel):
    """Antwortschema von POST /query; abwärtskompatibel zum Streamlit-Frontend."""

    answer: str
    sources: list[SourceModel] = []
    verified: bool | None = None
    query_class: str = ""
    q_class: str = ""
    context_sufficient: bool = True
    retrieved_documents: int = 0
    latency_ms: int = 0
    model: str = LLM_MODEL
    temperature: float | None = None   # s7: Einstellungen – vom Workflow gemeldet
    retrieval_method: str = "hybrid_bm25_dense_rrf"
    faithfulness_score: float | None = None
    fallback_reason: str | None = None
    cache_hit: bool = False
    reranked: bool = False
    request_id: str | None = None
    contexts: list[str] = []          # v3.1: volle Chunks für RAGAS
    unbelegte_werte: list[str] = []   # v3.1: Belegcheck


# ── /query — Delegation an n8n ───────────────────────────────────────
@router.post("", response_model=QueryResponse)
async def submit_query(payload: QueryRequest) -> QueryResponse:
    """Delegiert die RAG-Verarbeitung an den n8n-Workflow.

    Der Timeout (PETRA_N8N_TIMEOUT) liegt bewusst über dem NF-03-Ziel von
    5 s, da CRAG-Runde 2 und der Agentic-Pfad länger laufen dürfen.

    Ein Lock begrenzt auf eine gleichzeitige Anfrage; weitere erhalten
    sofort HTTP 503. Parallele Anfragen führten zu einem Rückstau in
    n8n/Ollama mit verwaisten Executions.

    Raises:
        HTTPException: 503 (belegt oder n8n nicht erreichbar),
            502 (Fehlerstatus oder ungültiges JSON von n8n).
    """
    if _query_lock.locked():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Eine vorherige Anfrage wird noch verarbeitet. Bitte kurz warten.",
        )

    async with _query_lock:
        try:
            async with httpx.AsyncClient(timeout=N8N_TIMEOUT) as client:
                response = await client.post(N8N_WEBHOOK_URL, json=payload.model_dump())
        except httpx.RequestError as exc:
            logger.error("n8n Webhook nicht erreichbar: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "RAG-Orchestrierung (n8n) derzeit nicht erreichbar. Bitte prüfen, ob "
                    "n8n und Ollama laufen (docker compose ps) und ob der Workflow "
                    "aktiviert ist."
                ),
            ) from exc

        if response.status_code >= 400:
            logger.error("n8n Webhook Fehler %s: %s", response.status_code, response.text[:500])
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"n8n-Workflow meldete Fehler {response.status_code}.",
            )

        try:
            data = response.json()
        except ValueError as exc:
            body_preview = response.text[:300] if response.text else "(leerer Body)"
            logger.error(
                "n8n Webhook lieferte Status %s, aber kein gueltiges JSON: %s",
                response.status_code, body_preview,
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=(
                    "n8n-Workflow hat eine leere oder ungültige Antwort geliefert. "
                    "Prüfen Sie den 'Respond to Webhook'-Node im Workflow und die "
                    "n8n-Execution-Logs für diese Anfrage."
                ),
            ) from exc

        if isinstance(data, list) and data:
            data = data[0]
        return QueryResponse(**_coerce_response(data))


def _coerce_response(data: dict) -> dict:
    """
    Übernimmt nur bekannte Felder von QueryResponse; `answer` hat einen Default.

    Zusätzliche oder fehlende Felder aus dem Workflow führen so nicht zu
    einem Validierungsfehler.
    """
    known = set(QueryResponse.model_fields)
    return {k: v for k, v in (data or {}).items() if k in known} | {
        "answer": (data or {}).get("answer", "Ich weiß es nicht.")
    }


@router.get("/health")
async def query_path_health() -> dict[str, str]:
    return {"status": "ok", "n8n_webhook": N8N_WEBHOOK_URL}


# ── Retrieval-Bausteine ohne n8n-Äquivalent ──────────────────────────
@retrieval_router.get("/bm25")
async def bm25_search(
    q: str = Query(min_length=1),
    top_k: int = Query(default=10, ge=1, le=50),
    manufacturer: str | None = Query(default=None),
) -> dict[str, Any]:
    """BM25 Sparse Retrieval.

    `top_k` = 10 entspricht dem Dense-Retrieval. Ungleiche Listenlängen
    verzerren RRF zulasten der kürzeren Liste.
    """
    index = get_bm25_index(BM25_INDEX_PATH)
    if index.is_empty():
        # Explizite Meldung statt stiller leerer Liste — ein leerer
        # BM25-Index ist ein Betriebsfehler (Ingestion nicht gelaufen)
        # und würde sonst als "keine Treffer" fehlinterpretiert.
        return {
            "results": [], "count": 0, "query": q,
            "warning": (
                "BM25-Index ist leer. Ingestion mit index_chunks_hybrid() ausführen "
                "(siehe MIGRATION.md, Schritt 6)."
            ),
        }
    results = index.search(q, top_k=top_k, manufacturer_filter=manufacturer)
    return {"results": results, "count": len(results), "query": q}


@retrieval_router.post("/fuse")
async def fuse_rankings(payload: dict) -> dict:
    """RRF-Fusion mit Prüfung der Chunk-ID-Konsistenz."""
    ranked_lists = payload.get("ranked_lists", {})
    result = fuse(
        ranked_lists,
        k=payload.get("k", 60),
        top_k=payload.get("top_k", 15),
    )
    return {
        "fused_chunks": to_dicts(result.items),
        "count": len(result.items),
        "id_consistency": result.id_consistency,
        "total_candidates": result.total_candidates,
        "warnings": result.warnings,
    }


@retrieval_router.post("/crag-filter")
async def crag_filter_endpoint(payload: dict) -> dict:
    """CRAG-Score-Filter (Kap. 4.2 [3b]) mit v2-Confidence-Metrik."""
    result = apply_crag_filter(
        payload.get("fused_chunks", []),
        threshold=payload.get("threshold", DEFAULT_THRESHOLD),
        retrieval_round=payload.get("retrieval_round", 1),
        dense_space=payload.get("dense_space", DENSE_SCORE_SPACE),
    )
    return result.to_dict()


@retrieval_router.post("/rerank")
async def rerank_endpoint(payload: dict) -> dict:
    """Cross-Encoder Reranking.

    Ohne Modell bleibt die RRF-Reihenfolge erhalten.
    """
    result = rerank(
        payload.get("query", ""),
        payload.get("chunks", []),
        input_top_n=payload.get("input_top_n", 10),
        output_top_n=payload.get("output_top_n", 5),
    )
    return result.to_dict()


@retrieval_router.post("/prompt")
async def build_prompt_endpoint(payload: dict) -> dict:
    """Prompt-Komposition mit Token-Budget."""
    return build_prompt(
        query=payload.get("query", ""),
        chunks=payload.get("chunks", []),
        query_class=payload.get("query_class", ""),
        language=payload.get("language", "Deutsch"),
    )


@retrieval_router.post("/sources")
async def sources_from_chunks(payload: dict) -> dict:
    """Quellenliste aus Chunk-Metadaten.

    Nicht vom LLM erzeugt, damit keine erfundenen Seitenzahlen entstehen.
    """
    sources = build_sources_from_chunks(payload.get("chunks", []))
    return {"sources": [s.__dict__ for s in sources], "count": len(sources)}


@retrieval_router.post("/dual")
async def dual_retrieval(payload: dict) -> dict:
    """UC-02 Hauptszenario 3: separate Suche je Produkt.

    benennt das Problem: Bei einer gemeinsamen Runde
    verdraengt das dokumentationsstaerkere Produkt das andere aus den
    Top-k-Slots. UC-02 A4 begrenzt auf zwei Produkte je Anfrage.
    """
    produkte = payload.get("products", [])
    if len(produkte) < 2:
        # UC-02 A1: nur ein Produkt erkannt
        return {
            "dual_retrieval": False,
            "products": produkte,
            "per_product": {},
            "hinweis": (
                "Fuer einen Vergleich werden mindestens zwei Produkte benoetigt. "
                "Bitte nennen Sie beide Produktbezeichnungen."
            ),
        }


    per_product = dual_retrieval_bm25(
        produkte,
        base_query=payload.get("query", ""),
        top_k_per_product=payload.get("top_k_per_product", 5),
        index_path=BM25_INDEX_PATH,
    )
    return {
        "dual_retrieval": True,
        "products": list(per_product),
        "per_product": per_product,
        "begrenzt_auf_zwei": len(produkte) > 2,
        "counts": {p: len(c) for p, c in per_product.items()},
    }


@retrieval_router.post("/difference-matrix")
async def difference_matrix(payload: dict) -> dict:
    """UC-03: Differenzmatrix aus den Chunks je Produkt inkl. Markdown für den Prompt.

    Das LLM ordnet belegte Werte ein, statt sie aus Fließtext zu
    extrahieren (Halluzinationsschutz, F-13).
    """
    produkt_chunks = payload.get("product_chunks", {})
    matrix = build_difference_matrix(produkt_chunks, payload.get("dimensions"))
    return {**matrix, "markdown": to_markdown(matrix)}


@retrieval_router.post("/prices")
async def price_extraction(payload: dict) -> dict:
    """
    Regelbasierte Preisextraktion aus Chunks (UC-04) inkl. Altersprüfung (12 Monate, UC-04 A3).
    """
    return extract_prices(payload.get("chunks", [])).to_dict()


@retrieval_router.post("/language")
async def language_detection(payload: dict) -> dict:
    """Spracherkennung der Anfrage (NF-08, UC-01 A5)."""
    result = detect_language(payload.get("query", ""))
    if not result["supported"]:
        result["meldung"] = unsupported_language_message()
    return result


# ── CAG-Cache ────────────────────────────────────────
@cache_router.post("/lookup")
async def cache_lookup(payload: dict) -> dict:
    if not cache_enabled():
        return {"cache_hit": False, "cache_reason": "Cache deaktiviert."}
    return get_cache().lookup(
        payload.get("query", ""),
        kind=payload.get("kind", "response"),
        extra=payload.get("extra", ""),
    ).to_dict()


@cache_router.post("/store")
async def cache_store(payload: dict) -> dict:
    if not cache_enabled():
        return {"stored": False, "reason": "Cache deaktiviert."}
    key = get_cache().store(
        payload.get("query", ""),
        payload.get("payload", {}),
        kind=payload.get("kind", "response"),
        extra=payload.get("extra", ""),
    )
    return {"stored": True, "key": key}


@cache_router.post("/invalidate")
async def cache_invalidate(payload: dict) -> dict:
    """Invalidiert den Cache nach einem Dokumentimport.

    Mit `corpus_version` werden Einträge anderer Versionen verworfen,
    ohne wird der Cache vollständig geleert.
    """
    version = payload.get("corpus_version")
    if version:
        return {"dropped": get_cache().set_corpus_version(version), "corpus_version": version}
    return {"dropped": get_cache().clear(), "corpus_version": get_cache().corpus_version}


@cache_router.get("/stats")
async def cache_stats() -> dict:
    return get_cache().stats()


# ── Agentic Extension (Kap. 4.2 Schicht 6) ───────────────────────────
@agent_router.post("/price")
async def price_agent(payload: dict) -> dict:
    """Q4-Preisanfrage über den ReAct-Pfad.

    Das lokale Retrieval wird als Tool übergeben und nutzt den BM25-Index
    direkt; ein Aufruf über n8n würde eine Webhook-Schleife erzeugen.
    """
    query = payload.get("query", "")

    def _window_around_match(text: str, needle: str, before: int = 20, after: int = 180) -> str:
        """Schneidet den Chunk-Text auf den Bereich um `needle` zu.

        Preislisten-Chunks enthalten mehrere Produktzeilen. Ohne Fenster
        kann extract_prices() den Preis eines Nachbarprodukts liefern.
        `after=180` deckt einen üblichen Produktblock ab.
        """
        idx = text.lower().find(needle.lower())
        if idx == -1:
            return text
        start = max(0, idx - before)
        end = min(len(text), idx + len(needle) + after)
        return text[start:end]

    def local_retrieval(q: str) -> list[dict]:
        """Lokales Retrieval-Tool des Preis-Agenten.

        1. Exakte Substring-Suche in Preislisten (Dateiname enthält
           "preisliste"; ein doc_type-Feld existiert nicht). Bei fehlendem
           Treffer wird die Anfrage schrittweise von vorne gekürzt, da der
           Herstellername in Preiszeilen meist fehlt.
        2. Nur ohne exakten Treffer: BM25-Ergebnisse. Eine Mischung würde
           Preise fremder Produkte mit ähnlichen Kürzeln einbringen.
        """
        index = get_bm25_index(BM25_INDEX_PATH)
        bm25_hits = index.search(q, top_k=payload.get("top_k", 5))

        exact_hits: list[dict] = []
        seen_exact: set[str] = set()
        words = q.split()
        for start in range(len(words)):
            candidate = " ".join(words[start:]).strip()
            if not candidate:
                continue
            hits = index.search_by_filename_substring(
                filename_substring="preisliste",
                text_substring=candidate,
            )
            for h in hits:
                if h["chunk_id"] in seen_exact:
                    continue
                seen_exact.add(h["chunk_id"])
                h = dict(h)
                h["text"] = _window_around_match(h["text"], candidate)
                exact_hits.append(h)
            if exact_hits:
                # Keine kürzeren, unpräziseren Varianten mehr prüfen.
                break

        if exact_hits:
            return exact_hits

        merged: list[dict] = []
        seen: set[str] = set()
        for h in bm25_hits:
            if h["chunk_id"] in seen:
                continue
            seen.add(h["chunk_id"])
            merged.append(h)
        return merged

    result = run_price_agent(query, local_retrieval)

    # Strukturierte Preisangabe mit Dokumentdatum und Altershinweis (UC-04).
    preise = extract_prices(result.answer_context)

    merged = result.to_dict()
    if preise.gefunden:
        # extract_prices() hat Vorrang, da nur dort die Altersprüfung (UC-04 A3) erfolgt.
        merged |= preise.to_dict()
    else:
        # Ein bereits von agent.py gefundener Preis darf nicht durch das
        # leere Ergebnis von extract_prices() überschrieben werden.
        merged.setdefault("widerspruechlich", False)
        merged["web_recherche_empfohlen"] = not merged.get("preis_gefunden", False)
        if not merged.get("preis_gefunden"):
            merged["preis_meldung"] = preise.meldung

    merged["web_agent_enabled"] = web_agent_enabled()
    return merged


# ── Telemetrie (Kap. 4.2 [5c]) ───────────────────────────────────────
@telemetry_router.post("/log")
async def telemetry_log(payload: dict) -> dict:
    """
    Schreibt einen Telemetrie-Datensatz; antwortet immer mit HTTP 200.

    Der aufrufende n8n-Node ist mit onError=continue konfiguriert.
    """
    return {"written": telemetry.log_request(payload)}


@telemetry_router.get("/summary")
async def telemetry_summary(limit: int | None = None) -> dict:
    """Aggregat gegen die Qualitätsziele."""
    return telemetry.summarize(telemetry.read_records(limit))


# ── Betriebsdiagnose ─────────────────────────────────────────────────
@ops_router.get("/health/deep")
async def deep_health() -> dict:
    """Prüft Laufzeitabhängigkeiten und Modellkonfiguration vor dem ersten Request.

    Erkennt falsche Servicenamen, fehlende Ollama-Modelle, einen leeren
    BM25-Index und gibt die Embedding-Konfiguration aus.
    """
    checks: dict[str, dict] = {}

    async with httpx.AsyncClient(timeout=5.0) as client:
        # Ollama + Modellverfügbarkeit
        try:
            response = await client.get(f"{OLLAMA_URL}/api/tags")
            models = [m.get("name", "") for m in response.json().get("models", [])]
            missing = [m for m in (LLM_MODEL, EMBEDDING_MODEL) if m not in models]
            checks["ollama"] = {
                "ok": not missing,
                "url": OLLAMA_URL,
                "models_found": len(models),
                "missing_models": missing,
                "hint": (
                    f"ollama pull {' && ollama pull '.join(missing)}" if missing else None
                ),
            }
        except Exception as exc:  # noqa: BLE001
            checks["ollama"] = {"ok": False, "url": OLLAMA_URL, "error": str(exc)}

        # ChromaDB
        try:
            response = await client.get(f"{CHROMA_URL}/api/v2/heartbeat")
            checks["chromadb"] = {"ok": response.status_code < 400, "url": CHROMA_URL}
        except Exception as exc:  # noqa: BLE001
            checks["chromadb"] = {"ok": False, "url": CHROMA_URL, "error": str(exc)}

        # n8n-Erreichbarkeit
        try:
            response = await client.get(N8N_WEBHOOK_URL.rsplit("/webhook/", 1)[0] + "/healthz")
            checks["n8n"] = {"ok": response.status_code < 400, "webhook": N8N_WEBHOOK_URL}
        except Exception as exc:  # noqa: BLE001
            checks["n8n"] = {"ok": False, "webhook": N8N_WEBHOOK_URL, "error": str(exc)}

    index = get_bm25_index(BM25_INDEX_PATH)
    checks["bm25"] = {
        "ok": not index.is_empty(),
        "path": BM25_INDEX_PATH,
        "hint": None if not index.is_empty() else "Ingestion mit index_chunks_hybrid() ausführen.",
    }
    checks["config"] = {
        "ok": True,
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": LLM_MODEL,
        "dense_score_space": DENSE_SCORE_SPACE,
        "reranker_enabled": reranker_enabled(),
        "cache_enabled": cache_enabled(),
        "web_agent_enabled": web_agent_enabled(),
        "telemetry_enabled": telemetry.is_enabled(),
        "hinweis_embedding": (
            "Das Embedding-Modell MUSS mit dem der Ingestion übereinstimmen. "
            "Bei Abweichung liefert Dense-Retrieval bedeutungsloses Ranking — "
            "ohne Fehlermeldung."
        ),
    }

    all_ok = all(check.get("ok") for check in checks.values())
    return {"ok": all_ok, "checks": checks}