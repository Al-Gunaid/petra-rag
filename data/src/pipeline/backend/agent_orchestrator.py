"""agent_orchestrator — Fassade der Agenten- und Routing-Schicht (NF-10).

Intent-Klassifikation und Routing [1] (UC-01, F-07), ReAct-Agent [6a]
(UC-11), Preisextraktion (UC-04) und CAG-Cache [1c].
"""
from backend.petra_hybrid.agent import (
    MAX_CYCLES, TIMEOUT_SECONDS, assess_price_age, run_price_agent,
    web_agent_enabled, web_search,
)
from backend.petra_hybrid.cache import CagCache, get_cache, normalize_query
from backend.petra_hybrid.classification import (
    FALLBACK_CLASS, QUERY_CLASSES, ROUTING_MATRIX, classify_query_heuristic,
    extract_candidate_products, routing_for,
)
from backend.petra_hybrid.price_extraction import assess_age, extract_prices

__all__ = [
    "ROUTING_MATRIX", "QUERY_CLASSES", "FALLBACK_CLASS", "routing_for",
    "classify_query_heuristic", "extract_candidate_products",
    "run_price_agent", "web_search", "web_agent_enabled", "assess_price_age",
    "MAX_CYCLES", "TIMEOUT_SECONDS", "extract_prices", "assess_age",
    "get_cache", "CagCache", "normalize_query",
]
