# ============================================================
# petra_hybrid/classification.py — Anfrageklassifikation
#
#
# Im Betrieb klassifiziert der n8n-Node "Text Classifier" (Ollama). Dieses
# Modul ist die zentrale Definition für:
#   1) Klassenmenge und Routing-Wirkung (ROUTING_MATRIX, Q1–Q5),
#   2) Produktcode-Erkennung,
#   3) einen deterministischen Regel-Fallback für Tests und bei Ausfall
#      des LLM-Classifiers.
#
# Jede Klasse hat einen Routing-Eintrag; unbekannte Klassen fallen auf
# die Restklasse (`fallback`), sodass kein Pfad ohne Antwort entsteht.
# ============================================================
from __future__ import annotations

import re

# -- Klassen und ihre Routing-Wirkung ------------------------------------
#
# Steuert, welche Retrieval-Pfade im n8n-Workflow aktiv sind.
ROUTING_MATRIX: dict[str, dict] = {
    "produktspezifikation": {
        "q_class": "Q1",
        "dense": True, "bm25": True, "tag": True,
        "hyde": False, "agentic": False,
        "beschreibung": "Attribut-Lookup zu EINEM benannten Produkt (~35 % der Anfragen).",
    },
    "produktvergleich": {
        "q_class": "Q2",
        "dense": True, "bm25": True, "tag": True,
        "hyde": False, "agentic": False,
        "dual_retrieval": True,
        "beschreibung": "Dimensionsweiser Vergleich von >=2 Produkten (~25 %).",
    },
    "produktsuche": {
        "q_class": "Q3",
        "dense": True, "bm25": True, "tag": False,
        "hyde": True, "agentic": False,
        "beschreibung": "Konzeptuelle Produktsuche ohne bekannten Code (~20 %).",
    },
    "preisanfrage": {
        "q_class": "Q4",
        "dense": True, "bm25": True, "tag": False,
        "hyde": False, "agentic": True,
        "beschreibung": "Preis/Verfügbarkeit — Agentic-Pfad, Kap. 4.4.2 (~12 %).",
    },
    "allgemeine_technische_frage": {
        "q_class": "Q5",
        "dense": True, "bm25": True, "tag": False,
        "hyde": True, "agentic": False,
        "fallback": True,
        "beschreibung": "Begriffs-/Konzeptfrage ohne Produktbezug ('Was ist IO-Link?').",
    },
}

QUERY_CLASSES = tuple(ROUTING_MATRIX)
FALLBACK_CLASS = next(
    name for name, cfg in ROUTING_MATRIX.items() if cfg.get("fallback")
)


def routing_for(query_class: str) -> dict:
    """Liefert die Routing-Konfiguration. Unbekannte Klassen landen
    deterministisch auf der Fallback-Klasse statt in einem Dead End."""
    config = ROUTING_MATRIX.get(query_class)
    if config is None:
        config = ROUTING_MATRIX[FALLBACK_CLASS]
        return {**config, "query_class": FALLBACK_CLASS, "routing_fallback_applied": True}
    return {**config, "query_class": query_class, "routing_fallback_applied": False}


# -- Produktcode-Erkennung ------------------------------------------------

# Whitelist realer Formate aus dem Use-Case-Dokument:
#   2903590000        Weidmüller/Phoenix-Artikelnummer (10-stellig)
#   WDU 2.5           Weidmüller Klemmenbezeichnung
_PRODUCT_CODE_PATTERNS = (
    re.compile(r"\b\d{10}\b"),
    # Alphanumerisch verschachtelte Sensorcodes: O1D302, O5D150, M18E...
    re.compile(r"\b[A-Z]{1,3}\d[A-Z]?\d{2,5}[A-Z]?\b"),
    re.compile(r"\b[A-Z]{2,4}\s?\d{2,6}[A-Z]?\b"),
    re.compile(r"\b[A-Z]{2,4}\s?\d{1,2}\.\d{1,2}\b"),
)

# Normen, Schutzarten und Protokolle mit produktcodeähnlicher Form. Ohne
# diese Ausschlüsse würde "IP67 vs IP69K" als Vergleich zweier Produkte
# erkannt.
_NON_PRODUCT_PREFIXES = (
    "IP", "ATEX", "DIN", "EN", "ISO", "IEC", "SIL", "NEMA", "UL", "VDE",
    "PL", "IO", "USB", "RS", "DC", "AC", "PNP", "NPN", "M12", "M8",
)
_NON_PRODUCT_EXACT = {"IO-LINK", "IOLINK", "ATEX", "PROFINET", "ETHERCAT", "MODBUS"}


def _is_product_code(token: str) -> bool:
    upper = token.upper().replace(" ", "")
    if upper in _NON_PRODUCT_EXACT:
        return False
    letters = "".join(ch for ch in upper if ch.isalpha())
    if letters and letters in _NON_PRODUCT_PREFIXES:
        return False
    if not any(ch.isdigit() for ch in upper):
        return False
    return True


def extract_candidate_products(query: str) -> list[str]:
    """
    Erkennt Produktcode-Kandidaten für TAG-Query und Dual-Retrieval (Q2).

    Duplikate werden entfernt, die Reihenfolge bleibt erhalten; sie
    bestimmt die Spaltenreihenfolge der Vergleichsmatrix.
    
    """
    found: list[str] = []
    for pattern in _PRODUCT_CODE_PATTERNS:
        for match in pattern.findall(query or ""):
            token = match.strip()
            if _is_product_code(token):
                found.append(token)
    return list(dict.fromkeys(found))


# -- Regel-Fallback -------------------------------------------------------

_COMPARISON_MARKERS = (
    "vergleich", "vergleiche", "unterschied", "versus", " vs ", " vs.",
    "gegenüber", "im vergleich zu", "besser als",
)
_PRICE_MARKERS = (
    "preis", "kostet", "kosten", "listenpreis", "verfügbar", "verfügbarkeit",
    "lieferzeit", "eur", "€", "angebot",
)
_SPEC_MARKERS = (
    "schutzart", "messbereich", "spannung", "spezifikation", "technische daten",
    "kenndaten", "welche", "wie hoch", "wie viel", "schnittstelle",
)
_SEARCH_MARKERS = (
    "eignet sich", "geeignet für", "empfehlung", "welcher sensor",
    "sensoren für", "produkte für", "gibt es", "suche",
)


def classify_query_heuristic(query: str) -> dict:
    """
    Regelbasierte Klassifikation (Tests und Fallback, nicht der Produktionspfad).

    Preis wird vor Vergleich geprüft: "Was kostet A im Vergleich zu B?"
    ist eine Preisanfrage und benötigt den Agentic-Pfad.

    Returns:
        Dict mit query_class, q_class, candidate_products,
        requires_dual_retrieval und routing.
    """
    text = (query or "").lower()
    products = extract_candidate_products(query or "")

    if any(marker in text for marker in _PRICE_MARKERS):
        klasse = "preisanfrage"
    elif any(marker in text for marker in _COMPARISON_MARKERS) or len(products) >= 2:
        klasse = "produktvergleich"
    elif products and any(marker in text for marker in _SPEC_MARKERS):
        klasse = "produktspezifikation"
    elif any(marker in text for marker in _SEARCH_MARKERS):
        klasse = "produktsuche"
    elif products:
        klasse = "produktspezifikation"
    else:
        klasse = FALLBACK_CLASS

    routing = routing_for(klasse)
    return {
        "query_class": klasse,
        "q_class": routing["q_class"],
        "candidate_products": products,
        "requires_dual_retrieval": bool(routing.get("dual_retrieval")) and len(products) >= 2,
        "routing": routing,
    }


# Kategorien des n8n-Text-Classifier-Node; müssen mit ROUTING_MATRIX
# übereinstimmen (geprüft in tests/test_hybrid_components.py).
N8N_CLASSIFIER_CATEGORIES = [
    {
        "category": "produktspezifikation",
        "description": "Q1 Attribut-Lookup: Nutzer nennt EIN konkretes Produkt (Produktcode oder Artikelnummer) und fragt nach einem technischen Attributwert, z. B. 'Welche Schutzart hat der Artikel 2903590000?'",
    },
    {
        "category": "produktvergleich",
        "description": "Q2 Produkt-Vergleich: Nutzer nennt ZWEI ODER MEHR konkrete Produkte und moechte sie dimensionsweise gegenueberstellen, z. B. 'Vergleiche O1D302 und Balluff BOS 0035 hinsichtlich Messbereich und Schnittstelle.'",
    },
    {
        "category": "produktsuche",
        "description": "Q3 Konzeptuelle Anfrage: Nutzer sucht ein PASSENDES Produkt fuer einen Anwendungsfall oder ein Kriterium, OHNE einen konkreten Produktcode zu nennen, z. B. 'Welche Sensoren eignen sich fuer Ex-Zonen Kategorie 2G?'",
    },
    {
        "category": "preisanfrage",
        "description": "Q4 Preis/Verfuegbarkeit: Nutzer fragt nach Preis, Listenpreis, Kosten, Lieferzeit oder Verfuegbarkeit eines Produkts, z. B. 'Was kostet der Weidmueller WDU 2.5 aktuell?'",
    },
    {
        "category": "allgemeine_technische_frage",
        "description": "Restklasse: allgemeine technische oder begriffliche Frage ohne Bezug zu einem konkreten Produkt und ohne Produktauswahl-Absicht, z. B. 'Was ist IO-Link?'. NICHT fuer Produktsuchen verwenden.",
    },
]
