# ============================================================
# petra_hybrid/agent.py — Agentic Extension (Schicht 6)
# Ablauf:
#   Zyklus 1  Produktname extrahieren, Query-Varianten bilden,
#             lokale Preislisten prüfen (VektordatenbankTool)
#   Zyklus 2  Web-Suche (DuckDuckGo) mit eskalierenden Query-Varianten
#   Zyklus 3  Preise aus Snippets extrahieren, ggf. Produktseiten abrufen
#             (JSON-LD, Meta-Tags, Fließtext)
#
# Ein aktueller lokaler Preis beendet den Ablauf nach Zyklus 1.
# Web-Suche nur bei PETRA_WEB_AGENT_ENABLED=true.
# ============================================================
from __future__ import annotations

import logging
import contextvars
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .price_extraction import extract_prices as extract_local_prices

logger = logging.getLogger("petra.agent")

MAX_CYCLES = 3
TIMEOUT_SECONDS = 15
MAX_SEARCH_RETRIES = 2
RETRY_BACKOFF_SECONDS = 1.5
PAGE_FETCH_TIMEOUT = 5.0
# s8: Abnahme K6 – feste Suchanbieter statt ddgs-'auto' (Yahoo hängt 8–9 s, bing liefert
# Werbelinks; Messung im Container 04.10.2026: duckduckgo 0,9 s, html 1,2 s)
WEB_BACKENDS = [b.strip() for b in os.getenv("PETRA_WEB_BACKENDS", "duckduckgo,html").split(",") if b.strip()]
WEB_SEARCH_TIMEOUT = float(os.getenv("PETRA_WEB_SEARCH_TIMEOUT", "6"))
_WERBE_LINK = re.compile(r"/aclick|/aclk|doubleclick\.net|googleadservices|bing\.com/ck/|/pagead/", re.I)
MAX_PAGES_TO_FETCH = 3
MAX_WEB_PRICES = 3 

# Domain-Fragmente von Shop- und Preisvergleichsseiten. Diese werden zuerst
# abgerufen: Snippets enthalten selten Preise, die Seiten meist schon
# (JSON-LD/schema.org oder Meta-Tags).

PRIORITY_DOMAIN_HINTS = ("eshop.", "shop.", "idealo.", "preisvergleich", "amazon.")


@dataclass
class AgentStep:
    cycle: int
    thought: str
    action: str
    action_input: dict
    observation: str
    duration_ms: int = 0


@dataclass
class AgentResult:
    answer_context: list[dict] = field(default_factory=list)
    web_sources: list[dict] = field(default_factory=list)
    trace: list[AgentStep] = field(default_factory=list)
    terminated_by: str = "completed"
    warnings: list[str] = field(default_factory=list)
    duration_ms: int = 0
    preise: list[dict] = field(default_factory=list)
    preis_gefunden: bool = False

    def to_dict(self) -> dict:
        return {
            "agent_context": self.answer_context,
            "web_sources": self.web_sources,
            "agent_trace": [
                {
                    "cycle": s.cycle, "thought": s.thought, "action": s.action,
                    "action_input": s.action_input, "observation": s.observation,
                    "duration_ms": s.duration_ms,
                }
                for s in self.trace
            ],
            "agent_terminated_by": self.terminated_by,
            "agent_warnings": self.warnings,
            "agent_duration_ms": self.duration_ms,
            "preise": self.preise,
            "preis_gefunden": self.preis_gefunden,
        }


# s7: Einstellungen – Vorgabe je Anfrage (Frontend-Schalter); None = Umgebung
WEB_AGENT_OVERRIDE: contextvars.ContextVar = contextvars.ContextVar("petra_web_agent", default=None)


def web_agent_enabled() -> bool:
    vorgabe = WEB_AGENT_OVERRIDE.get()
    if isinstance(vorgabe, bool):
        return vorgabe
    return os.getenv("PETRA_WEB_AGENT_ENABLED", "false").lower() in ("true", "1", "yes")


# ------------------------------------------------------------------
# Preis-Extraktion: EIN kombiniertes Pattern-Paar statt 9 sich
# überlappender Varianten. finditer statt findall -> wir haben
# Match-Objekte mit echter Position (kein text.find()-Rateraten).
# ------------------------------------------------------------------
_PRICE_PATTERNS = [
    re.compile(
        r'(?:Listenpreis|Preis)?\s*[:.]?\s*(\d{1,5}(?:[.,]\d{1,2})?)\s*(EUR|€|Euro)',
        re.IGNORECASE,
    ),
    re.compile(
        r'(EUR|€|Euro)\s*(\d{1,5}(?:[.,]\d{1,2})?)',
        re.IGNORECASE,
    ),
]


def extract_prices_from_text(text: str, source_url: str = "", retrieved_at: str = "") -> list[dict]:
    seen: set[tuple[float, str]] = set()
    results: list[dict] = []

    for pattern_idx, pattern in enumerate(_PRICE_PATTERNS):
        for m in pattern.finditer(text):
            groups = m.groups()
            # Reihenfolge der Gruppen unterscheidet sich je nach Pattern
            if pattern_idx == 0:
                betrag_str, waehrung_raw = groups
            else:
                waehrung_raw, betrag_str = groups

            try:
                betrag = float(betrag_str.replace(',', '.'))
            except (ValueError, AttributeError):
                continue

            if betrag <= 0 or betrag > 10000:
                continue

            waehrung = "EUR" if waehrung_raw.upper() in ("EUR", "€", "EURO") else waehrung_raw.upper()
            key = (round(betrag, 2), waehrung)
            if key in seen:
                continue
            seen.add(key)
            # Feldnamen entsprechen dem n8n-Node "Preis-Agent-Ergebnis mappen"
            # (dokument, seite, dokumentdatum, veraltet, warnhinweis).
            results.append({
                "betrag": betrag,
                "waehrung": waehrung,
                "quelle": "web",
                "dokument": source_url or None,
                "seite": None,
                "dokumentdatum": retrieved_at or None,
                "veraltet": False,
                "warnhinweis": None,
                # Alte Feldnamen zusaetzlich behalten fuer
                # Rueckwaertskompatibilitaet mit anderen Aufrufern.
                "url": source_url,
                "retrieved_at": retrieved_at,
            })

    return results


# ------------------------------------------------------------------
# Web-Suche: robustere Fehlerbehandlung + Retry/Backoff.
# Unterscheidet "Paket fehlt/Bug" von "temporär geblockt/Rate-Limit"
# von "wirklich 0 Treffer", damit das im Log auch so ankommt.
# ------------------------------------------------------------------
def _get_ddgs_client():
    """Bevorzugt das aktuelle 'ddgs'-Paket, fällt auf das alte,
    umbenannte 'duckduckgo_search' zurück, falls nur das installiert ist."""
    try:
        from ddgs import DDGS
        return DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
            logger.warning(
                "Paket 'duckduckgo_search' ist umbenannt zu 'ddgs' und wird "
                "nicht mehr gepflegt. Bitte 'pip install ddgs' ausführen."
            )
            return DDGS
        except ImportError:
            return None


def web_search(query: str, max_results: int = 5, region: str = "de-de",
               zeitlimit: float | None = None) -> list[dict]:
    """
    DuckDuckGo-Textsuche mit Retry und linearem Backoff.

    Returns:
        Treffer [{title, url, snippet, retrieved_at, source_kind}]; leere
        Liste, wenn der Web-Agent deaktiviert ist, das Paket fehlt oder
        alle Versuche scheitern.
    """
    if not web_agent_enabled():
        return []

    DDGS = _get_ddgs_client()
    if DDGS is None:
        logger.error("Weder 'ddgs' noch 'duckduckgo_search' ist installiert.")
        return []

    ende = time.time() + (zeitlimit if zeitlimit is not None else WEB_SEARCH_TIMEOUT * len(WEB_BACKENDS or [1]))
    fehler: list[str] = []
    for backend in (WEB_BACKENDS or ["duckduckgo"]):
        rest = ende - time.time()
        if rest < 1.0:
            fehler.append("Zeitbudget aufgebraucht")
            break
        try:
            client = DDGS(timeout=max(1, int(min(WEB_SEARCH_TIMEOUT, rest))))
            hits = list(client.text(query, region=region, max_results=max_results, backend=backend))
        except Exception as exc:  # noqa: BLE001 – nächster Anbieter
            fehler.append(f"{backend}: {type(exc).__name__}: {str(exc)[:80]}")
            continue
        retrieved_at = datetime.now(timezone.utc).date().isoformat()
        ergebnisse = []
        for hit in hits:
            url = hit.get("href") or hit.get("url", "")
            if not url or _WERBE_LINK.search(url):
                continue
            ergebnisse.append({
                "title": hit.get("title", ""),
                "url": url,
                "snippet": hit.get("body", ""),
                "retrieved_at": retrieved_at,
                "source_kind": "web",
                "backend": backend,
            })
        if ergebnisse:
            logger.info("Web-Suche '%s': %d Treffer über %s.", query, len(ergebnisse), backend)
            return ergebnisse
        fehler.append(f"{backend}: 0 verwertbare Treffer")

    logger.warning("Web-Suche ohne Ergebnis für '%s': %s", query, " | ".join(fehler))
    return []


def _prioritize_sources(sources: list[dict]) -> list[dict]:
    """
    Sortiert Shop-/Preisvergleichsseiten nach vorn und behält je Domain nur einen Treffer.

    Ohne Domain-Begrenzung könnte eine einzelne Domain mit Bot-Schutz
    (z. B. HTTP 403) das gesamte Abrufbudget (MAX_PAGES_TO_FETCH) belegen.
    """
    from urllib.parse import urlparse

    def score(src: dict) -> int:
        url = (src.get("url") or "").lower()
        return 0 if any(hint in url for hint in PRIORITY_DOMAIN_HINTS) else 1

    ranked = sorted(sources, key=score)
    seen_domains: set[str] = set()
    diversified: list[dict] = []
    for src in ranked:
        domain = urlparse(src.get("url") or "").netloc
        if domain and domain in seen_domains:
            continue
        seen_domains.add(domain)
        diversified.append(src)
    return diversified


_JSON_LD_RE = re.compile(
    r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.DOTALL | re.IGNORECASE,
)
_META_PRICE_RE = re.compile(
    r'<meta[^>]+(?:itemprop=["\']price["\']|property=["\']product:price:amount["\']|'
    r'property=["\']og:price:amount["\'])[^>]+content=["\']([\d.,]+)["\']',
    re.IGNORECASE,
)


def fetch_price_from_page(url: str, retrieved_at: str = "", timeout: float = PAGE_FETCH_TIMEOUT) -> dict | None:
    """
    Ruft eine Produktseite ab und extrahiert den Preis.

    Reihenfolge: JSON-LD (schema.org Offer) -> Meta-Tags -> Fließtext
    (erste 20.000 Zeichen des HTML).

    Returns:
        Preis-Dict oder None bei Abruffehler bzw. ohne Preis.
    """
    import json as _json

    if not retrieved_at:
        retrieved_at = datetime.now(timezone.utc).date().isoformat()

    try:
        import httpx
        resp = httpx.get(
            url, timeout=timeout, follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; PETRA-Agent/1.0)"},
        )
        resp.raise_for_status()
        html = resp.text
    except Exception as exc:
        logger.warning("Seiten-Abruf fehlgeschlagen für %s: %s", url, exc)
        return None

    # 1. JSON-LD / schema.org Offer — am zuverlässigsten bei Shops
    for block in _JSON_LD_RE.findall(html):
        try:
            data = _json.loads(block)
        except (ValueError, TypeError):
            continue
        for item in (data if isinstance(data, list) else [data]):
            if not isinstance(item, dict):
                continue
            offers = item.get("offers")
            if isinstance(offers, list):
                offers = offers[0] if offers else None
            if isinstance(offers, dict) and "price" in offers:
                try:
                    return {
                        "betrag": float(str(offers["price"]).replace(",", ".")),
                        "waehrung": offers.get("priceCurrency", "EUR"),
                        "quelle": "structured_data",
                        "dokument": url,
                        "seite": None,
                        "dokumentdatum": retrieved_at,
                        "veraltet": False,
                        "warnhinweis": None,
                        "url": url,
                        "retrieved_at": retrieved_at,
                    }
                except ValueError:
                    continue

    # 2. Meta-Tags (og:price:amount / itemprop="price")
    meta_match = _META_PRICE_RE.search(html)
    if meta_match:
        try:
            return {
                "betrag": float(meta_match.group(1).replace(",", ".")),
                "waehrung": "EUR",
                "quelle": "meta_tag",
                "dokument": url,
                "seite": None,
                "dokumentdatum": retrieved_at,
                "veraltet": False,
                "warnhinweis": None,
                "url": url,
                "retrieved_at": retrieved_at,
            }
        except ValueError:
            pass

    # 3. Letzter Fallback: bestehende Fließtext-Regex auf HTML-Ausschnitt
    text_prices = extract_prices_from_text(html[:20000], source_url=url, retrieved_at=retrieved_at)
    if text_prices:
        p = text_prices[0]
        p["quelle"] = "page_text"
        return p

    return None


def build_query_variants(product_name: str) -> list[str]:
    """v3.2-patch: immer mit Herstellername. Ohne 'Weidmüller' ist z. B.
    'PRO ECO3' für eine Suchmaschine mehrdeutig (Treffer: ProSieben, Duden)."""
    name = product_name.strip()
    prefix = "" if "weidm" in name.lower() else "Weidmüller "
    variants = [f'{prefix}"{name}" Preis', f"{prefix}{name} Listenpreis", f"{prefix}{name} kaufen"]
    artnr = re.findall(r"\b\d{10}\b", name) or _ziel_artnr(name)[:1]  # v3.2b-patch
    if artnr:
        variants.insert(0, f"Weidmüller {artnr[0]}")
    return variants


def _ziel_artnr(product_name: str) -> list[str]:
    """v3.2b-patch: Artikelnummer(n) des exakt aufgelösten Produkts."""
    try:
        from backend.product_router import resolve_products
        r = resolve_products([product_name], product_name)
        return r.get("artikelnummern", [])[:3] if r.get("exakt") else []
    except Exception:  # noqa: BLE001
        return []


_REL_STOP = {"DER", "DIE", "DAS", "UND", "MIT", "FÜR", "FUER", "PREIS", "AKTUELL", "WIE", "WAS"}


def is_relevant_source(src: dict, product_name: str) -> bool:
    """v3.2-patch: Treffer nur, wenn Hersteller/Artikelnummer UND ein
    markantes Produkt-Token (mit Ziffer) in Titel/Snippet/URL stehen."""
    hay = " ".join([src.get("title", ""), src.get("snippet", ""), src.get("url", "")]).upper()
    hay_norm = hay.replace(" ", "")
    _arts = _ziel_artnr(product_name)  # v3.2b-patch: genau diese Artikelnummer
    if _arts:
        return any(x in hay for x in _arts)
    has_maker = "WEIDM" in hay or bool(re.search(r"\b\d{10}\b", hay))
    tokens = [t for t in re.split(r"[\s/]+", product_name.upper())
              if t and t not in _REL_STOP and any(ch.isdigit() for ch in t)]
    has_token = any(t in hay_norm for t in tokens) if tokens else True
    return has_maker and has_token


# BUGFIX (Regression aus vorheriger Version): [A-Z]+ ohne Wortgrenze am
# Ende kann mitten in ein grossgeschriebenes Substantiv wie "Preis"
# hineinschneiden und nur den Anfangsbuchstaben ("P") als vermeintliche
# roemische Ziffer/Suffix mitnehmen -> Produktname "... II P" statt
# "... II". Die negative Lookahead (?![a-zäöüß]) verhindert das: der
# Grossbuchstaben-Lauf darf nicht unmittelbar von einem Kleinbuchstaben
# desselben Wortes gefolgt sein.
_PRODUCT_RE = re.compile(
    r'([A-Za-zÄÖÜäöüß0-9\s\-]+?\d+\s*W\s+\d+\s*V\s+\d+\s*A(?:\s+[A-Z]+(?![a-zäöüß]))*)'
)

# Fuellphrasen, die trotz erfolgreichem Produkt-Match noch am Anfang
# haengen bleiben koennen (z. B. "Was kostet das PRO ECO3 ..." ->
# die Zeichenklasse oben erlaubt bewusst Kleinbuchstaben fuer Umlaute
# in Herstellernamen wie "Weidmüller" und laesst dadurch auch
# Fragewoerter durch). Wird iterativ angewendet, damit auch
# Kombinationen aus Frage + Artikel ("Was kostet das ") vollstaendig
# verschwinden.
_LEADING_FILLERS_RE = re.compile(
    r'^\s*(?:Was kostet|Wie viel kostet|Was kosten|Preis für|Preis von|'
    r'Kosten für|Kosten von|Wie hoch ist der Preis (?:für|von)|'
    r'der|die|das|the|price of|cost of)\s+',
    re.IGNORECASE,
)


def _strip_leading_fillers(name: str) -> str:
    prev = None
    while prev != name:
        prev = name
        name = _LEADING_FILLERS_RE.sub('', name, count=1).strip()
    return name



def extract_product_name(query: str) -> str:
    """Generalisiert: beliebige Watt-/Volt-/Ampere-Angabe statt
    fester Whitelist (240W|120W|480W|960W). Fuellwoerter werden in
    JEDEM Pfad (Regex-Treffer und Fallback) einheitlich entfernt."""
    # v3.2-patch: zuerst das Produkt-Lexikon (erkennt alle indexierten
    # Produktnamen, nicht nur Netzteile mit W/V/A-Angabe).
    try:
        from backend.product_router import _load_lexicon, names_in_query
        _treffer = [o for o, _k, e in names_in_query(query, _load_lexicon().get("by_name", {})) if e]
        if _treffer:
            return max(_treffer, key=len)
    except Exception:  # noqa: BLE001
        pass
    product_match = _PRODUCT_RE.search(query)
    if product_match:
        name = product_match.group(1).strip()
        return _strip_leading_fillers(name)

    # BUGFIX: Fillerwoerter ('Was kostet die ' etc.) MUESSEN vor der
    # Laengenkuerzung entfernt werden, sonst geht das Zeichenbudget
    # fuer die Frageformulierung drauf statt fuer den Produktnamen.
    name = _strip_leading_fillers(query.split('?')[0].strip())
    return name[:50]


def run_price_agent(
    query: str,
    local_retrieval,
    max_cycles: int = MAX_CYCLES,
    timeout_seconds: int = TIMEOUT_SECONDS,
) -> AgentResult:
    started = time.time()
    result = AgentResult()
    cycles = min(max_cycles, MAX_CYCLES)

    def remaining() -> float:
        return timeout_seconds - (time.time() - started)

    # -- Zyklus 1: Produktname extrahieren & Query-Varianten bauen ---
    step_start = time.time()
    product_name = extract_product_name(query)
    query_variants = build_query_variants(product_name)
    logger.info("Agent: Produktname='%s', Query-Varianten=%s", product_name, query_variants)

    result.trace.append(
        AgentStep(
            cycle=1,
            thought="Produktname extrahiert, eskalierende Query-Varianten vorbereitet.",
            action="QueryBuilder",
            action_input={"product_name": product_name, "variants": query_variants},
            observation=f"{len(query_variants)} Query-Varianten erzeugt.",
            duration_ms=int((time.time() - step_start) * 1000),
        )
    )

    # -- Zyklus 1b: Lokale Preisliste pruefen (BUGFIX) ----------------
    # Vorher wurde `local_retrieval` als Parameter entgegengenommen,
    # aber im gesamten Funktionskoerper nie aufgerufen -> der Agent
    # sprang bei JEDER Preisanfrage direkt ins Web, auch wenn die
    # Preisliste laengst lokal importiert war (siehe Kap. 4.4.2:
    # Zyklus 1 MUSS zuerst das VektordatenbankTool befragen).
    step_start = time.time()
    local_chunks: list[dict] = []
    local_ergebnis = None
    try:
        local_chunks = local_retrieval(product_name) or []
    except Exception as exc:  # noqa: BLE001 — Retrieval darf den Agenten nicht crashen
        logger.warning("Lokales Retrieval im Preis-Agenten fehlgeschlagen: %s", exc)
        result.warnings.append(f"local_retrieval_failed: {exc}")

    if local_chunks:
        local_ergebnis = extract_local_prices(local_chunks)

    lokal_aktuell_gefunden = bool(
        local_ergebnis and local_ergebnis.gefunden and not local_ergebnis.web_recherche_empfohlen
    )

    result.trace.append(
        AgentStep(
            cycle=1,
            thought=(
                "Aktuelle Preisangabe in lokaler Dokumentenbasis gefunden; "
                "Web-Recherche nicht erforderlich."
                if lokal_aktuell_gefunden
                else "Lokale Preisliste geprueft; kein aktueller lokaler Preis "
                     "gefunden oder Angabe veraltet. Web-Recherche erforderlich."
            ),
            action="VektordatenbankTool",
            action_input={"query": product_name, "chunks_gefunden": len(local_chunks)},
            observation=(local_ergebnis.meldung if local_ergebnis
                         else "Keine Preis-Chunks in lokaler Dokumentenbasis gefunden."),
            duration_ms=int((time.time() - step_start) * 1000),
        )
    )

    if local_ergebnis and local_ergebnis.gefunden:
        # Auch bei veraltetem lokalem Preis als Kontext mitgeben, damit
        # die Antwort beide (alten + neuen) Preis nennen kann, analog
        # zum Beispiel-Durchlauf in Kap. 4.4.2.
        result.preise = [p.to_dict() for p in local_ergebnis.preise]
        result.preis_gefunden = True
        result.answer_context.append({
            "chunk_id": local_ergebnis.preise[0].chunk_id or "local-price",
            "text": (
                f"Gefundener Preis (lokale Dokumentenbasis): "
                f"{local_ergebnis.preise[0].betrag:.2f} "
                f"{local_ergebnis.preise[0].waehrung} "
                f"[Quelle: {local_ergebnis.preise[0].dokument or 'lokale Dokumentenbasis'}"
                f"{f', S. {local_ergebnis.preise[0].seite}' if local_ergebnis.preise[0].seite else ''}"
                f", Stand {local_ergebnis.preise[0].dokumentdatum or 'unbekannt'}]"
            ),
            "metadata": {
                "document": local_ergebnis.preise[0].dokument,
                "source_type": "local",
                "page": local_ergebnis.preise[0].seite,
                "dokumentdatum": local_ergebnis.preise[0].dokumentdatum,
                "veraltet": local_ergebnis.preise[0].veraltet,
            },
        })

    # v3.4-patch P8: die komplette Preiszeile (inkl. Preiseinheit/VPE)
    # als Kontext, nicht nur "Gefundener Preis: <Betrag>".
    if local_ergebnis and local_ergebnis.gefunden and result.answer_context:
        _zeilen = [c for c in local_chunks if (c.get("metadata") or {}).get("preiszeile")][:3]
        if _zeilen:
            _p = local_ergebnis.preise[0]
            result.answer_context[-1]["text"] = (
                f"Lokale Preisliste ({_p.dokument or 'Preisliste'}, S. {_p.seite or '?'}, "
                f"Stand: {_p.dokumentdatum or 'undatiert'}):\n"
                + "\n".join(z["text"] for z in _zeilen))
            _m = result.answer_context[-1].setdefault("metadata", {})
            _m["artikelnummer"] = _zeilen[0]["metadata"].get("artikelnummer")
            _m["produkt"] = _zeilen[0]["metadata"].get("produkt")
            _m["preiseinheit"] = _zeilen[0]["metadata"].get("preiseinheit")

    if lokal_aktuell_gefunden:
        # UC-04: aktueller lokaler Preis reicht aus — Web-Recherche
        # entfaellt, spart 10-30s Latenz fuer den haeufigen Fall.
        result.terminated_by = "completed"
        result.duration_ms = int((time.time() - started) * 1000)
        return result

    # -- Zyklus 2: Web-Suche mit Eskalation ---------------------------
    if cycles >= 2 and remaining() > 0:
        step_start = time.time()
        used_query = None
        if web_agent_enabled():
            for variant in query_variants:
                if remaining() <= 0:
                    break
                sources = [s for s in web_search(variant, max_results=8, zeitlimit=remaining())
                           if is_relevant_source(s, product_name)][:5]  # v3.2-patch
                if sources:
                    result.web_sources = sources
                    used_query = variant
                    break
            if used_query:
                observation = f"Treffer mit Query '{used_query}': {len(result.web_sources)} Ergebnisse."
            else:
                observation = f"Alle {len(query_variants)} Query-Varianten lieferten 0 Ergebnisse."
                result.warnings.append("web_search_no_results_all_variants")
        else:
            observation = "Web-Suche deaktiviert."
            result.warnings.append("web_agent_disabled")

        result.trace.append(
            AgentStep(
                cycle=2,
                thought="Web-Suche mit eskalierenden Query-Varianten ausführen.",
                action="WebSearchTool",
                action_input={"variants_tried": query_variants, "used_query": used_query},
                observation=observation,
                duration_ms=int((time.time() - step_start) * 1000),
            )
        )

    # -- Zyklus 3: Preise extrahieren ---------------------------------
    if cycles >= 3 and remaining() > 0 and result.web_sources:
        step_start = time.time()
        # v3.4-patch P7/P7b/P14: bis zu MAX_WEB_PRICES Preise, je Händler-
        # Domain einer, jeder mit eigener Quelle (UC-04 A2).
        from urllib.parse import urlparse
        from .price_extraction import _PRICE_PATTERNS as _PX, _parse_amount

        def _dom(u: str) -> str:
            return urlparse(u or "").netloc.lower().replace("www.", "")

        _rauschen = re.compile(
            r"[^.|\n]*(?:versand|porto|zuschlag|mindermenge|shipping|lieferkosten)[^.|\n]*",
            re.IGNORECASE)

        def _snippet_preis(src: dict) -> dict | None:
            text = _rauschen.sub(" ", f"{src.get('title', '')} {src.get('snippet', '')}")
            for pat in _PX:
                for m in pat.finditer(text):
                    betrag = _parse_amount(m.group(1))
                    if betrag and 0 < betrag <= 100000:
                        return {
                            "betrag": betrag, "waehrung": "EUR", "quelle": "snippet",
                            "dokument": src.get("url") or None, "seite": None,
                            "dokumentdatum": src.get("retrieved_at") or None,
                            "veraltet": False, "warnhinweis": None,
                            "url": src.get("url", ""), "retrieved_at": src.get("retrieved_at", ""),
                        }
            return None

        all_prices: list[dict] = []
        doms: set[str] = set()
        for src in result.web_sources:
            d = _dom(src.get("url", ""))
            if not d or d in doms:
                continue
            p = _snippet_preis(src)
            if p:
                all_prices.append(p)
                doms.add(d)
            if len(all_prices) >= MAX_WEB_PRICES:
                break

        pages_fetched = 0
        fetch_attempts: list[str] = []
        for src in _prioritize_sources(result.web_sources):
            if len(all_prices) >= MAX_WEB_PRICES or pages_fetched >= MAX_PAGES_TO_FETCH:
                break
            if remaining() <= 1:
                fetch_attempts.append("abgebrochen: Zeitbudget aufgebraucht")
                break
            url = src.get("url", "")
            d = _dom(url)
            if not url or d in doms:
                continue
            pages_fetched += 1
            page_price = fetch_price_from_page(
                url, retrieved_at=src.get("retrieved_at", ""),
                timeout=max(1.0, min(PAGE_FETCH_TIMEOUT, remaining() - 1)))
            # Nur strukturierte Daten (JSON-LD/Meta) - Freitext-Regex auf HTML
            # trifft zu oft Versand- oder Zubehörpreise ("entweder korrekt oder nichts").
            if page_price and page_price.get("quelle") in ("structured_data", "meta_tag"):
                all_prices.append(page_price)
                doms.add(d)
                fetch_attempts.append(f"{url}: Preis gefunden ({page_price['quelle']})")
            else:
                fetch_attempts.append(f"{url}: kein strukturierter Preis")
        if fetch_attempts:
            result.warnings.append("page_fetch_attempts: " + " | ".join(fetch_attempts))

        context_chunks = []
        if all_prices:
            all_prices.sort(key=lambda x: x["betrag"])
            for i, p in enumerate(all_prices, 1):
                context_chunks.append({
                    "chunk_id": f"web-price-{i}",
                    "text": (f"Web-Preis {i} von {len(all_prices)}: {p['betrag']:.2f} {p['waehrung']} "
                             f"[Quelle: {p.get('url', 'Web')}, abgerufen {p.get('retrieved_at', '')}; "
                             f"Preiseinheit laut Händlerseite prüfen]"),
                    "metadata": {
                        "document": p.get("url", ""), "source_type": "web",
                        "url": p.get("url", ""), "abrufdatum": p.get("retrieved_at", ""),
                    },
                })
            result.preise = list(result.preise or []) + all_prices
            result.preis_gefunden = True
            observation = (f"{len(all_prices)} Web-Preis(e) von {len(doms)} Händler(n) "
                           f"({pages_fetched} Seite(n) direkt abgerufen).")
        else:
            context_chunks.append({
                "chunk_id": "no-price",
                "text": "In den Web-Suchergebnissen wurde kein Preis gefunden. Bitte prüfe die angegebenen Quellen direkt.",
                "metadata": {"source_type": "web", "document": "Web-Recherche"}
            })
            result.preis_gefunden = bool(result.preise)
            observation = "Kein Preis in Web-Ergebnissen gefunden."

        result.answer_context.extend(context_chunks)
        result.trace.append(
            AgentStep(
                cycle=3,
                thought="Preise aus Web-Ergebnissen extrahieren (dedupliziert).",
                action="PreisExtraktion",
                action_input={"quellen": len(result.web_sources)},
                observation=observation,
                duration_ms=int((time.time() - step_start) * 1000),
            )
        )

    # -- Terminierung ------------------------------------------------
    if remaining() <= 0:
        result.terminated_by = "timeout"
        result.warnings.append(f"Agent nach {timeout_seconds}s Timeout beendet.")
    elif len(result.trace) >= cycles:
        result.terminated_by = "max_cycles"

    result.duration_ms = int((time.time() - started) * 1000)
    return result