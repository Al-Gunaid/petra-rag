# ============================================================
# backend/product_router.py — Produkt-Lexikon + produktgefiltertes Retrieval
#
# ARCHITEKTURBEZUG: Kap. 4.7.2 Stufe 2 "Graph-Metadaten-Layer (JSON)",
# Kap. 4.5.1 Achse 3 (Wissenstyp-Komplementarität), UC-01/UC-02.
#
# PROBLEM: Die Anfrage nennt "WDU 2.5", die Datenblätter heißen aber
# 1020000000_de.pdf. Ohne Auflösung Name → Artikelnummer konkurrieren
# alle Varianten (WDU 2.5 GN, WDU 2.5/TC TYP J, ...) um dieselben
# Top-k-Plätze; das richtige Datenblatt erreichte den Kontext nur in 34 %.
#
# ENDPUNKTE:
#   POST /product/resolve   {products:[...], query}   → Artikelnummern
#   POST /product/retrieve  {products:[...], query, top_k}
#        → Dense-Suche NUR innerhalb der Chunks der aufgelösten Artikel
#          (Chroma where-Filter auf metadata.artikelnummer).
#
# Einbindung in api_server.py (zwei Zeilen):
#   from backend.product_router import product_router
#   app.include_router(product_router)
# ============================================================
from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter

logger = logging.getLogger("petra.product_router")

LEXICON_PATH = os.getenv("PETRA_PRODUCT_LEXICON", "/data/product_lexicon.json")
OLLAMA_URL = os.getenv("PETRA_OLLAMA_URL", "http://ollama:11434")
EMBEDDING_MODEL = os.getenv("PETRA_EMBEDDING_MODEL", "bge-m3:latest")
CHROMA_HOST = os.getenv("CHROMA_HOST", "chromadb")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
COLL_TEXT = os.getenv("PETRA_COLLECTION_TEXT_V3", "petra_text_chunks_v3")
COLL_TAG = os.getenv("PETRA_COLLECTION_TAG_V3", "petra_tag_chunks_v3")
MAX_VARIANTS = int(os.getenv("PETRA_MAX_VARIANTS", "6"))

product_router = APIRouter(prefix="/product", tags=["product"])

_lock = threading.Lock()
_lexicon: dict[str, Any] | None = None
_lexicon_mtime: float = 0.0
_chroma = None

_ARTNR_RE = re.compile(r"\b\d{10}\b")


def norm_name(name: str) -> str:
    """Identisch zu reindex_v3_contextual.norm_name – NICHT abweichen lassen."""
    return re.sub(r"\s+", "", (name or "").upper().replace(",", "."))


# ── v3.2: Produktnamen direkt aus der Anfrage erkennen ──────────────────
# Die Regex-Extraktion in n8n ("Routing & Produktextraktion") arbeitet nur
# mit Großbuchstaben-Mustern und verfehlt Namen wie "A2C 2.5",
# "Klippon POK 080806" oder "STRIPAX" bzw. liefert Fragmente ("UR20",
# "HA 10"). Hier wird die Anfrage gegen das Lexikon selbst gematcht:
# Token-N-Gramme, längster Treffer zuerst, nur an Wortgrenzen.
_STRIP = "\"'„“”‚‘?!,;:()[]{}"
_sorted_keys: list[str] = []
_sorted_keys_src: int = 0


def _keys_sorted(by_name: dict) -> list[str]:
    global _sorted_keys, _sorted_keys_src
    if _sorted_keys_src != id(by_name):
        _sorted_keys = sorted(by_name)
        _sorted_keys_src = id(by_name)
    return _sorted_keys


def _prefix_matches(keys: list[str], prefix: str, limit: int) -> list[str]:
    import bisect
    i = bisect.bisect_left(keys, prefix)
    out = []
    while i < len(keys) and keys[i].startswith(prefix):
        k = keys[i]
        # Ziffer direkt nach einer Ziffer = anderer Wert ('WDU2.5' ≠ 'WDU2.55');
        # nach einem Buchstaben beginnt dagegen ein neues Namenssegment
        # ('SL5.08HC/09/90G' + '3.2SNBKBX').
        if len(k) > len(prefix) and not (k[len(prefix)].isdigit() and prefix[-1].isdigit()):
            out.append(k)
            if len(out) > limit:
                break
        i += 1
    return out


def names_in_query(query: str, by_name: dict, max_n: int = 8) -> list[tuple[str, str, bool]]:
    """Findet Lexikon-Namen in der Anfrage.

    Rückgabe: [(Originaltext, Lexikon-Schlüssel oder Präfix, exakt)].
    Regeln gegen Fehltreffer:
      - Schlüssel ohne Ziffer nur, wenn ≥ 6 Zeichen UND im Original ≥ 2
        Großbuchstaben (STRIPAX, PrintJet – nicht "Klemme").
      - Kein Treffer, wenn das nächste Token mit einer Ziffer beginnt
        (der Name geht dann vermutlich weiter, z. B. "HDC-KIT-HA 10.110").
      - Präfix-Varianten nur für lange Schlüssel (≥ 8 Zeichen, mit Ziffer)
        und nur, wenn höchstens MAX_VARIANTS Varianten existieren.
    """
    toks = [t.strip(_STRIP) for t in (query or "").split()]
    toks = [t[:-1] if t.endswith(".") and not re.search(r"\d\.$", t[:-1] + ".") else t for t in toks]
    n = len(toks)
    used: set[int] = set()
    found: list[tuple[str, str, bool]] = []

    def ok(seg: list[str], key: str) -> bool:
        if len(key) < 4:
            return False
        if not re.search(r"\d", key):
            upp = sum(ch.isupper() for ch in "".join(seg))
            return len(key) >= 6 and upp >= 2
        return True

    spans = [(L, i) for L in range(min(max_n, n), 0, -1) for i in range(n - L + 1)]
    for L, i in spans:                                   # 1) exakt
        if any(j in used for j in range(i, i + L)) or not toks[i] or not toks[i + L - 1]:
            continue
        seg = toks[i:i + L]
        key = norm_name("".join(seg))
        if key in by_name and ok(seg, key):
            if i + L < n and toks[i + L][:1].isdigit():
                continue
            found.append((" ".join(seg), key, True))
            used.update(range(i, i + L))
    keys = _keys_sorted(by_name)
    for L, i in spans:                                   # 2) Präfix (Varianten)
        if any(j in used for j in range(i, i + L)) or not toks[i] or not toks[i + L - 1]:
            continue
        seg = toks[i:i + L]
        key = norm_name("".join(seg))
        if len(key) < 8 or not re.search(r"\d", key):
            continue
        var = _prefix_matches(keys, key, MAX_VARIANTS)
        if 0 < len(var) <= MAX_VARIANTS:
            found.append((" ".join(seg), key, False))
            used.update(range(i, i + L))
    return found


def _load_lexicon() -> dict[str, Any]:
    """Lädt das Lexikon und lädt es neu, wenn die Datei sich ändert
    (nach einer erneuten Re-Indexierung ist kein Neustart nötig)."""
    global _lexicon, _lexicon_mtime
    path = Path(LEXICON_PATH)
    with _lock:
        if not path.exists():
            if _lexicon is None:
                logger.warning("Produkt-Lexikon fehlt: %s (reindex_v3_contextual.py ausführen)", path)
            return _lexicon or {"by_name": {}, "by_artnr": {}}
        mtime = path.stat().st_mtime
        if _lexicon is None or mtime != _lexicon_mtime:
            with path.open("r", encoding="utf-8") as fh:
                _lexicon = json.load(fh)
            _lexicon_mtime = mtime
            logger.info("Produkt-Lexikon geladen: %d Produkte, %d Namen.",
                        len(_lexicon.get("by_artnr", {})), len(_lexicon.get("by_name", {})))
        return _lexicon


def _get_chroma():
    global _chroma
    if _chroma is None:
        import chromadb
        _chroma = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    return _chroma


def resolve_products(products: list[str], query: str = "") -> dict[str, Any]:
    """Name → Artikelnummer(n).

    Reihenfolge der Strategien:
      1. 10-stellige Artikelnummer (in products ODER in der Anfrage)
      2. exakter Namenstreffer (normalisiert: 'WDU 2,5' == 'WDU 2.5')
      3. Präfix-Treffer = Varianten ('WDU 2.5' → 'WDU 2.5 GN', ...),
         nur wenn kein exakter Treffer existiert, max. MAX_VARIANTS.
    """
    lex = _load_lexicon()
    by_name: dict[str, list[dict]] = lex.get("by_name", {})
    by_artnr: dict[str, dict] = lex.get("by_artnr", {})

    resolved: dict[str, dict] = {}
    artnrs: list[str] = []
    infos: list[dict] = []

    def add(entry: dict) -> None:
        a = entry.get("artikelnummer")
        if a and a not in artnrs:
            artnrs.append(a)
            infos.append({k: entry.get(k, "") for k in ("artikelnummer", "produkt", "ausfuehrung", "document")})

    cleaned = [str(p).strip().strip("\"'„“”?!.,;:()[]").strip() for p in (products or [])]
    aus_anfrage = names_in_query(query, by_name) if query else []
    candidates = list(dict.fromkeys(
        [orig for orig, _k, _e in aus_anfrage] + [c for c in cleaned if c] + _ARTNR_RE.findall(query or "")))
    for name in candidates:
        n = norm_name(name)
        if _ARTNR_RE.fullmatch(n):
            entry = by_artnr.get(n)
            resolved[name] = {"artikelnummern": [n], "exakt": True, "im_bestand": entry is not None}
            add(entry or {"artikelnummer": n})
            continue
        if n in by_name:
            entries = by_name[n]
            resolved[name] = {"artikelnummern": [e["artikelnummer"] for e in entries], "exakt": True}
            for e in entries:
                add(e)
            continue
        # Präfix: Variante beginnt mit dem Namen und das nächste Zeichen ist
        # KEINE Ziffer ('WDU2.5' passt zu 'WDU2.5GN', nicht zu 'WDU2.55').
        # v3.2: Fragmente ("UR20", "HA10") und mehrdeutige Präfixe liefern
        # KEINE Varianten mehr – früher wurden die ersten 6 von u. U.
        # hunderten Treffern genommen, also zufällige Fremdprodukte.
        variants = _prefix_matches(_keys_sorted(by_name), n, MAX_VARIANTS) if len(n) >= 5 else []
        if len(variants) > MAX_VARIANTS:
            variants = []
        variants.sort(key=len)
        resolved[name] = {
            "artikelnummern": [e["artikelnummer"] for k in variants for e in by_name[k]],
            "exakt": False,
            "varianten": [by_name[k][0].get("produkt", k) for k in variants],
        }
        for k in variants:
            for e in by_name[k]:
                add(e)

    # v3.1b: Sobald EIN Kandidat exakt aufgelöst ist, werden Präfix-Varianten
    # anderer (unsauber extrahierter) Kandidaten verworfen. Sonst verdrängen
    # z. B. 'WDU 2.5/1.5/ZR', 'WDU 2.5 GN' ... das gefragte Datenblatt aus den
    # Top-k-Plätzen (beobachtet in der Streamlit-Evaluation, Fall 1).
    exakt_keys = [k for k, r in resolved.items() if r.get("exakt")]
    if exakt_keys:
        keep = {a for k in exakt_keys for a in resolved[k]["artikelnummern"]}
        for k, r in resolved.items():
            if not r.get("exakt") and r.get("artikelnummern"):
                r["verworfen"] = True
        artnrs = [a for a in artnrs if a in keep]
        infos = [i for i in infos if i.get("artikelnummer") in keep]
    logger.info("Produkt-Auflösung: Kandidaten=%s -> %s (exakt=%s)",
                candidates, artnrs, bool(exakt_keys))
    return {"resolved": resolved, "artikelnummern": artnrs, "produkt_info": infos,
            "exakt": bool(exakt_keys)}


def _embed_query(text: str) -> list[float]:
    with httpx.Client(timeout=20.0) as client:
        resp = client.post(f"{OLLAMA_URL}/api/embed", json={"model": EMBEDDING_MODEL, "input": text})
        if resp.status_code == 404:   # ältere Ollama-Versionen
            resp = client.post(f"{OLLAMA_URL}/api/embeddings", json={"model": EMBEDDING_MODEL, "prompt": text})
            resp.raise_for_status()
            vec = resp.json()["embedding"]
        else:
            resp.raise_for_status()
            vec = resp.json()["embeddings"][0]
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _query_collection(name: str, emb: list[float], artnrs: list[str], n: int) -> list[dict]:
    try:
        coll = _get_chroma().get_collection(name)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Collection %s nicht verfügbar: %s", name, exc)
        return []
    where = {"artikelnummer": artnrs[0]} if len(artnrs) == 1 else {"artikelnummer": {"$in": artnrs}}
    res = coll.query(query_embeddings=[emb], n_results=n, where=where,
                     include=["documents", "metadatas", "distances"])
    out = []
    for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
        out.append({"chunk_id": cid, "text": doc or "", "metadata": meta or {},
                    "score": round(max(0.0, 1.0 - float(dist)), 4)})
    return out


@product_router.post("/resolve")
def resolve_endpoint(payload: dict) -> dict:
    return resolve_products(payload.get("products") or [], payload.get("query", ""))


@product_router.post("/retrieve")
def retrieve_endpoint(payload: dict) -> dict:
    """Dense-Suche innerhalb der Chunks der aufgelösten Produkte.

    Liefert dasselbe Format wie /retrieval/bm25, damit der n8n-Normalizer
    unverändert wiederverwendet werden kann.
    """
    query = payload.get("query", "")
    top_k = int(payload.get("top_k", 8))
    res = resolve_products(payload.get("products") or [], query)
    artnrs = res["artikelnummern"]
    if not artnrs:
        return {**res, "results": [], "count": 0,
                "hinweis": "Kein Produkt im Lexikon aufgelöst – normaler Hybrid-Pfad."}
    try:
        emb = _embed_query(query)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Query-Embedding fehlgeschlagen: %s", exc)
        return {**res, "results": [], "count": 0, "warning": f"Embedding fehlgeschlagen: {exc}"}

    n_tag = max(2, top_k // 2)
    pool = _query_collection(COLL_TEXT, emb, artnrs, top_k * 3) + \
        _query_collection(COLL_TAG, emb, artnrs, n_tag * 2)
    hits = select_by_section(pool, query, top_k + n_tag)
    for i, h in enumerate(hits, start=1):
        h["rank"] = i
    return {**res, "results": hits, "count": len(hits)}


# ── v3.2: Abschnittsbewusste Auswahl ────────────────────────────────────
# Evaluation 28.09.: 27 % aller Kontexte stammten aus "Zubehör", 18 % aus
# "Allgemeine Bestelldaten", 11 % waren zerlegte Zubehör-Tabellen
# ("|Best.-Nr.|Best.-Nr.|…", "Col4"). Sie verdrängen die Abschnitte mit
# den gefragten Werten (Bemessungsdaten, Technische Daten …) aus dem
# Kontextfenster. Zubehör bleibt voll verfügbar, wenn danach gefragt wird.
_ZUBEHOER_FRAGE = re.compile(
    r"zubeh|artikelnummer|art\.-?nr|bestell|passend|passt|querverbind|abschlussplatte|endwinkel|"
    r"markier|montageplatte|erdungsbolzen|adapter|kompatib|welche .*nehme|stückliste|\b\d{10}\b",
    re.IGNORECASE)
_JUNK_TABLE = re.compile(r"\|Col\d+\||(?:Best\.-Nr\.\|){3}|(?:GTIN \(EAN\)\|){3}|(?:VPE\|){3}")
_ABSCHNITT = re.compile(r"Abschnitt:\s*([^\]|]+)")
_CAP = {"zubehör": 2, "allgemeine bestelldaten": 2}


def _section(text: str) -> str:
    m = _ABSCHNITT.search((text or "")[:300])
    return m.group(1).strip().lower() if m else ""


def select_by_section(pool: list[dict], query: str, k: int) -> list[dict]:
    zubehoer_gefragt = bool(_ZUBEHOER_FRAGE.search(query or ""))
    seen, rest = set(), []
    for h in pool:
        if h["chunk_id"] in seen:
            continue
        seen.add(h["chunk_id"])
        sec = _section(h["text"])
        h.setdefault("metadata", {})["abschnitt_v3"] = sec
        junk = bool(_JUNK_TABLE.search(h["text"] or ""))
        score = h["score"]
        if not zubehoer_gefragt:
            if junk:
                score *= 0.5
            elif sec in _CAP:
                score *= 0.85
        rest.append((score, junk, sec, h))
    rest.sort(key=lambda t: t[0], reverse=True)
    out, count = [], {}
    for score, junk, sec, h in rest:
        if not zubehoer_gefragt:
            if junk and len(rest) > k:
                continue
            if sec in _CAP:
                count[sec] = count.get(sec, 0) + 1
                if count[sec] > _CAP[sec]:
                    continue
        out.append(h)
        if len(out) >= k:
            break
    if len(out) < k:                     # auffüllen, falls zu streng gefiltert
        ids = {h["chunk_id"] for h in out}
        out += [h for _s, _j, _sec, h in rest if h["chunk_id"] not in ids][: k - len(out)]
    return out


# ── v3.2: Produktvergleich (UC-02) über das Lexikon ─────────────────────
# /retrieval/dual suchte bisher per BM25 nach den Regex-Kandidaten aus n8n.
# Die sind bei Vergleichsfragen oft Fragmente ("HA 10", "HE 10", "MIT 12",
# "TYP 1"), die BM25-Treffer gehörten dann zu beliebigen Produkten und die
# daraus gebaute Differenzmatrix war falsch. Hier: beide Produkte über das
# Lexikon auflösen und je Produkt NUR in dessen Datenblatt suchen.
# v3.6b (V11): Allgemeine Vergleichsfragen ohne genanntes Merkmal werden fuer
# Kandidatensuche und Reranking je Produkt um die UC-03-Standarddimensionen
# ergaenzt (= Schluessel von difference_matrix.STANDARD_DIMENSIONEN).
_VERGLEICH_WORT_RE = re.compile(
    r"unterschied|unterscheid|vergleich|gegen(?:ü|ue)ber|austausch|umstell|ersetz|\bstatt\b|anstelle|\bvs\.?(?:\s|$)",
    re.IGNORECASE)
_MERKMAL_RE = re.compile(
    r"strom|spannung|querschnitt|schutzart|\bip\s?\d|temperatur|abmessung|breite|h(?:ö|oe)he|tiefe|gewicht|"
    r"material|drehmoment|abisolier|leistung|frequenz|kontakt|polzahl|messbereich|schnittstelle|zulassung|"
    r"norm|farbe|l(?:ä|ae)nge|anschlussart|schirm|led|protokoll",
    re.IGNORECASE)
_UC03_DIMENSIONEN = ("Schutzart, Messbereich, Betriebsspannung, Schnittstelle, "
                     "Umgebungstemperatur, Anschlussquerschnitt, Bemessungsstrom, Material")


def _rerank_frage(query: str, name: str) -> str:
    q = query or ""
    if _VERGLEICH_WORT_RE.search(q) and not _MERKMAL_RE.search(q):
        return f"{q} Relevante technische Daten von {name}: {_UC03_DIMENSIONEN}."
    return q


def _emb_vergleich(query: str, name: str, emb: list[float]) -> list[float]:
    q = _rerank_frage(query, name)
    if q == (query or ""):
        return emb
    try:
        return _embed_query(q)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Vergleich: erweitertes Query-Embedding fehlgeschlagen (%s): %s", name, exc)
        return emb


# v3.7 (R4): Zertifikats-/ATEX-Abschnitte abwerten, wenn nicht danach gefragt
# (Amiraz et al. 2025, Distracting Effect). Gleiche Muster wie im n8n-CRAG-Filter.
_CERT_TEXT = re.compile(r"ATEX|IECEx|60079|Zertifikat|Ex-Kennzeichnung|\bEx\s?e[bc]?\b", re.IGNORECASE)
_CERT_ABSCHNITT = re.compile(r"Abschnitt:\s*[^\]|]*(zulassung|zertifi|atex|iecex|klassifikation)", re.IGNORECASE)
_CERT_FRAGE = re.compile(r"atex|iecex|\bex\b|explosion|zulassung|zertifi|\bul\b|\bcsa\b|cul|approbation|"
                         r"kennzeichnung|ukca|\bccc\b|\beac\b|dnv|60079", re.IGNORECASE)


def _zertifikate_abwerten(query: str, chunks: list[dict]) -> list[dict]:
    if _CERT_FRAGE.search(query or ""):
        return chunks
    out = []
    for c in chunks:
        t = str(c.get("text") or "")
        s = c.get("rerank_score")
        if isinstance(s, (int, float)) and (_CERT_ABSCHNITT.search(t[:300]) or _CERT_TEXT.search(t)):
            c = dict(c, rerank_score_roh=s, rerank_score=round(float(s) * 0.5, 4), zertifikat_abgewertet=True)
        out.append(c)
    return sorted(out, key=lambda c: c.get("rerank_score") if isinstance(c.get("rerank_score"), (int, float)) else -1,
                  reverse=True)


def _rerank_je_produkt(query: str, name: str, chunks: list[dict], k: int) -> list[dict]:
    """v3.4-patch V10: Cross-Encoder-Reranking je Produkt. Degradiert still
    auf die Abschnittsreihenfolge, wenn das Modell fehlt."""
    if len(chunks) <= 1:
        return chunks[:k]
    try:
        from backend.petra_hybrid.reranker import rerank
        res = rerank(_rerank_frage(query, name), chunks, input_top_n=len(chunks),
                     output_top_n=len(chunks))  # v3.6b (V11); v3.7: erst abwerten, dann kuerzen
        if res.reranked and res.chunks:
            return _zertifikate_abwerten(query, list(res.chunks))[:k]  # v3.7 (R4)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Rerank je Produkt fehlgeschlagen (%s): %s", name, exc)
    return chunks[:k]


def compare_retrieval(query: str, products: list[str], k_per_product: int = 5) -> dict[str, list[dict]]:
    lex = _load_lexicon()
    by_name = lex.get("by_name", {})
    namen: list[tuple[str, list[str]]] = []
    for orig, key, exakt in names_in_query(query, by_name):
        if exakt:
            namen.append((by_name[key][0].get("produkt") or orig, [e["artikelnummer"] for e in by_name[key]]))
    for p in products or []:
        key = norm_name(p)
        if key in by_name and all(key != norm_name(n) for n, _ in namen):
            namen.append((by_name[key][0].get("produkt") or p, [e["artikelnummer"] for e in by_name[key]]))
    for a in _ARTNR_RE.findall(query or ""):
        e = lex.get("by_artnr", {}).get(a)
        if e and all(a not in arts for _, arts in namen):
            namen.append((e.get("produkt") or a, [a]))
    if len(namen) < 2:
        return {}
    try:
        emb = _embed_query(query)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Vergleich: Query-Embedding fehlgeschlagen: %s", exc)
        return {}
    out: dict[str, list[dict]] = {}
    for name, arts in namen[:2]:                       # UC-02 A4: max. zwei Produkte
        emb_p = _emb_vergleich(query, name, emb)       # v3.6b (V11)
        pool = _query_collection(COLL_TEXT, emb_p, arts, k_per_product * 3) + \
            _query_collection(COLL_TAG, emb_p, arts, k_per_product)
        # v3.4-patch V10: erst Abschnittsgewichtung (doppelte Menge), dann
        # Cross-Encoder je Produkt (Kap. 4.4.1 + 4.3 [3c]).
        out[name] = _rerank_je_produkt(query, name, select_by_section(pool, query, k_per_product * 2), k_per_product)
    logger.info("Vergleich über Lexikon: %s", {n: len(c) for n, c in out.items()})
    return out
