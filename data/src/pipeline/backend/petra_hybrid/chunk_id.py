# ============================================================
# petra_hybrid/chunk_id.py — Deterministische Chunk-IDs
#
# Architektur: Kap. 4.2 [3a] (Reciprocal Rank Fusion).
#
# RRF dedupliziert über `chunk_id`. Vergeben BM25-Index und ChromaDB
# unterschiedliche IDs für denselben Abschnitt, entsteht kein kombinierter
# Score; die Fusion wird zu einer reinen Aneinanderreihung.
#
# Alle Indexierungspfade (ChromaDB `ids`, ChromaDB `metadata.chunk_id`,
# BM25-JSONL) verwenden daher make_chunk_id(). Die ID hängt nur von
# Dokument, Seite, Position und Typ ab und bleibt über Re-Ingestions
# stabil (Voraussetzung für Cache-Invalidierung und reproduzierbare
# Evaluationen).
# ============================================================
from __future__ import annotations

import hashlib
import re

_NORM_RE = re.compile(r"\s+")

CHUNK_ID_PREFIX_TEXT = "txt"
CHUNK_ID_PREFIX_TABLE = "tbl"


def _normalize_document_name(document: str) -> str:
    """Vereinheitlicht Dokumentnamen, damit Pfadvarianten desselben
    Dokuments (`./a/b.pdf` vs `b.pdf`) dieselbe ID erzeugen."""
    name = (document or "unbekannt").strip().replace("\\", "/")
    name = name.rsplit("/", 1)[-1]
    return _NORM_RE.sub(" ", name).lower()


def make_chunk_id(
    document: str,
    page: int | None,
    chunk_index: int,
    source_type: str = "text",
) -> str:
    """Erzeugt die kanonische Chunk-ID.

    Format: ``<prefix>-<12-stelliger sha256-Präfix>``, z.B. ``txt-9f2c1ab30de4``.

    Der Prefix ist bewusst NICHT quellenabhängig (nicht "dense"/"bm25"),
    sondern inhaltstypabhängig — sonst wäre die Dedup über Retrieval-Pfade
    hinweg per Konstruktion unmöglich (genau der Defekt aus P-03).
    """
    prefix = CHUNK_ID_PREFIX_TABLE if source_type == "table" else CHUNK_ID_PREFIX_TEXT
    raw = f"{_normalize_document_name(document)}|{page if page is not None else '-'}|{chunk_index}|{source_type}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"


def make_content_fallback_id(text: str, source_type: str = "text") -> str:
    """
    Ersatz-ID aus dem normalisierten Textinhalt für Chunks ohne Metadaten.

    Gleicher Text aus BM25 und Dense ergibt dieselbe ID; eine
    positionsbasierte ID (`dense-0`) würde nie übereinstimmen.
    """
    normalized = _NORM_RE.sub(" ", (text or "").strip().lower())
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]
    prefix = CHUNK_ID_PREFIX_TABLE if source_type == "table" else CHUNK_ID_PREFIX_TEXT
    return f"{prefix}-{digest}"


def id_consistency_ratio(list_a: list[dict], list_b: list[dict]) -> float:
    """Anteil der IDs der kleineren Liste, die auch in der anderen vorkommen.

    Dauerhaft 0.0 über viele Anfragen bedeutet, dass die Indizes nicht
    ID-kompatibel sind.
    """
    ids_a = {c.get("chunk_id") for c in list_a if c.get("chunk_id")}
    ids_b = {c.get("chunk_id") for c in list_b if c.get("chunk_id")}
    if not ids_a or not ids_b:
        return 0.0
    smaller = min(len(ids_a), len(ids_b))
    return len(ids_a & ids_b) / smaller
