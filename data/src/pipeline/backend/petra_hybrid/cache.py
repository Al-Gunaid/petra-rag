# ============================================================
# petra_hybrid/cache.py — Cache-Augmented Generation (CAG)
#
# RAG-Archetiktur [1c] beschreibt einen KV-Cache (vorberechnete Attention-
# Einträge). Ollama stellt den Attention-State über die REST-API nicht
# bereit; ein KV-Cache wäre nur mit direkter llama.cpp-Anbindung möglich.
# Implementiert ist daher ein Antwort- und Kontext-Cache:
#   * kind="response": Exact-Match über normalisierte Anfrage und
#     Korpusversion; ein Treffer überspringt die gesamte Pipeline.
#   * Retrieval-Cache: spart Embedding, ChromaDB- und BM25-Abfrage.
#
# Invalidierung: Der Schlüssel enthält die `corpus_version`. Nach einem
# Import setzt die Ingestion eine neue Version; alle Einträge werden
# verworfen.
# ============================================================
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("petra.cache")

_WS_RE = re.compile(r"\s+")
DEFAULT_TTL_SECONDS = 24 * 3600
DEFAULT_MAX_ENTRIES = 500


def normalize_query(query: str) -> str:
    """Normalisiert die Anfrage für den Cache-Key.

    Bewusst konservativ: Nur Groß-/Kleinschreibung, Whitespace und
    Satzendzeichen. Kein Stemming, keine Synonymauflösung — bei
    technischen Anfragen ist der Unterschied zwischen "O1D302" und
    "O1D303" ein Zeichen, und ein zu aggressiv normalisierender Cache
    liefert falsche Produktdaten. Ein Cache-Miss kostet Latenz, ein
    falscher Cache-Hit kostet Vertrauen.
    """
    text = (query or "").strip().lower()
    text = _WS_RE.sub(" ", text)
    return text.rstrip("?!. ")


@dataclass
class CacheEntry:
    """Ein Cache-Eintrag samt Korpus-Version fuer die Invalidierung."""
    key: str
    payload: dict
    created_at: float
    corpus_version: str
    hits: int = 0
    kind: str = "response"

    def is_expired(self, ttl: float) -> bool:
        """Ob der Eintrag die konfigurierte TTL ueberschritten hat."""
        return (time.time() - self.created_at) > ttl


@dataclass
class CacheLookup:
    """Ergebnis einer Cache-Abfrage (Treffer, Alter, Begruendung)."""
    hit: bool
    payload: dict | None = None
    kind: str = "response"
    age_seconds: int = 0
    reason: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """JSON-serialisierbare Darstellung fuer n8n und FastAPI."""
        return {
            "cache_hit": self.hit,
            "cache_kind": self.kind,
            "cache_age_seconds": self.age_seconds,
            "cache_reason": self.reason,
            "payload": self.payload,
        }


class CagCache:
    """Thread-sicherer In-Memory-Cache mit optionaler JSON-Persistenz."""

    def __init__(
        self,
        persist_path: str | Path | None = None,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ) -> None:
        self._lock = threading.RLock()
        self._entries: dict[str, CacheEntry] = {}
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._persist_path = Path(persist_path) if persist_path else None
        self._corpus_version = "unversioned"
        if self._persist_path and self._persist_path.exists():
            self._load()

    # -- Korpus-Version / Invalidierung ------------------------------
    def set_corpus_version(self, version: str) -> int:
        """Setzt die Korpus-Version. Weicht sie von der bisherigen ab,
        werden alle Einträge verworfen. Wird von der Ingestion nach jedem
        Dokument-Import aufgerufen (siehe ingestion_hook.py)."""
        with self._lock:
            if version == self._corpus_version:
                return 0
            dropped = len(self._entries)
            self._entries.clear()
            self._corpus_version = version
            self._persist()
            logger.info(
                "Cache invalidiert: Korpus-Version %s -> %s (%d Einträge verworfen)",
                self._corpus_version, version, dropped,
            )
            return dropped

    @property
    def corpus_version(self) -> str:
        """Aktuelle Korpus-Version; steuert die Cache-Invalidierung."""
        return self._corpus_version

    # -- Lookup / Store ----------------------------------------------
    def _key(self, query: str, kind: str, extra: str = "") -> str:
        raw = f"{kind}|{self._corpus_version}|{normalize_query(query)}|{extra}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    def lookup(self, query: str, kind: str = "response", extra: str = "") -> CacheLookup:
        """Sucht einen gueltigen Eintrag zur normalisierten Anfrage."""
        with self._lock:
            key = self._key(query, kind, extra)
            entry = self._entries.get(key)
            if entry is None:
                return CacheLookup(hit=False, kind=kind, reason="Kein Eintrag.")
            if entry.corpus_version != self._corpus_version:
                self._entries.pop(key, None)
                return CacheLookup(hit=False, kind=kind, reason="Korpus-Version geändert.")
            if entry.is_expired(self._ttl):
                self._entries.pop(key, None)
                return CacheLookup(hit=False, kind=kind, reason="TTL abgelaufen.")
            entry.hits += 1
            return CacheLookup(
                hit=True,
                payload=entry.payload,
                kind=kind,
                age_seconds=int(time.time() - entry.created_at),
                reason=f"Treffer (bisher {entry.hits} Zugriffe).",
            )

    def store(self, query: str, payload: dict, kind: str = "response", extra: str = "") -> str:
        """Legt eine Antwort unter der normalisierten Anfrage ab."""
        with self._lock:
            key = self._key(query, kind, extra)
            self._entries[key] = CacheEntry(
                key=key,
                payload=payload,
                created_at=time.time(),
                corpus_version=self._corpus_version,
                kind=kind,
            )
            self._evict_if_needed()
            self._persist()
            return key

    def _evict_if_needed(self) -> None:
        if len(self._entries) <= self._max_entries:
            return
        # LRU-Näherung: ältester Eintrag mit den wenigsten Treffern zuerst.
        victims = sorted(self._entries.values(), key=lambda e: (e.hits, e.created_at))
        for entry in victims[: len(self._entries) - self._max_entries]:
            self._entries.pop(entry.key, None)

    def stats(self) -> dict:
        """Kennzahlen fuer den /cache/stats-Endpunkt."""
        with self._lock:
            return {
                "entries": len(self._entries),
                "corpus_version": self._corpus_version,
                "ttl_seconds": self._ttl,
                "max_entries": self._max_entries,
                "total_hits": sum(e.hits for e in self._entries.values()),
            }

    def clear(self) -> int:
        """Verwirft alle Eintraege und gibt deren Anzahl zurueck."""
        with self._lock:
            dropped = len(self._entries)
            self._entries.clear()
            self._persist()
            return dropped

    # -- Persistenz ---------------------------------------------------
    def _persist(self) -> None:
        if not self._persist_path:
            return
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "corpus_version": self._corpus_version,
                "entries": [
                    {
                        "key": e.key, "payload": e.payload, "created_at": e.created_at,
                        "corpus_version": e.corpus_version, "hits": e.hits, "kind": e.kind,
                    }
                    for e in self._entries.values()
                ],
            }
            self._persist_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            logger.warning("Cache-Persistenz fehlgeschlagen: %s", exc)

    def _load(self) -> None:
        try:
            data = json.loads(self._persist_path.read_text(encoding="utf-8"))
            self._corpus_version = data.get("corpus_version", "unversioned")
            for raw in data.get("entries", []):
                self._entries[raw["key"]] = CacheEntry(**raw)
            logger.info("Cache geladen: %d Einträge", len(self._entries))
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("Cache konnte nicht geladen werden: %s", exc)


_default_cache: CagCache | None = None


def get_cache() -> CagCache:
    """Liefert die prozessweite CAG-Cache-Instanz (Lazy Singleton)."""
    global _default_cache
    if _default_cache is None:
        _default_cache = CagCache(
            persist_path=os.getenv("PETRA_CACHE_PATH", "data/cag_cache.json"),
            ttl_seconds=int(os.getenv("PETRA_CACHE_TTL_SECONDS", DEFAULT_TTL_SECONDS)),
            max_entries=int(os.getenv("PETRA_CACHE_MAX_ENTRIES", DEFAULT_MAX_ENTRIES)),
        )
    return _default_cache


def is_enabled() -> bool:
    """Ob der CAG-Cache aktiviert ist (PETRA_CACHE_ENABLED)."""
    return os.getenv("PETRA_CACHE_ENABLED", "true").lower() not in ("false", "0", "no")
