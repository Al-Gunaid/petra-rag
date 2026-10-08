# ============================================================
# petra_hybrid/bm25_index.py — BM25 Sparse Retrieval
#
# n8n bietet keinen BM25-Node. Der Index wird über GET /retrieval/bm25
# (query_router_v3.py) von einem n8n-"HTTP Request"-Node abgefragt.
#
# Der Index wird bei der Ingestion aus denselben Chunks wie ChromaDB
# aufgebaut (gleiche chunk_ids, siehe chunk_id.py), damit RRF beide
# Pfade zusammenführen kann.
# ============================================================
from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import math
from collections import Counter

logger = logging.getLogger("petra.bm25")

_TOKEN_RE = re.compile(r"[A-Za-zÄÖÜäöüß0-9]+")


def tokenize(text: str) -> list[str]:
    """Sprachneutrale Tokenisierung ohne Stemming; Produktcodes bleiben exakt erhalten."""
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


class BM25Okapi:
    """Abhängigkeitsfreie BM25-Okapi-Implementierung.

    Ein invertierter Index (Term -> {Dokument: Häufigkeit}) beschränkt
    get_scores() auf Dokumente, die den Term enthalten. Ergebnis ist
    identisch zur vollständigen Iteration über den Korpus.
    """
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_len) / len(corpus) if corpus else 0.0
        self.doc_freqs: list[Counter] = [Counter(doc) for doc in corpus]

        # Invertierter Index: term -> {doc_index: freq}
        self.postings: dict[str, dict[int, int]] = {}
        for i, freqs in enumerate(self.doc_freqs):
            for term, f in freqs.items():
                self.postings.setdefault(term, {})[i] = f

        self.idf: dict[str, float] = {}
        self._compute_idf()

    def _compute_idf(self) -> None:
        n_docs = len(self.corpus)
        for term, postings in self.postings.items():
            freq = len(postings)
            self.idf[term] = math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1)

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        """BM25-Score je Dokument für die Anfrage-Tokens."""
        scores = [0.0] * len(self.corpus)
        for term in query_tokens:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, f in self.postings.get(term, {}).items():
                dl = self.doc_len[i] or 1
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        return scores


@dataclass
class BM25Chunk:
    """Textabschnitt im BM25-Index mit stabiler chunk_id."""
    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)


class BM25Index:
    """Thread-sicherer In-Memory-BM25-Index mit JSONL-Persistenz.

    Format je Zeile: {"chunk_id", "text", "metadata"}. Beim Start wird der
    Index aus der Datei neu aufgebaut. Für > 100k Chunks (NF-09) wäre ein
    festplattenbasierter Index erforderlich.
    """

    def __init__(self, persist_path: str | Path) -> None:
        self._persist_path = Path(persist_path)
        self._lock = threading.RLock()
        self._chunks: list[BM25Chunk] = []
        self._bm25: BM25Okapi | None = None
        self._chunk_id_to_pos: dict[str, int] = {}
        if self._persist_path.exists():
            self._load()

    # -- persistence -------------------------------------------------
    def _load(self) -> None:
        with self._persist_path.open("r", encoding="utf-8") as fh:
            chunks = [BM25Chunk(**json.loads(line)) for line in fh if line.strip()]
        self._rebuild(chunks)
        logger.info("BM25-Index geladen: %d Chunks aus %s", len(chunks), self._persist_path)

    def _persist(self) -> None:
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        with self._persist_path.open("w", encoding="utf-8") as fh:
            for c in self._chunks:
                fh.write(json.dumps({"chunk_id": c.chunk_id, "text": c.text, "metadata": c.metadata}, ensure_ascii=False) + "\n")

    # -- build ---------------------------------------------------------
    def _rebuild(self, chunks: list[BM25Chunk]) -> None:
        self._chunks = chunks
        self._chunk_id_to_pos = {c.chunk_id: i for i, c in enumerate(chunks)}
        tokenized = [tokenize(c.text) for c in chunks] or [[]]
        self._bm25 = BM25Okapi(tokenized)

    def add_or_replace(self, chunks: Iterable[BM25Chunk]) -> int:
        """Upsert über chunk_id (analog zu ChromaDB, UC-10); baut den Index neu auf und persistiert.

        Returns:
            Anzahl neu hinzugefügter Chunks.
        """
        with self._lock:
            existing = {c.chunk_id: c for c in self._chunks}
            n_new = 0
            for c in chunks:
                if c.chunk_id not in existing:
                    n_new += 1
                existing[c.chunk_id] = c
            self._rebuild(list(existing.values()))
            self._persist()
            return n_new

    def is_empty(self) -> bool:
        """True bei leerem Index (im Betrieb ein Ingestion-Fehler)."""
        return len(self._chunks) == 0

    # -- query -----------------------------------------------------------
    def search(self, query: str, top_k: int = 10, manufacturer_filter: str | None = None) -> list[dict]:
        """BM25-Suche.

        Returns:
            Bis zu `top_k` Treffer [{chunk_id, score, rank, text, metadata}],
            absteigend nach Score; Rang 1 = bester Treffer. Treffer mit
            Score 0 (keine Termüberlappung) werden ausgelassen.
        """
        with self._lock:
            if self._bm25 is None or not self._chunks:
                return []
            scores = self._bm25.get_scores(tokenize(query))
            ranked = sorted(
                range(len(self._chunks)), key=lambda i: scores[i], reverse=True
            )
            results = []
            rank = 0
            for i in ranked:
                chunk = self._chunks[i]
                if manufacturer_filter and chunk.metadata.get("manufacturer") != manufacturer_filter:
                    continue
                if scores[i] <= 0:
                    # BM25-Score 0 bedeutet keinerlei Term-Überlappung -> nicht relevant
                    continue
                rank += 1
                results.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "score": float(scores[i]),
                        "rank": rank,
                        "text": chunk.text,
                        "metadata": chunk.metadata,
                    }
                )
                if rank >= top_k:
                    break
            return results


    # ------------------------------------------------------------------
    def search_by_filename_substring(
        self,
        filename_substring: str,
        text_substring: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """
        Exakter Filter über Dateiname und optional Textinhalt (ohne Ranking).

        Für Preisanfragen (Filter Dokumenttyp "Preisliste"):
        In großen Preislisten verdrängt BM25 die passende Zeile hinter kurze
        Datenblätter mit hoher Termdichte. Da kein doc_type-Feld existiert,
        wird die Preisliste über den Dateinamen erkannt.

        Vergleich case-insensitive, linearer Scan.

        Returns:
            Bis zu `limit` Treffer; `score` und `rank` sind None.
        """
        with self._lock:
            fn_needle = filename_substring.lower()
            txt_needle = text_substring.lower() if text_substring else None
            results: list[dict] = []
            for chunk in self._chunks:
                fn = str(chunk.metadata.get("filename", "")).lower()
                if fn_needle not in fn:
                    continue
                if txt_needle and txt_needle not in chunk.text.lower():
                    continue
                results.append({
                    "chunk_id": chunk.chunk_id,
                    "score": None,
                    "rank": None,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                })
                if len(results) >= limit:
                    break
            return results


_default_index: BM25Index | None = None


def get_bm25_index(persist_path: str | Path = "data/bm25_index.jsonl") -> BM25Index:
    """
    Prozessweite BM25-Index-Instanz (Lazy Singleton).

    `persist_path` wird nur beim ersten Aufruf berücksichtigt.
    """
    global _default_index
    if _default_index is None:
        _default_index = BM25Index(persist_path)
    return _default_index