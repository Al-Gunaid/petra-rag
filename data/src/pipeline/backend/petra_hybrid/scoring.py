# ============================================================
# petra_hybrid/scoring.py — Vergleichbare Retrieval-Scores
#
# ============================================================
# petra_hybrid/scoring.py — Vergleichbare Retrieval-Scores
#
# RAG-Architektur: schicht [3a]/[3b]
#
# Der RRF-Score eignet sich nicht als Qualitätsschwelle: Bei k=60 und drei
# Quellen ist das Maximum 3 * 1/61 ≈ 0.049, eine Schwelle von 0.65 ist
# unerreichbar. Ein reiner Dense-Score als Ersatz benachteiligt Chunks,
# die nur BM25 gefunden hat (z. B. exakte Produktcodes).
#
# Daher:
#   * RRF-Score  -> nur Ranking.
#   * Confidence -> Qualitätsschwelle; Maximum der je Quelle
#     normalisierten Scores. Ein Chunk passiert, wenn eine Quelle ihn
#     mit hoher Sicherheit gefunden hat.
#
# ChromaDB liefert je nach Konfiguration eine Distanz (kleiner = besser).
# Unerkannt würde der Filter die schlechtesten Chunks durchlassen; daher
# erfolgt hier die Umrechnung in eine Ähnlichkeit.
# ============================================================
from __future__ import annotations

# Dense-Scores > 1.0 gelten im Modus "auto" als Distanz: Kosinus-Ähnlichkeit
# liegt in [-1, 1], L2-Distanzen überschreiten 1.0 regelmäßig.
_DISTANCE_HEURISTIC_THRESHOLD = 1.0


def dense_score_to_similarity(value: float, space: str = "auto") -> float:
    """Normalisiert einen Dense-Score auf eine Ähnlichkeit in [0,1].

    Args:
        value: Rohwert aus ChromaDB.
        space: ``"cosine_similarity"``, ``"cosine_distance"``, ``"l2"``
            oder ``"auto"``. ``auto`` nutzt eine konservative Heuristik.

    Die Heuristik ist ausdrücklich nur eine Absicherung. Die
    Konfiguration sollte über ``PETRA_CHROMA_SCORE_SPACE`` explizit
    gesetzt werden, da eine falsche Annahme im Betrieb nicht sichtbar ist.
    
    """
    if value is None:
        return 0.0
    v = float(value)

    if space == "cosine_similarity":
        return _clamp(v)
    if space == "cosine_distance":
        return _clamp(1.0 - v)
    if space == "l2":
        return _clamp(1.0 / (1.0 + max(v, 0.0)))

    # auto
    if v < 0.0:
        return _clamp((v + 1.0) / 2.0)  # Kosinus in [-1,1]
    if v > _DISTANCE_HEURISTIC_THRESHOLD:
        return _clamp(1.0 / (1.0 + v))  # sehr wahrscheinlich eine Distanz
    return _clamp(v)


def normalize_bm25_scores(hits: list[dict]) -> dict[str, float]:
    """Min-Max-Normalisierung der BM25-Scores innerhalb einer Trefferliste.

    BM25-Scores sind zwischen Anfragen nicht vergleichbar (abhängig von IDF
    und Anfragelänge). Die Normalisierung bewertet, wie deutlich sich ein
    Treffer von den übrigen Kandidaten abhebt.

    Das Ergebnis wird mit ``min(1.0, top / 6.0)`` gedeckelt, damit eine
    Liste schwacher Zufallstreffer ihren besten Treffer nicht auf 1.0 hebt.

    Returns:
        Mapping chunk_id -> normalisierter Score in [0, 1].
    """
    if not hits:
        return {}
    raw = [float(h.get("score") or 0.0) for h in hits]
    top = max(raw)
    if top <= 0.0:
        return {h.get("chunk_id"): 0.0 for h in hits}

    # Dämpfung bei schwacher lexikalischer Übereinstimmung: Erst ab einem
    # Rohscore von 6.0 ist volle Confidence möglich.

    confidence_cap = min(1.0, top / 6.0)

    lo = min(raw)
    span = (top - lo) or 1.0
    out: dict[str, float] = {}
    for hit, value in zip(hits, raw):
        cid = hit.get("chunk_id")
        if not cid:
            continue
        relative = (value - lo) / span if len(raw) > 1 else 1.0
        out[cid] = _clamp(relative * confidence_cap)
    return out


def compute_confidence(
    source_scores: dict,
    bm25_normalized: float | None = None,
    dense_space: str = "auto",
) -> float:
    """Confidence eines fusionierten Chunks in [0, 1].

    Maximum über alle Quellen, die den Chunk gefunden haben. TAG-Scores
    werden wie Dense-Scores umgerechnet (gleiche Metrik in ChromaDB).
    """
    candidates: list[float] = []

    dense = source_scores.get("dense")
    if isinstance(dense, (int, float)):
        candidates.append(dense_score_to_similarity(dense, dense_space))

    tag = source_scores.get("tag")
    if isinstance(tag, (int, float)):
        candidates.append(dense_score_to_similarity(tag, dense_space))

    if isinstance(bm25_normalized, (int, float)):
        candidates.append(_clamp(float(bm25_normalized)))

    return max(candidates) if candidates else 0.0


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
