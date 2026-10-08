# ============================================================
# petra_hybrid/telemetry.py — Anfrage-Logging (Schicht 5c)
#
# RAG-Architektur: (Faithfulness ≥ 0.78, Context Precision ≥ 0.74, E2E ≤ 5 s).
#
# Datengrundlage für den Nachweis der Qualitätsziele und die
# RAGAs-Auswertung ([5c]).
#
# Format: JSONL, eine Zeile je Anfrage. Wird nur sequenziell gelesen und
# benötigt keinen zusätzlichen Dienst (Offline-First, P3).
#
# Datenschutz (P1/NF-01): keine Nutzeridentitäten. Mit
# PETRA_TELEMETRY_REDACT_QUERY=true wird der Anfragetext durch einen
# SHA-256-Präfix ersetzt.
# ============================================================
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("petra.telemetry")

_lock = threading.Lock()    # serialisiert Schreibzugriffe auf die JSONL-Datei


@dataclass
class TelemetryRecord:
    """Ein Protokolleintrag einer Anfrage"""

    request_id: str
    timestamp: str = ""
    query: str = ""
    query_class: str = ""
    q_class: str = ""
    # Tatsächlich ausgeführte Retrieval-Pfade; nötig zur Interpretation der Metriken.
    activated_paths: list[str] = field(default_factory=list)
    retrieval_round: int = 1
    retrieval_method: str = ""
    chunk_ids: list[str] = field(default_factory=list)
    top_score: float | None = None
    crag_status: str = ""
    id_consistency: float | None = None
    reranked: bool = False
    cache_hit: bool = False
    faithfulness_score: float | None = None
    verified: bool | None = None
    latency_ms: int = 0
    stage_latencies_ms: dict = field(default_factory=dict)
    answer_length: int = 0
    error: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """JSON-Darstellung für n8n und FastAPI."""
        return asdict(self)


def _redact(query: str) -> str:
    """Ersetzt die Anfrage durch "sha256:<16 Zeichen>", falls PETRA_TELEMETRY_REDACT_QUERY aktiv ist."""
    if os.getenv("PETRA_TELEMETRY_REDACT_QUERY", "false").lower() in ("true", "1", "yes"):
        return "sha256:" + hashlib.sha256((query or "").encode("utf-8")).hexdigest()[:16]
    return query


def is_enabled() -> bool:
    """Ob die Telemetrie aktiviert ist (PETRA_TELEMETRY_ENABLED)."""
    return os.getenv("PETRA_TELEMETRY_ENABLED", "true").lower() not in ("false", "0", "no")


def log_path() -> Path:
    """Pfad der JSONL-Protokolldatei (PETRA_TELEMETRY_PATH)."""
    return Path(os.getenv("PETRA_TELEMETRY_PATH", "data/telemetry.jsonl"))


def log_request(record: TelemetryRecord | dict) -> bool:
    """Hängt einen Eintrag an die JSONL-Datei an.

    Wirft keine Exception; Schreibfehler werden protokolliert. Der
    n8n-Node ruft den Endpoint mit onError=continue auf.

    Returns:
        True, wenn geschrieben wurde.
    """
    if not is_enabled():
        return False

    data = record.to_dict() if isinstance(record, TelemetryRecord) else dict(record)
    data.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    data["query"] = _redact(data.get("query", ""))

    try:
        path = log_path()
        with _lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(data, ensure_ascii=False) + "\n")
        return True
    except (OSError, TypeError, ValueError) as exc:
        logger.warning("Telemetrie-Schreibfehler (nicht blockierend): %s", exc)
        return False


def read_records(limit: int | None = None) -> list[dict]:
    """Liest Einträge (bei `limit` die letzten n); ungültige Zeilen werden übersprungen."""
    path = log_path()
    if not path.exists():
        return []
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except ValueError:
                continue
    return records[-limit:] if limit else records


def summarize(records: list[dict] | None = None) -> dict:
    """Aggregat für den ``/telemetry/summary``-Endpunkt (Abnahmekennzahlen).

    Enthält Latenz-Perzentile und NF-03-Quote (≤ 5 s), Faithfulness-Mittel
    und -Zielquote (≥ 0.78), CRAG-Statusverteilung, Cache- und Rerank-Quote.
    """
    records = records if records is not None else read_records()
    if not records:
        return {"count": 0, "note": "Keine Telemetriedaten vorhanden."}

    latencies = sorted(r.get("latency_ms", 0) for r in records if r.get("latency_ms"))
    faith = [r["faithfulness_score"] for r in records if isinstance(r.get("faithfulness_score"), (int, float))]

    def pct(values: list[int], p: float) -> int:
        """Perzentil per Nearest-Rank auf einer sortierten Liste."""
        if not values:
            return 0
        idx = min(len(values) - 1, int(round((len(values) - 1) * p)))
        return values[idx]

    status_counts: dict[str, int] = {}
    for record in records:
        status_counts[record.get("crag_status", "unbekannt")] = (
            status_counts.get(record.get("crag_status", "unbekannt"), 0) + 1
        )

    return {
        "count": len(records),
        "latency_ms": {
            "p50": pct(latencies, 0.5),
            "p95": pct(latencies, 0.95),
            "max": latencies[-1] if latencies else 0,
            "nf03_erfuellt_anteil": round(
                sum(1 for value in latencies if value <= 5000) / len(latencies), 3
            ) if latencies else None,
        },
        "faithfulness": {
            "mean": round(sum(faith) / len(faith), 3) if faith else None,
            "ziel_0_78_erreicht_anteil": round(
                sum(1 for value in faith if value >= 0.78) / len(faith), 3
            ) if faith else None,
        },
        "crag_status": status_counts,
        "cache_hit_rate": round(
            sum(1 for r in records if r.get("cache_hit")) / len(records), 3
        ),
        "rerank_rate": round(
            sum(1 for r in records if r.get("reranked")) / len(records), 3
        ),
    }
