"""
services/file_cache.py — Dateibasierter Cache, der Neustarts überdauert.

Speichert API-Antworten als JSON im Verzeichnis `.cache` mit
zeitstempelbasierter TTL (Standard 600 s).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

CACHE_DIR = Path(__file__).parent.parent / ".cache"


def _get_cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Nicht alphanumerische Zeichen durch "_" ersetzen (gültiger Dateiname).
    safe_key = "".join(c if c.isalnum() else "_" for c in key)
    return CACHE_DIR / f"{safe_key}.json"


def get_file_cache(key: str, ttl: float = 600.0) -> Optional[Any]:
    """Liefert den Cache-Eintrag oder None, wenn er fehlt, abgelaufen oder unlesbar ist."""
    path = _get_cache_path(key)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            wrapper = json.load(f)
        timestamp = wrapper.get("timestamp", 0)
        if time.time() - timestamp > ttl:
            return None
        return wrapper.get("data")
    except Exception:
        return None


def set_file_cache(key: str, data: Any) -> None:
    """Schreibt `data` mit aktuellem Zeitstempel; Schreibfehler werden ignoriert."""
    path = _get_cache_path(key)
    try:
        wrapper = {
            "timestamp": time.time(),
            "data": data,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(wrapper, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def clear_file_cache() -> None:
    """Löscht alle JSON-Dateien im Cache-Verzeichnis."""
    try:
        if CACHE_DIR.exists():
            for p in CACHE_DIR.glob("*.json"):
                p.unlink()
    except Exception:
        pass
