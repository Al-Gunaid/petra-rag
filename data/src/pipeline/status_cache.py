#!/usr/bin/env python3
# ============================================================
# status_cache.py — Dokumentstatistik (GET /status) im Hintergrund vorhalten
#
# Problem: chromadb_status.py liest bei jedem Aufruf ALLE Chunk-Metadaten
# (≈ 720 000) in 500er-Blöcken über HTTP. Das dauert mehrere Minuten; das
# Dashboard bricht nach 60 s ab, der CSV-Export nach 120 s.
#
# Lösung: Das Ergebnis wird einmal berechnet und als JSON gespeichert
# (REPORT_DIR/status_cache.json). /status und /export-report antworten sofort
# aus dieser Datei; eine Neuberechnung läuft im Hintergrund, wenn die Daten
# älter als STATUS_CACHE_MAX_AGE sind oder seit der letzten Berechnung
# importiert wurde. Jede Antwort trägt den Stand ("stand").
# ============================================================
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("status_cache")

REPORT_DIR = Path(os.environ.get("REPORT_DIR", "/data/processed"))
CACHE = REPORT_DIR / "status_cache.json"
MAX_ALTER_S = int(os.environ.get("STATUS_CACHE_MAX_AGE", str(6 * 3600)))

_start_lock = threading.Lock()
_laeuft = threading.Event()
_veraltet = False


def lesen() -> dict[str, Any] | None:
    try:
        return json.loads(CACHE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def ist_veraltet(daten: dict[str, Any] | None = None) -> bool:
    daten = daten if daten is not None else lesen()
    if daten is None:
        return True
    return _veraltet or (time.time() - float(daten.get("stand_ts", 0))) > MAX_ALTER_S


def als_veraltet_markieren() -> None:
    """Nach einem Import aufrufen: Die nächste /status-Anfrage stößt eine Neuberechnung an."""
    global _veraltet
    _veraltet = True


def laeuft() -> bool:
    return _laeuft.is_set()


def _berechnen() -> None:
    global _veraltet
    try:
        _veraltet = False
        # Größere Blöcke: 5000 statt 500 Metadaten je HTTP-Abruf (≈ 145 statt ≈ 1440 Abrufe)
        os.environ.setdefault("STATUS_BATCH_SIZE", "5000")
        from chromadb_status import get_status   # gleiches Verzeichnis wie api_server.py

        t0 = time.time()
        daten = get_status()
        daten["stand_ts"] = int(time.time())
        daten["stand"] = time.strftime("%d.%m.%Y %H:%M:%S")
        daten["dauer_s"] = round(time.time() - t0, 1)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        tmp = CACHE.with_suffix(".tmp")
        tmp.write_text(json.dumps(daten, ensure_ascii=False), encoding="utf-8")
        tmp.replace(CACHE)
        log.info("Dokumentstatistik aktualisiert: %s Dateien, %s Chunks, %.0f s",
                 daten.get("gesamt_dateien"), daten.get("gesamt_chunks"), daten["dauer_s"])
    except Exception:  # noqa: BLE001 – Hintergrundaufgabe darf die API nie stören
        log.exception("Dokumentstatistik konnte nicht berechnet werden")
    finally:
        _laeuft.clear()


def aktualisieren_im_hintergrund() -> bool:
    """Startet die Neuberechnung, falls nicht schon eine läuft."""
    with _start_lock:
        if _laeuft.is_set():
            return False
        _laeuft.set()
    threading.Thread(target=_berechnen, daemon=True, name="status-cache").start()
    return True


def warten(sekunden: float) -> dict[str, Any] | None:
    ende = time.time() + sekunden
    while _laeuft.is_set() and time.time() < ende:
        time.sleep(0.5)
    return lesen()


def beim_start() -> None:
    if lesen() is None:
        aktualisieren_im_hintergrund()
