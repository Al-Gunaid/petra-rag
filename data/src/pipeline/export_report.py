#!/usr/bin/env python3
# ============================================================
# export_report.py — Node 7: Importprotokoll als CSV (Audit-Trail)
#
# Eingabe: JSON von chromadb_status.py über stdin, ersatzweise sys.argv[1]
# Ausgabe : JSON über stdout; CSV-Datei → ~/petra-rag/data/processed/
#
# CSV-Format für deutsches Excel: Trennzeichen ";", Kodierung utf-8-sig
# (BOM für korrekte Umlaute). Nur Standardbibliothek.
# ============================================================

from __future__ import annotations

import csv
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# ── Logging auf stderr ────────────────────────────────────────
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] export_report – %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Ausgabepfad aus ENV oder Default ─────────────────────────
OUTPUT_DIR: Path = Path(
    os.environ.get("REPORT_DIR", os.path.expanduser("~/petra-rag/data/processed"))
)

# Trennzeichen für deutsches Excel (NF-12 Installierbarkeit)
CSV_DELIMITER: str = ";"


def write_csv(data: dict[str, Any], output_path: Path) -> None:
    """
    Schreibt Zusammenfassung und Tabelle je Datei als CSV.

    Args:
        data: Ausgabe von chromadb_status.py (gesamt_chunks,
            gesamt_dateien, dateien).
        output_path: Vollständiger Ausgabepfad
    """
    dateien:        list[dict[str, Any]] = data.get("dateien", [])
    gesamt_chunks:  int = data.get("gesamt_chunks", 0)
    gesamt_dateien: int = data.get("gesamt_dateien", 0)
    ok_dateien:     int = sum(1 for d in dateien if "✅" in d.get("status", ""))
    fehler_dateien: int = gesamt_dateien - ok_dateien

    with output_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh, delimiter=CSV_DELIMITER)

        # ── Titel ────────────────────────────────────────────
        writer.writerow(["PETRA-RAG — Import-Protokoll"])
        writer.writerow([
            f"Erstellt am: {datetime.now().strftime('%d.%m.%Y um %H:%M:%S Uhr')}"
        ])
        writer.writerow([])

        # ── Zusammenfassung ──────────────────────────────────
        writer.writerow(["ZUSAMMENFASSUNG"])
        writer.writerow(["Dateien gesamt",  gesamt_dateien])
        writer.writerow(["Erfolgreich ✅",   ok_dateien])
        writer.writerow(["Fehlerhaft ❌",    fehler_dateien])
        writer.writerow(["Chunks gesamt",   gesamt_chunks])
        writer.writerow([])

        # ── Tabellen-Kopfzeile ───────────────────────────────
        writer.writerow([
            "Nr.",
            "Dateiname",
            "Status",
            "Chunks gesamt",
            "Text-Chunks",
            "Tabellen-Chunks",
            "OCR-Chunks",
            "Indexierte Seiten",
        ])

        # ── Datenzeilen ──────────────────────────────────────
        for i, d in enumerate(dateien, start=1):
            writer.writerow([
                i,
                d.get("datei",         ""),
                d.get("status",        ""),
                d.get("chunks_gesamt", 0),
                d.get("text_chunks",   0),
                d.get("tabellen",      0),
                d.get("ocr_chunks",    0),
                d.get("seiten",        0),
            ])

    log.info("CSV geschrieben: %s", output_path)


def main() -> None:
    """Liest die Statusdaten, schreibt die CSV und gibt Pfad und Kennzahlen als JSON aus."""
    raw = sys.stdin.read().strip()

    if not raw and len(sys.argv) >= 2:
        raw = sys.argv[1]

    if not raw:
        log.error("Keine JSON-Daten erhalten.")
        sys.exit(1)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        log.error("Ungültiger JSON-Input: %s", exc)
        sys.exit(1)

    # Ausgabeverzeichnis erstellen
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Zeitstempel im Dateinamen: ältere Protokolle bleiben erhalten.
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"import_protokoll_{timestamp}.csv"

    try:
        write_csv(data, output_path)
    except OSError as exc:
        log.error("CSV-Schreibfehler: %s", exc)
        sys.exit(1)

    print(json.dumps({
        "status":      "✅ CSV exportiert",
        "pfad":        str(output_path),
        "dateien":     data.get("gesamt_dateien", 0),
        "chunks":      data.get("gesamt_chunks",  0),
        "zeitstempel": timestamp,
    }, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log.critical("Unerwarteter Fehler: %s", exc, exc_info=True)
        sys.exit(1)
