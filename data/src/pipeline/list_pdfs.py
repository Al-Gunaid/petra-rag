#!/usr/bin/env python3
# ============================================================
# list_pdfs.py — PDF-Erkennung und Duplikatprüfung
#
# Durchsucht den Watch-Folder und berechnet je PDF einen SHA-256-Hash.
# Verarbeitete Dateien werden über <HASH_FOLDER>/<hash>.done erkannt.
# PDFs über MAX_PDF_SIZE_MB erhalten needs_split=True und werden vom
# n8n-Workflow über /split-pdf geteilt.
#
# Ausgabe: JSON über stdout; Logging über stderr.
# ============================================================

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any


# stdout ist für die JSON-Antwort an n8n reserviert.
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] list_pdfs – %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

log = logging.getLogger(__name__)


# Ordner mit den eingehenden PDFs
PDF_FOLDER = Path(
    os.environ.get(
        "WATCH_FOLDER",
        os.path.expanduser("~/petra-rag/data/pdfs"),
    )
)

# Ablage der .done-Marker (save_to_chromadb.py)
HASH_FOLDER = Path(
    os.environ.get(
        "HASH_FOLDER",
        os.path.expanduser("~/petra-rag/data/processed/hashes"),
    )
)

# PDFs über dieser Grenze werden nicht verworfen,
# sondern für die automatische Teilung markiert.
MAX_FILE_SIZE_MB = int(
    os.environ.get("MAX_PDF_SIZE_MB", "50")
)

MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


def ensure_directories() -> None:
    """Legt Watch-Folder und Hash-Ordner an, falls nicht vorhanden."""

    PDF_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    HASH_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )


def compute_sha256(filepath: Path) -> str:
    """
    Berechnet den SHA-256-Hash einer Datei.

    Die Datei wird blockweise gelesen, damit auch große PDFs
    nicht vollständig in den Arbeitsspeicher geladen werden.
    """

    sha = hashlib.sha256()

    with filepath.open("rb") as file_handle:
        for block in iter(
            lambda: file_handle.read(1024 * 1024),
            b"",
        ):
            sha.update(block)

    return sha.hexdigest()


# Hash-Cache: SHA-256 nur für neue oder geänderte Dateien berechnen.
# Schlüssel = Pfad; gültig, solange Größe und Änderungszeit gleich bleiben.
HASH_CACHE_FILE = HASH_FOLDER.parent / "hash_cache.json"
_HASH_CACHE: dict[str, dict[str, Any]] = {}
_HASH_CACHE_GEAENDERT = False


def _cache_laden() -> None:
    global _HASH_CACHE
    try:
        _HASH_CACHE = json.loads(HASH_CACHE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        _HASH_CACHE = {}


def _cache_speichern() -> None:
    if not _HASH_CACHE_GEAENDERT:
        return
    try:
        tmp = HASH_CACHE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(_HASH_CACHE), encoding="utf-8")
        tmp.replace(HASH_CACHE_FILE)
    except OSError as exc:
        log.warning("Hash-Cache konnte nicht gespeichert werden: %s", exc)


def _hash_mit_cache(filepath: Path) -> str:
    global _HASH_CACHE_GEAENDERT
    st = filepath.stat()
    alt = _HASH_CACHE.get(str(filepath))
    if alt and alt.get("size") == st.st_size and alt.get("mtime_ns") == st.st_mtime_ns:
        return alt["hash"]
    file_hash = compute_sha256(filepath)
    _HASH_CACHE[str(filepath)] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "hash": file_hash}
    _HASH_CACHE_GEAENDERT = True
    return file_hash


def is_processed(file_hash: str) -> bool:
    """
    Prüft, ob für diesen Datei-Hash bereits
    ein Done-Marker existiert.
    """

    marker = HASH_FOLDER / f"{file_hash}.done"

    return marker.exists()


def scan_pdfs() -> list[dict[str, Any]]:
    """
    Liest alle PDFs aus dem Watch-Folder.

    Wichtig:
    Große PDFs werden nicht mehr als ungültig markiert.
    Sie erhalten needs_split=True.
    
    Returns:
        Je Datei: filename, path, hash, size_bytes, size_mb, needs_split,
        neu, valid, reason. Lesefehler ergeben valid=False.
    """

    results: list[dict[str, Any]] = []

    for entry in sorted(PDF_FOLDER.iterdir()):
        # Nur echte PDF-Dateien berücksichtigen
        if (
            not entry.is_file()
            or entry.suffix.lower() != ".pdf"
        ):
            continue

        log.info("Prüfe: %s", entry.name)

        size_bytes = entry.stat().st_size
        size_mb = size_bytes / 1024 / 1024

        # Große Datei nur markieren, nicht verwerfen
        needs_split = size_bytes > MAX_FILE_SIZE_BYTES

        try:
            file_hash = _hash_mit_cache(entry)   # Hash-Cache

        except OSError as exc:
            log.error(
                "Lesefehler bei '%s': %s",
                entry.name,
                exc,
            )

            results.append(
                {
                    "filename": entry.name,
                    "path": str(entry),
                    "hash": "",
                    "size_bytes": size_bytes,
                    "size_mb": round(size_mb, 2),
                    "needs_split": needs_split,
                    "neu": False,
                    "valid": False,
                    "reason": f"Lesefehler: {exc}",
                }
            )

            continue

        already_processed = is_processed(file_hash)

        if needs_split:
            reason = (
                f"PDF ist größer als {MAX_FILE_SIZE_MB} MB "
                "und wird automatisch geteilt"
            )

            log.info(
                "'%s' wird zur Teilung markiert: %.2f MB",
                entry.name,
                size_mb,
            )

        elif already_processed:
            reason = "bereits verarbeitet"

        else:
            reason = ""

        results.append(
            {
                "filename": entry.name,
                "path": str(entry),
                "hash": file_hash,
                "size_bytes": size_bytes,
                "size_mb": round(size_mb, 2),

                # Entscheidend für den neuen n8n-IF-Node
                "needs_split": needs_split,

                # Auch eine große neue PDF bleibt "neu"
                "neu": not already_processed,

                # Große PDFs sind gültig und werden nicht verworfen
                "valid": True,

                "reason": reason,
            }
        )

    return results


def main() -> None:
    """Scannt den Watch-Folder und gibt Liste und Zusammenfassung als JSON aus."""

    ensure_directories()

    _cache_laden()
    results = scan_pdfs()
    _cache_speichern()

    gesamt = len(results)

    neu = sum(
        1
        for item in results
        if item["neu"] and item["valid"]
    )

    invalid = sum(
        1
        for item in results
        if not item["valid"]
    )

    needs_split = sum(
        1
        for item in results
        if item["neu"]
        and item["valid"]
        and item["needs_split"]
    )

    log.info(
        "Ergebnis: %d total | %d neu | "
        "%d zu teilen | %d ungültig",
        gesamt,
        neu,
        needs_split,
        invalid,
    )

    # JSON-Antwort für n8n
    print(
        json.dumps(
            {
                "pdfs": results,
                "gesamt": gesamt,
                "neu": neu,
                "needs_split": needs_split,
                "invalid": invalid,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        log.critical(
            "Unerwarteter Fehler: %s",
            exc,
            exc_info=True,
        )

        sys.exit(1)