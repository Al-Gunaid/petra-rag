"""
split_pdf.py — Teilt große PDFs im Watch-Folder in Teildateien.

Aufruf über POST /split-pdf (api_server.py). Teildateien
"<name>__part_NNN.pdf" werden im Watch-Folder abgelegt; das Original wird
nach erfolgreicher Teilung nach PDF_ARCHIVE_FOLDER verschoben.
"""
from __future__ import annotations

import math
import os
import shutil
from pathlib import Path

import fitz  # PyMuPDF


# Ordner mit den neuen PDF-Dateien
PDF_DIR = Path(
    os.environ.get("WATCH_FOLDER", "/data/pdfs")
)

# Hier wird das große Original nach erfolgreicher Teilung archiviert
ARCHIVE_DIR = Path(
    os.environ.get(
        "PDF_ARCHIVE_FOLDER",
        "/data/pdfs_original",
    )
)

# Zielgröße bewusst etwas kleiner als die erlaubten 50 MB
TARGET_SIZE_MB = int(
    os.environ.get("PDF_SPLIT_TARGET_MB", "48")
)

TARGET_SIZE_BYTES = TARGET_SIZE_MB * 1024 * 1024


def save_pages(
    source_document: fitz.Document,
    start_page: int,
    end_page: int,
    output_path: Path,
) -> int:
    """
    Speichert einen Seitenbereich als neue PDF.

    Returns:
        Dateigröße in Bytes.
    """

    part_document = fitz.open()

    try:
        part_document.insert_pdf(
            source_document,
            from_page=start_page,
            to_page=end_page,
        )

        part_document.save(
            output_path,
            garbage=4,
            deflate=True,
        )

    finally:
        part_document.close()

    return output_path.stat().st_size


def split_pdf(filename: str) -> dict:
    """
    Teilt PDFs über 48 MB automatisch in kleinere Teile.

    Ablauf:
    1. Kleine PDFs werden unverändert zurückgegeben.
    2. Für große PDFs wird eine ungefähre Seitenzahl pro Teil berechnet.
    3. Ist ein Teil noch zu groß, wird seine Seitenzahl reduziert.
    4. Nach erfolgreicher Teilung wird das Original archiviert.

    Returns:
        Dict mit status ("not_needed"|"split"), original, parts und ggf.
        archived_original, part_count.

    Raises:
        FileNotFoundError: PDF nicht vorhanden.
        ValueError: Datei ist keine PDF.
        RuntimeError: einzelne Seite größer als die Zielgröße.
        FileExistsError: Archivdatei existiert bereits.
    """

    # Nur den Dateinamen übernehmen, keine fremden Pfade erlauben
    safe_filename = Path(filename).name
    source_path = PDF_DIR / safe_filename

    if not source_path.exists():
        raise FileNotFoundError(
            f"PDF nicht gefunden: {source_path}"
        )

    if source_path.suffix.lower() != ".pdf":
        raise ValueError("Die Datei ist keine PDF.")

    source_size = source_path.stat().st_size

    # Kleine Dateien müssen nicht geteilt werden
    if source_size <= TARGET_SIZE_BYTES:
        return {
            "status": "not_needed",
            "original": safe_filename,
            "parts": [
                {
                    "filename": safe_filename,
                    "size_bytes": source_size,
                    "size_mb": round(
                        source_size / 1024 / 1024,
                        2,
                    ),
                }
            ],
        }

    source_document = fitz.open(source_path)
    created_paths: list[Path] = []

    try:
        page_count = source_document.page_count

        # Sicherheitsfaktor 0,90:
        # Die berechneten Teile werden zunächst etwas kleiner angesetzt.
        estimated_pages_per_part = max(
            1,
            math.floor(
                page_count
                * TARGET_SIZE_BYTES
                / source_size
                * 0.90
            ),
        )

        start_page = 0
        part_number = 1

        while start_page < page_count:
            pages_in_part = min(
                estimated_pages_per_part,
                page_count - start_page,
            )

            while True:
                end_page = start_page + pages_in_part - 1

                output_name = (
                    f"{source_path.stem}"
                    f"__part_{part_number:03d}.pdf"
                )
                output_path = PDF_DIR / output_name
                temporary_path = PDF_DIR / (
                    f".{output_name}.tmp.pdf"
                )

                # Eventuelle temporäre Datei eines früheren Versuchs entfernen
                temporary_path.unlink(missing_ok=True)

                part_size = save_pages(
                    source_document=source_document,
                    start_page=start_page,
                    end_page=end_page,
                    output_path=temporary_path,
                )

                # Teil passt unter die Zielgröße
                if part_size <= TARGET_SIZE_BYTES:
                    temporary_path.replace(output_path)
                    created_paths.append(output_path)
                    break

                # Dieser Teil ist noch zu groß:
                # Seitenzahl um 20 Prozent reduzieren und erneut versuchen.
                temporary_path.unlink(missing_ok=True)

                if pages_in_part == 1:
                    raise RuntimeError(
                        f"Seite {start_page + 1} ist alleine "
                        f"größer als {TARGET_SIZE_MB} MB."
                    )

                pages_in_part = max(
                    1,
                    math.floor(pages_in_part * 0.80),
                )

            start_page = end_page + 1
            part_number += 1

    except Exception:
        # Bei einem Fehler keine unvollständigen Teile zurücklassen
        for path in created_paths:
            path.unlink(missing_ok=True)

        # Auch temporäre Dateien entfernen
        for path in PDF_DIR.glob(
            f".{source_path.stem}__part_*.tmp.pdf"
        ):
            path.unlink(missing_ok=True)

        raise

    finally:
        source_document.close()

    parts = [
        {
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "size_mb": round(
                path.stat().st_size / 1024 / 1024,
                2,
            ),
        }
        for path in created_paths
    ]

    # Original erst archivieren, wenn alle Teile erfolgreich erstellt wurden
    ARCHIVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    archive_path = ARCHIVE_DIR / source_path.name

    # Bereits vorhandenes gleichnamiges Archiv nicht überschreiben
    if archive_path.exists():
        raise FileExistsError(
            f"Archivdatei existiert bereits: {archive_path}"
        )

    shutil.move(
        str(source_path),
        str(archive_path),
    )

    return {
        "status": "split",
        "original": safe_filename,
        "archived_original": str(archive_path),
        "part_count": len(parts),
        "parts": parts,
    }