#!/usr/bin/env python3
# ============================================================
# ocr_fallback.py — Node 3: Tesseract OCR für Scan-PDFs
#
# Verarbeitet nur die übergebenen Seiten (OCR-Kandidaten aus
# extract_text.py). Lokal und offline (NF-01).
#
# Aufruf : ocr_fallback.py <pdf_pfad> '[seitennummern als JSON-Liste]'
# Ausgabe: JSON über stdout; Logging über stderr.
#
# Rendering mit 300 DPI (Empfehlung der Tesseract-Dokumentation).
# ============================================================

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import fitz          # PyMuPDF – Seitenrendering
import pytesseract   # Tesseract OCR Python-Wrapper
from PIL import Image

# ── Logging auf stderr ────────────────────────────────────────
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] ocr_fallback – %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Konfiguration via ENV ─────────────────────────────────────
DPI: int = int(os.environ.get("OCR_DPI", "300"))
# Skalierungsfaktor: Standard-PDF = 72 DPI; 300/72 ≈ 4.17×
DPI_SCALE: float = DPI / 72.0

# Sprachpaket: DE+EN für gemischte Weidmüller-Dokumente
OCR_LANG: str = os.environ.get("OCR_LANG", "deu+eng")

# Konfidenz-Schwellwert laut Spez A.3 (≥ 0.75 = 75%)
OCR_CONFIDENCE_THRESHOLD: float = float(
    os.environ.get("OCR_CONFIDENCE_THRESHOLD", "0.75")
)


# ── Kernfunktionen ────────────────────────────────────────────

def render_page_to_pil(page: fitz.Page) -> Image.Image:
    """
    Rendert die gesamte Seite mit DPI als RGB-Bild.

    Warum Matrix statt clip: Matrix skaliert die gesamte Seite
    gleichmäßig ohne Randverluste.
    """
    mat = fitz.Matrix(DPI_SCALE, DPI_SCALE)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def run_tesseract(img: Image.Image, lang: str) -> tuple[str, float]:
    """
    Führt Tesseract aus und gibt (text, avg_confidence) zurück.

    Warum image_to_data statt image_to_string: liefert Konfidenz-
    Scores pro Wort, ermöglicht Qualitäts-Metadatum im Chunk
    (Spez FA-ING-10 Audit-Trail, A.3 Qualitätsschwellen).

    Returns:
        (text, mittlere Konfidenz 0.0–1.0)

    Raises:
        pytesseract.TesseractNotFoundError: Tesseract nicht installiert.
    """
    try:
        data = pytesseract.image_to_data(
            img,
            lang=lang,
            output_type=pytesseract.Output.DICT,
        )

        confidences: list[float] = []

        for raw_confidence in data["conf"]:
            try:
                confidence = float(raw_confidence)
            except (TypeError, ValueError):
                continue

            if confidence > 0:
                confidences.append(confidence)

        avg_conf = (
            sum(confidences) / len(confidences) / 100.0
            if confidences
            else 0.0
        )

        text = pytesseract.image_to_string(
            img,
            lang=lang,
        ).strip()

        return text, avg_conf

    except pytesseract.TesseractNotFoundError:
        log.error("Tesseract nicht gefunden – bitte installieren: apt install tesseract-ocr")
        raise


def ocr_pdf(
    pdf_path: Path,
    page_numbers: list[int] | None = None,
) -> dict[str, Any]:
    """
    Führt OCR nur auf den angegebenen Seiten durch.

    page_numbers:
        1-basierte Seitennummern, zum Beispiel [2, 5, 7].

        Wenn keine Seitenliste übergeben wird, werden aus
        Sicherheitsgründen keine Seiten verarbeitet. Dadurch wird
        verhindert, dass gemischte PDFs komplett doppelt erfasst werden.

    Returns:
        Dict mit filename, ocr_blocks, requested_pages, processed_pages,
        low_confidence_pages, failed_pages und chunks (type="ocr").
    """
    filename = pdf_path.name

    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)

    requested_pages = page_numbers or []

    # Nur gültige, eindeutige Seitennummern verwenden.
    valid_pages = sorted({
        page_num
        for page_num in requested_pages
        if isinstance(page_num, int) and 1 <= page_num <= total_pages
    })

    log.info(
        "Starte OCR: %s | Seiten=%s | DPI=%d | lang=%s",
        filename,
        valid_pages,
        DPI,
        OCR_LANG,
    )

    chunks: list[dict[str, Any]] = []
    low_conf_pages: list[int] = []
    failed_pages: list[int] = []

    for page_num in valid_pages:
        page = doc[page_num - 1]

        log.info(
            "OCR Seite %d/%d",
            page_num,
            total_pages,
        )

        try:
            img = render_page_to_pil(page)
            text, conf = run_tesseract(img, OCR_LANG)

            if not text:
                log.warning(
                    "Seite %d: OCR hat keinen Text erkannt",
                    page_num,
                )
                continue

            if conf < OCR_CONFIDENCE_THRESHOLD:
                log.warning(
                    "Seite %d: niedrige OCR-Konfidenz %.2f (< %.2f)",
                    page_num,
                    conf,
                    OCR_CONFIDENCE_THRESHOLD,
                )
                low_conf_pages.append(page_num)

            chunks.append({
                "text": text,
                "page": page_num,
                "source": filename,
                "type": "ocr",
                "ocr_confidence": round(conf, 4),
            })

        except Exception as exc:  # noqa: BLE001
            log.error("OCR-Fehler Seite %d: %s", page_num, exc)
            failed_pages.append(page_num)

    doc.close()

    log.info(
        "OCR abgeschlossen: %d Blöcke aus %d angeforderten Seiten",
        len(chunks),
        len(valid_pages),
    )

    return {
        "filename": filename,
        "ocr_blocks": len(chunks),
        "requested_pages": requested_pages,
        "processed_pages": valid_pages,
        "low_confidence_pages": low_conf_pages,
        "failed_pages": failed_pages,
        "chunks": chunks,
    }

def main() -> None:
    """CLI: liest PDF-Pfad und optionale JSON-Seitenliste aus argv."""
    if len(sys.argv) < 2:
        log.error(
            "Aufruf: ocr_fallback.py <pdf_pfad> "
            "'[seitennummern]'"
        )
        sys.exit(1)

    pdf_path = Path(sys.argv[1])

    if not pdf_path.exists():
        log.error("PDF nicht gefunden: %s", pdf_path)
        sys.exit(1)

    page_numbers: list[int] = []

    if len(sys.argv) >= 3 and sys.argv[2].strip():
        try:
            raw_pages = json.loads(sys.argv[2])

            if not isinstance(raw_pages, list):
                raise ValueError(
                    "Seitenangabe muss eine JSON-Liste sein"
                )

            page_numbers = [
                int(page)
                for page in raw_pages
                if str(page).strip().isdigit()
            ]

        except (json.JSONDecodeError, ValueError) as exc:
            log.error(
                "Ungültige Seitenliste '%s': %s",
                sys.argv[2],
                exc,
            )
            sys.exit(1)

    if not page_numbers:
        log.warning(
            "Keine OCR-Seiten übergeben – OCR wird nicht ausgeführt"
        )

    result = ocr_pdf(
        pdf_path=pdf_path,
        page_numbers=page_numbers,
    )

    print(json.dumps(result, ensure_ascii=False))

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log.critical("Unerwarteter Fehler: %s", exc, exc_info=True)
        sys.exit(1)
