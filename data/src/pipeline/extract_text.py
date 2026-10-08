#!/usr/bin/env python3
# ============================================================
# extract_text.py — Node 2: — Text- und Tabellenextraktion mit PyMuPDF
#
# Extrahiert Fließtext und Tabellen (als Markdown, TAG) seitenweise.
# Kennzeichnet Seiten mit wenig Text und hohem Bildanteil als
# OCR-Kandidaten. Optional: Erkennung/Entfernung wiederkehrender
# Kopf- und Fußzeilen sowie Filterung nutzloser Tabellen.
#
# Warum PyMuPDF: schnellste Python-PDF-Bibliothek, C++-Backend,
#   native Table-Finder-API, VRAM-neutral (NF-05).
# ============================================================

from __future__ import annotations

import json
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF (import-Alias 'fitz' ist historisch)

# ── Logging auf stderr ────────────────────────────────────────
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] extract_text – %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Scan-Erkennung ───────────────────────────────────────────
# Eine Seite wird als OCR-Kandidat eingestuft, wenn:
#   - weniger als OCR_MIN_TEXT_LENGTH Zeichen extrahiert wurden
#   - gleichzeitig mindestens OCR_MIN_IMAGE_COVERAGE der Seite
#     aus Bildern besteht.
#
# Standardwerte:
#   OCR_MIN_TEXT_LENGTH = 80 Zeichen
#   OCR_MIN_IMAGE_COVERAGE = 0.35 (35 % Bildfläche)

MIN_TEXT_LENGTH: int = int(
    os.environ.get("OCR_MIN_TEXT_LENGTH", "80")
)

MIN_IMAGE_COVERAGE: float = float(
    os.environ.get("OCR_MIN_IMAGE_COVERAGE", "0.35")
)
# ── Kopf-/Fußzeilen-Analyse ───────────────────────────────────
# analyze = nur Kandidaten erkennen, nichts löschen
# clean   = erkannte Kandidaten entfernen
# off     = Funktion deaktivieren
HEADER_FOOTER_MODE: str = os.environ.get(
    "HEADER_FOOTER_MODE",
    "analyze",
).lower()

HEADER_TOP_RATIO: float = float(
    os.environ.get("HEADER_TOP_RATIO", "0.18")
)

FOOTER_BOTTOM_RATIO: float = float(
    os.environ.get("FOOTER_BOTTOM_RATIO", "0.82")
)

HEADER_FOOTER_REPEAT_RATIO: float = float(
    os.environ.get("HEADER_FOOTER_REPEAT_RATIO", "0.75")
)

HEADER_FOOTER_MIN_PAGES: int = int(
    os.environ.get("HEADER_FOOTER_MIN_PAGES", "3")
)

DEFAULT_RULES_PATH = Path(__file__).with_name(
    "header_footer_rules.json"
)

HEADER_FOOTER_RULES_PATH = Path(
    os.environ.get(
        "HEADER_FOOTER_RULES_PATH",
        str(DEFAULT_RULES_PATH),
    )
)

# ── Tabellenfilter ────────────────────────────────────────────
# off     = keine Tabellen filtern
# analyze = Tabellen klassifizieren und protokollieren,
#           aber nichts entfernen
# clean   = nur eindeutig nutzlose Tabellen entfernen
TABLE_FILTER_MODE: str = os.environ.get(
    "TABLE_FILTER_MODE",
    "analyze",
).lower()

TABLE_CLEAN_MODE: str = os.environ.get(
    "TABLE_CLEAN_MODE",
    "analyze",
).lower()

# ── Datenstrukturen ───────────────────────────────────────────

def _make_chunk(
    text: str,
    page: int,
    source: str,
    chunk_type: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Erzeugt einen einheitlichen Chunk-Dict für alle Nodes."""
    base: dict[str, Any] = {
        "text":       text,
        "page":       page,
        "source":     source,
        "type":       chunk_type,
    }
    if extra:
        base.update(extra)
    return base


# ── Kernfunktionen ────────────────────────────────────────────

def get_image_coverage(page: fitz.Page) -> float:
    """
    Ermittelt, wie viel der Seite ungefähr von Bildern bedeckt ist.

    Rückgabe:
        Wert zwischen 0.0 und 1.0
    """
    page_area = page.rect.width * page.rect.height
    if page_area <= 0:
        return 0.0

    image_area = 0.0

    try:
        for image in page.get_image_info(xrefs=True):
            bbox = image.get("bbox")
            if not bbox:
                continue

            rect = fitz.Rect(bbox)
            image_area += max(0.0, rect.width * rect.height)

    except Exception as exc:  # noqa: BLE001
        log.warning("Bildflächen-Erkennung fehlgeschlagen: %s", exc)
        return 0.0

    return min(image_area / page_area, 1.0)


def detect_is_scanned(page: fitz.Page, text: str) -> tuple[bool, float]:
    """
    Eine Seite gilt als OCR-Kandidat, wenn:

    1. nur wenig extrahierbarer Text vorhanden ist und
    2. Bilder mindestens 50 Prozent der Seite bedecken.

    Dadurch lösen kleine Logos oder Produktbilder nicht sofort OCR aus.
    """
    clean_text = text.strip()
    image_coverage = get_image_coverage(page)

    is_scanned = (
        len(clean_text) < MIN_TEXT_LENGTH
        and image_coverage >= MIN_IMAGE_COVERAGE
    )

    return is_scanned, image_coverage
def load_header_footer_rules() -> dict[str, list[str]]:
    """
    Lädt Regeln zur Klassifizierung wiederkehrender Kopf-
    und Fußzeilen.

    Fehlt die Datei oder ist sie ungültig, werden keine Zeilen
    automatisch entfernt.
    """
    empty_rules: dict[str, list[str]] = {
        "remove_exact": [],
        "remove_patterns": [],
        "keep_exact": [],
        "keep_patterns": [],
    }

    if not HEADER_FOOTER_RULES_PATH.exists():
        log.warning(
            "Regeldatei nicht gefunden: %s",
            HEADER_FOOTER_RULES_PATH,
        )
        return empty_rules

    try:
        with HEADER_FOOTER_RULES_PATH.open(
            "r",
            encoding="utf-8",
        ) as handle:
            data = json.load(handle)

    except (OSError, json.JSONDecodeError) as exc:
        log.error(
            "Regeldatei konnte nicht geladen werden: %s",
            exc,
        )
        return empty_rules

    rules = empty_rules.copy()

    for key in rules:
        values = data.get(key, [])

        if not isinstance(values, list):
            log.warning(
                "Regel '%s' ist keine Liste und wird ignoriert",
                key,
            )
            continue

        rules[key] = [
            str(value).strip()
            for value in values
            if str(value).strip()
        ]

    log.info(
        "Header/Footer-Regeln geladen: "
        "%d remove_exact | %d remove_patterns | "
        "%d keep_exact | %d keep_patterns",
        len(rules["remove_exact"]),
        len(rules["remove_patterns"]),
        len(rules["keep_exact"]),
        len(rules["keep_patterns"]),
    )

    return rules


def matches_any_pattern(
    text: str,
    patterns: list[str],
) -> bool:
    """Prüft Text gegen eine Liste regulärer Ausdrücke."""
    for pattern in patterns:
        try:
            if re.search(pattern, text, flags=re.IGNORECASE):
                return True
        except re.error as exc:
            log.warning(
                "Ungültiges Regex-Muster '%s': %s",
                pattern,
                exc,
            )

    return False


def classify_margin_line(
    text: str,
    rules: dict[str, list[str]],
) -> str:
    """
    Klassifiziert eine Randzeile.

    Rückgabe:
        keep    = ausdrücklich behalten
        remove  = ausdrücklich entfernen
        unknown = keine passende Regel
    """
    clean = re.sub(r"\s+", " ", text).strip()

    keep_exact = {
        item.casefold()
        for item in rules["keep_exact"]
    }

    remove_exact = {
        item.casefold()
        for item in rules["remove_exact"]
    }

    # Keep-Regeln haben immer Vorrang.
    if clean.casefold() in keep_exact:
        return "keep"

    if matches_any_pattern(clean, rules["keep_patterns"]):
        return "keep"

    if clean.casefold() in remove_exact:
        return "remove"

    if matches_any_pattern(clean, rules["remove_patterns"]):
        return "remove"

    return "unknown"

def normalize_repeated_line(text: str) -> str:
    """
    Normalisiert eine Zeile ausschließlich zur Erkennung von
    wiederkehrenden Kopf- und Fußzeilen.
    """
    text = text.replace("\u00ad", "")
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return ""

    # Datumsangaben vereinheitlichen
    text = re.sub(
        r"\b\d{1,2}\.\d{1,2}\.\d{4}\b",
        "<DATE>",
        text,
    )

    # Uhrzeiten vereinheitlichen
    text = re.sub(
        r"\b\d{1,2}:\d{2}(?::\d{2})?\b",
        "<TIME>",
        text,
    )

    # Reine Seitenzahlen vereinheitlichen
    if re.fullmatch(r"\d{1,4}", text):
        return "<PAGE_NUMBER>"

    return text


def get_margin_lines(page: fitz.Page) -> list[dict[str, str]]:
    """
    Liefert Textzeilen aus dem oberen und unteren Seitenbereich.
    Der mittlere Seiteninhalt wird nicht berücksichtigt.
    """
    results: list[dict[str, str]] = []

    page_height = float(page.rect.height)
    top_limit = page_height * HEADER_TOP_RATIO
    bottom_limit = page_height * FOOTER_BOTTOM_RATIO

    page_dict = page.get_text("dict")

    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue

        for line in block.get("lines", []):
            bbox = line.get("bbox")

            if not bbox or len(bbox) < 4:
                continue

            center_y = (float(bbox[1]) + float(bbox[3])) / 2

            if center_y <= top_limit:
                position = "top"
            elif center_y >= bottom_limit:
                position = "bottom"
            else:
                continue

            original = "".join(
                str(span.get("text", ""))
                for span in line.get("spans", [])
            ).strip()

            normalized = normalize_repeated_line(original)

            if not normalized:
                continue

            results.append(
                {
                    "position": position,
                    "original": original,
                    "normalized": normalized,
                }
            )

    return results


def detect_repeated_margin_lines(
    doc: fitz.Document,
    rules: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """
    Erkennt Zeilen, die innerhalb eines PDFs auf vielen Seiten
    im Kopf- oder Fußbereich vorkommen.
    """
    total_pages = len(doc)

    if total_pages < HEADER_FOOTER_MIN_PAGES:
        return []

    page_occurrences: dict[
        tuple[str, str],
        set[int],
    ] = defaultdict(set)

    examples: dict[
        tuple[str, str],
        set[str],
    ] = defaultdict(set)

    for page_index, page in enumerate(doc):
        seen_on_page: set[tuple[str, str]] = set()

        for item in get_margin_lines(page):
            key = (
                item["position"],
                item["normalized"],
            )

            examples[key].add(item["original"])

            if key not in seen_on_page:
                page_occurrences[key].add(page_index + 1)
                seen_on_page.add(key)

    candidates: list[dict[str, Any]] = []

    for (position, normalized), page_numbers in page_occurrences.items():
        repeat_ratio = len(page_numbers) / total_pages

        if repeat_ratio < HEADER_FOOTER_REPEAT_RATIO:
            continue

        example = sorted(
            examples[(position, normalized)]
        )[0]

        candidates.append(
            {
                "position": position,
                "normalized": normalized,
                "example": example,
                "pages_found": len(page_numbers),
                "total_pages": total_pages,
                "repeat_ratio": round(repeat_ratio, 3),
                "classification": classify_margin_line(
                    example,
                    rules,
                ),
            }
        )

    candidates.sort(
        key=lambda item: (
            item["position"],
            -item["repeat_ratio"],
            item["normalized"],
        )
    )

    return candidates

def clean_page_text(
    page: fitz.Page,
    repeated_lines: set[tuple[str, str]],
) -> tuple[str, list[str]]:
    """
    Entfernt nur bestätigte wiederkehrende Zeilen aus dem
    oberen und unteren Seitenbereich.

    Der mittlere Seiteninhalt wird niemals über diese Logik gelöscht.
    """
    removed_lines: list[str] = []
    kept_lines: list[str] = []

    page_height = float(page.rect.height)
    top_limit = page_height * HEADER_TOP_RATIO
    bottom_limit = page_height * FOOTER_BOTTOM_RATIO

    page_dict = page.get_text("dict")

    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue

        for line in block.get("lines", []):
            bbox = line.get("bbox")

            if not bbox or len(bbox) < 4:
                continue

            original = "".join(
                str(span.get("text", ""))
                for span in line.get("spans", [])
            ).strip()

            if not original:
                continue

            center_y = (float(bbox[1]) + float(bbox[3])) / 2

            position: str | None = None

            if center_y <= top_limit:
                position = "top"
            elif center_y >= bottom_limit:
                position = "bottom"

            normalized = normalize_repeated_line(original)

            if (
                position is not None
                and (position, normalized) in repeated_lines
            ):
                removed_lines.append(original)
                continue

            kept_lines.append(original)

    cleaned_text = "\n".join(kept_lines).strip()

    return cleaned_text, removed_lines

def extract_page_text(
    page: fitz.Page,
    page_num: int,
    source: str,
    repeated_lines: set[tuple[str, str]] | None = None,
) -> dict[str, Any] | None:
    """
    Extrahiert Fließtext einer Seite.

    Im Modus clean werden ausschließlich erkannte wiederkehrende
    Zeilen im oberen oder unteren Seitenbereich entfernt.
    """
    removed_lines: list[str] = []

    if HEADER_FOOTER_MODE == "clean" and repeated_lines:
        text, removed_lines = clean_page_text(
            page,
            repeated_lines,
        )
    else:
        text = page.get_text("text").strip()

    if not text:
        return None

    is_scan, image_coverage = detect_is_scanned(page, text)

    return _make_chunk(
        text=text,
        page=page_num,
        source=source,
        chunk_type="text",
        extra={
            "is_likely_scanned": is_scan,
            "text_length": len(text),
            "image_coverage": round(image_coverage, 3),
            "header_footer_cleaned": (
                HEADER_FOOTER_MODE == "clean"
            ),
            "removed_header_footer_lines": len(removed_lines),
        },
    )

def normalize_table_text(text: str) -> str:
    """Normalisiert Tabelleninhalt für die Qualitätsprüfung."""
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def count_markdown_table_rows(markdown: str) -> int:
    """
    Zählt in einer Markdown-Tabelle die inhaltlichen Zeilen.

    Die Markdown-Trennzeile wie |---|---| wird nicht gezählt.
    """
    row_count = 0

    for line in markdown.splitlines():
        line = line.strip()

        if not line or "|" not in line:
            continue

        cells = [
            cell.strip()
            for cell in line.strip("|").split("|")
        ]

        if not cells:
            continue

        # Markdown-Trennzeile erkennen: |---|:---:|---:|
        if all(
            re.fullmatch(r":?-{3,}:?", cell or "")
            for cell in cells
        ):
            continue

        if any(cells):
            row_count += 1

    return row_count

def remove_empty_placeholder_columns(
    markdown: str,
) -> tuple[str, list[str]]:
    """
    Entfernt nur künstliche ColN-Spalten, deren Datenzellen
    vollständig leer sind.

    Andere Spalten und Inhalte bleiben unverändert.
    """
    lines = [
        line.strip()
        for line in markdown.splitlines()
        if line.strip()
    ]

    if len(lines) < 2:
        return markdown, []

    rows = [
        [
            cell.strip()
            for cell in line.strip("|").split("|")
        ]
        for line in lines
        if "|" in line
    ]

    if len(rows) < 2:
        return markdown, []

    column_count = len(rows[0])

    if any(len(row) != column_count for row in rows):
        return markdown, []

    header = rows[0]
    removable: list[int] = []

    for column_index, header_cell in enumerate(header):
        if not re.fullmatch(
            r"col(?:umn)?\s*\d+",
            header_cell,
            flags=re.IGNORECASE,
        ):
            continue

        # Zeile 1 ist Header, Zeile 2 normalerweise Markdown-Trenner.
        data_cells = [
            row[column_index].strip()
            for row in rows[2:]
        ]

        if data_cells and all(not cell for cell in data_cells):
            removable.append(column_index)

    if not removable:
        return markdown, []

    removed_names = [
        header[column_index]
        for column_index in removable
    ]

    for row in rows:
        for column_index in reversed(removable):
            del row[column_index]

    cleaned = "\n".join(
        "|" + "|".join(row) + "|"
        for row in rows
    )

    return cleaned + "\n", removed_names

def classify_table(markdown: str) -> dict[str, Any]:
    """
    Klassifiziert eine extrahierte Tabelle konservativ.

    Rückgabe:
        classification:
            keep    = klar nützliche Tabelle
            remove  = eindeutig nutzloses Header/Footer-Fragment
            unknown = nicht eindeutig; sicherheitshalber behalten
    """
    normalized = normalize_table_text(markdown)
    lower = normalized.casefold()
    row_count = count_markdown_table_rows(markdown)

    has_url = bool(
        re.search(
            r"(?:https?://|www\.|"
            r"\b[\w.-]+\.(?:de|com|net|org|eu)\b)",
            lower,
            flags=re.IGNORECASE,
        )
    )

    has_product_identifier = bool(
        re.search(
            r"\b(?:"
            r"gtin|ean|"
            r"artikel(?:nummer)?|artikelnr\.?|"
            r"article\s*(?:number|no\.?)|"
            r"bestell(?:nummer)?|bestellnr\.?|"
            r"order\s*(?:number|no\.?)|"
            r"materialnummer|material\s*(?:number|no\.?)"
            r")\b",
            lower,
            flags=re.IGNORECASE,
        )
    )

    has_technical_keyword = bool(
        re.search(
            r"\b(?:"
            r"spannung|voltage|"
            r"strom|current|"
            r"leistung|power|"
            r"querschnitt|cross[- ]?section|"
            r"gewicht|weight|"
            r"länge|length|"
            r"breite|width|"
            r"höhe|height|"
            r"drehmoment|torque|"
            r"schutzart|protection|"
            r"temperatur|temperature|"
            r"abmessung|dimensions?"
            r")\b",
            lower,
            flags=re.IGNORECASE,
        )
    )

    has_technical_value = bool(
        re.search(
            r"\b\d+(?:[.,]\d+)?\s*"
            r"(?:"
            r"mm²?|cm²?|m²?|"
            r"mm|cm|m|"
            r"mv|v|kv|"
            r"ma|a|ka|"
            r"mw|w|kw|"
            r"hz|khz|mhz|"
            r"°c|"
            r"mg|g|kg|"
            r"nm|ncm|"
            r"bar|pa|kpa|mpa"
            r")\b",
            lower,
            flags=re.IGNORECASE,
        )
    )

    has_long_number = bool(
        re.search(r"\b\d{8,14}\b", lower)
    )

    useful_signals = sum(
        [
            has_product_identifier,
            has_technical_keyword,
            has_technical_value,
            has_long_number,
        ]
    )

    # Klare Produkt- oder technische Tabelle.
    if useful_signals > 0:
        classification = "keep"
        reason = "Produktkennung oder technische Daten erkannt"

    # Sehr kleine Web-/Footer-Tabelle ohne Produktsignale.
    elif(row_count <= 2 and has_url and useful_signals == 0):
        classification = "remove"
        reason = "kleine URL-Tabelle ohne Produktdaten"

    # Extrem kleine Tabelle ohne Informationsgehalt.
    elif row_count <= 1 and len(normalized) < 80:
        classification = "remove"
        reason = "sehr kleine Tabelle ohne erkennbaren Nutzinhalt"

    else:
        # Bei Unsicherheit immer behalten.
        classification = "unknown"
        reason = "nicht eindeutig klassifizierbar"

    return {
        "classification": classification,
        "reason": reason,
        "row_count": row_count,
        "has_url": has_url,
        "has_product_identifier": has_product_identifier,
        "has_technical_keyword": has_technical_keyword,
        "has_technical_value": has_technical_value,
        "has_long_number": has_long_number,
    }

def extract_page_tables(page: fitz.Page, page_num: int, source: str) -> list[dict[str, Any]]:
    """
    Extrahiert Tabellen einer Seite als Markdown.

    Im Modus:
        off     werden alle Tabellen übernommen
        analyze werden Tabellen klassifiziert, aber alle behalten
        clean   werden nur eindeutig nutzlose Tabellen verworfen
    """
    table_chunks: list[dict[str, Any]] = []

    try:
        for idx, table in enumerate(page.find_tables()):
            raw_md = table.to_markdown()

            if not raw_md or not raw_md.strip():
                continue

            removed_columns: list[str] = []

            if TABLE_CLEAN_MODE == "clean":
                md, removed_columns = remove_empty_placeholder_columns(
                    raw_md
                )
            else:
                md = raw_md

            if removed_columns:
                log.info(
                    "Tabelle Seite %d Nr.%d bereinigt: leere Spalten %s",
                    page_num,
                    idx + 1,
                    ", ".join(removed_columns),
                )

            analysis = classify_table(md)
            classification = analysis["classification"]

            log.info(
                "Tabelle Seite %d Nr.%d: %s – %s "
                "(Zeilen=%d, URL=%s, Produktkennung=%s, "
                "Technik=%s, technischer Wert=%s)",
                page_num,
                idx + 1,
                classification.upper(),
                analysis["reason"],
                analysis["row_count"],
                analysis["has_url"],
                analysis["has_product_identifier"],
                analysis["has_technical_keyword"],
                analysis["has_technical_value"],
            )

            if (
                TABLE_FILTER_MODE == "clean"
                and classification == "remove"
            ):
                log.info(
                    "Tabelle Seite %d Nr.%d entfernt: %s",
                    page_num,
                    idx + 1,
                    analysis["reason"],
                )
                continue

            table_chunks.append(
                _make_chunk(
                    text=(
                        f"[TABELLE Seite {page_num} Nr.{idx + 1}]\n"
                        f"{md}"
                    ),
                    page=page_num,
                    source=source,
                    chunk_type="table",
                    extra={
                        "table_classification": classification,
                        "table_filter_reason": analysis["reason"],
                        "table_row_count": analysis["row_count"],
                        "table_cleaned": bool(removed_columns),
                        "table_removed_columns": ",".join(removed_columns),
                    },
                )
            )

    except Exception as exc:  # noqa: BLE001
        log.warning(
            "Tabellen-Parsing Seite %d fehlgeschlagen: %s",
            page_num,
            exc,
        )

    return table_chunks


def extract_pdf(pdf_path: Path) -> dict[str, Any]:
    """
    Verarbeitet das gesamte PDF seitenweise.

    Returns:
        Dict mit filename, has_text, text_blocks, table_blocks,
        scanned_pages, total_pages, chunks.
    """
    filename = pdf_path.name
    log.info("Öffne PDF: %s", filename)

    doc = fitz.open(str(pdf_path))

    header_footer_rules = load_header_footer_rules()

    repeated_candidates = detect_repeated_margin_lines(
        doc,
        header_footer_rules,
    )

    removable_repeated_lines: set[tuple[str, str]] = {
        (
            candidate["position"],
            candidate["normalized"],
        )
        for candidate in repeated_candidates
        if candidate["classification"] == "remove"
    }

    if HEADER_FOOTER_MODE == "analyze":
        log.info(
            "%d mögliche wiederkehrende Kopf-/Fußzeilen erkannt",
            len(repeated_candidates),
        )

        for candidate in repeated_candidates:
            log.info(
                "Kandidat [%s] [%s] %.1f%%: %s",
                candidate["position"],
                candidate["classification"].upper(),
                candidate["repeat_ratio"] * 100,
                candidate["example"],
            )
    text_chunks:  list[dict[str, Any]] = []
    table_chunks: list[dict[str, Any]] = []
    scanned_page_numbers: list[int] = []
    total_pages: int = len(doc)

    for page_num_0, page in enumerate(doc):
        page_num = page_num_0 + 1  # 1-basiert für Quellenangaben

        try:
            # Fliesstext extrahieren
            text_chunk = extract_page_text(page, page_num, filename, removable_repeated_lines,)
            if text_chunk:
                if text_chunk.get("is_likely_scanned"):
                    scanned_page_numbers.append(page_num)

                text_chunks.append(text_chunk)

            else:
                # Kein Text allein bedeutet noch nicht Scan.
                # Nur eine große Bildfläche macht die Seite zum OCR-Kandidaten.
                image_coverage = get_image_coverage(page)

                if image_coverage >= MIN_IMAGE_COVERAGE:
                    scanned_page_numbers.append(page_num)

            # Tabellen extrahieren (TAG)
            table_chunks.extend(extract_page_tables(page, page_num, filename))

        except Exception as exc:  # noqa: BLE001
            log.error("Fehler auf Seite %d: %s", page_num, exc)

    doc.close()
    has_text = len(text_chunks) > 0
    needs_ocr = len(scanned_page_numbers) > 0
    log.info(
        "Abgeschlossen: %d Text-Blöcke | %d Tabellen | %d/%d OCR-Kandidaten",
        len(text_chunks),
        len(table_chunks),
        len(scanned_page_numbers),
        total_pages,
    )

    return {
        "filename": filename,
        "has_text": has_text,
        "needs_ocr": needs_ocr,
        "text_blocks": len(text_chunks),
        "table_blocks": len(table_chunks),
        "scanned_pages": len(scanned_page_numbers),
        "scanned_page_numbers": scanned_page_numbers,
        "total_pages": total_pages,
        "header_footer_mode": HEADER_FOOTER_MODE,
        "header_footer_candidates": repeated_candidates,
                "header_footer_removable_count": len(
            removable_repeated_lines
        ),
        "chunks": text_chunks + table_chunks,
    }


def main() -> None:
    if len(sys.argv) < 2:
        log.error("Aufruf: extract_text.py <pdf_pfad>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        log.error("PDF nicht gefunden: %s", pdf_path)
        sys.exit(1)

    result = extract_pdf(pdf_path)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log.critical("Unerwarteter Fehler: %s", exc, exc_info=True)
        sys.exit(1)
