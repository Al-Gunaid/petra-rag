# ============================================================
# petra_hybrid/table_serializer.py — Partial TAG (Schicht 2c)
#
# Serialisiert jede Tabellenzeile als eigenen Text-Chunk inkl.
# Spaltenüberschriften, statt die Tabelle durch Standard-Chunking zu
# zerlegen. Kein Text-to-SQL.
#
# Eingabe: Tabellen als List[List[str]] (Zeilen -> Zellen), wie
# PyMuPDF `table.extract()` sie liefert. Die Kernfunktion ist von PyMuPDF
# unabhängig und separat testbar.
# ============================================================
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TableChunk:
    """Eine strukturerhaltend serialisierte Tabellenzeile (Partial TAG)."""
    chunk_id: str
    text: str  # Kopf- und genau eine Datenzeile als "Spalte: Wert | …"
    metadata: dict = field(default_factory=dict)


def _clean_cell(cell: str | None) -> str:
    if cell is None:
        return ""
    return " ".join(str(cell).split())


def serialize_table(
    table_rows: list[list[str]],
    *,
    document: str,
    page: int,
    manufacturer: str = "Weidmüller",
    product_hint: str | None = None,
    table_index: int = 0,
) -> list[TableChunk]:
    """Serialisiert eine Tabelle zeilenweise in Chunks.

    Jede Datenzeile wird mit den Spaltenüberschriften kombiniert
    ("Schutzart: IP67 | Messbereich: 0.2-2m | …"). Damit finden BM25 und
    Dense Retrieval auch Anfragen zu einem einzelnen Attribut; die
    Redundanz ist für kleine Produkttabellen (< 50 Zeilen) vertretbar.

    Args:
        table_rows: Erste Zeile = Kopfzeile, übrige = Datenzeilen.
        document: Dokumentname für Quellenangabe und chunk_id.
        page: Seitennummer.
        manufacturer: Wert für den Hersteller-Metadatenfilter.
        product_hint: Produktname/Artikelnummer, falls die erste Spalte leer ist.
        table_index: Index der Tabelle auf der Seite (Teil der chunk_id).

    Returns:
        Liste von TableChunk; leer ohne Kopf- oder Datenzeilen.
    """
    if not table_rows or len(table_rows) < 2:
        return []

    header = [_clean_cell(h) for h in table_rows[0]]
    if not any(header):
        return []

    chunks: list[TableChunk] = []
    for row_idx, row in enumerate(table_rows[1:], start=1):
        cells = [_clean_cell(c) for c in row]
        # Zeile ohne jeglichen Inhalt überspringen (F-06 Chunk-Filterung analog)
        if not any(cells):
            continue    # leere Zeile

        pairs = [
            f"{h}: {c}" for h, c in zip(header, cells) if h and c
        ]
        if not pairs:
            continue

        # Erste Spalte enthält in der Regel die Produktbezeichnung.
        row_product = cells[0] if cells and cells[0] else product_hint or "unbekannt"

        row_text = (
            f"[Tabelle aus {document}, Seite {page}] "
            f"Produkt/Zeile: {row_product}. " + " | ".join(pairs)
        )

        chunk_id = f"{document}::p{page}::tbl{table_index}::row{row_idx}"
        chunks.append(
            TableChunk(
                chunk_id=chunk_id,
                text=row_text,
                metadata={
                    "source_type": "table",
                    "document": document,
                    "page": page,
                    "manufacturer": manufacturer,
                    "product": row_product,
                    "table_index": table_index,
                    "row_index": row_idx,
                    "columns": header,
                },
            )
        )
    return chunks


def serialize_tables_from_pymupdf(
    page_tables: list,  # page.find_tables().tables
    *,
    document: str,
    page: int,
    manufacturer: str = "Weidmüller",
    product_hint: str | None = None,
) -> list[TableChunk]:
    """Adapter für PyMuPDF-Tabellenobjekte (UC-06).

    Tabellen, deren Extraktion fehlschlägt, werden übersprungen; ihr Inhalt
    gelangt über den regulären Chunking-Pfad als Fließtext in den Index.
    """
    all_chunks: list[TableChunk] = []
    for idx, table in enumerate(page_tables or []):
        try:
            rows = table.extract()  # List[List[str]]
        except Exception:  # noqa: BLE001 — Fallback auf Fließtext, siehe Docstring
            continue
        all_chunks.extend(
            serialize_table(
                rows,
                document=document,
                page=page,
                manufacturer=manufacturer,
                product_hint=product_hint,
                table_index=idx,
            )
        )
    return all_chunks
