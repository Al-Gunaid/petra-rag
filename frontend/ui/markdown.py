"""
ui/markdown.py — Darstellung von Chat-Text mit Markdown-Tabellen.

`st.markdown` stellt breite Vergleichstabellen ohne horizontales Scrollen
dar, ignoriert die Spaltenausrichtung der Trennzeile und erkennt Tabellen
ohne Trennzeile (häufig bei LLM-Ausgaben) nicht. Tabellenblöcke werden
daher als HTML-`<table>` in einem scrollbaren Container gerendert
(ui/theme.py: `.petra-table-wrap`); eine fehlende Trennzeile wird toleriert.

Voraussetzung sind erhaltene Zeilenumbrüche beim Streaming
(services/chat.py: _stream_chunks).

Sicherheit: Zellinhalte werden zuerst mit `html.escape` maskiert; danach
wird nur eine begrenzte Inline-Syntax (fett, kursiv, Code, Zeilenumbruch,
http(s)-Link) in HTML umgesetzt. HTML aus der Modellausgabe wird so nicht
ausgeführt.

Text außerhalb von Tabellen (Fließtext, Listen, Codeblöcke, Zitate) wird
unverändert an `st.markdown` übergeben.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from typing import Optional

import streamlit as st

# Zerlegt eine Zeile an Pipe-Zeichen, die NICHT mit \ maskiert sind.
_SPLIT_PIPE_RE = re.compile(r"(?<!\\)\|")

# Trennzeile einer Markdown-Tabelle: ---, :---, ---:, :---:
_DELIM_CELL_RE = re.compile(r"^:?-{1,}:?$")

# Beginn/Ende eines Codeblocks (``` oder ~~~) — Inhalt bleibt unangetastet.
_FENCE_RE = re.compile(r"^\s*(?:```|~~~)")

# Zahlenartige Zelle (inkl. Einheit) -> wird rechtsbündig ausgerichtet,
# sofern die Trennzeile keine explizite Ausrichtung vorgibt.
_NUMERIC_RE = re.compile(
    r"^[-+±~<>≤≥]?\s*\d[\d.,\s–-]*"
    r"(?:\s*(?:%|°C|°|mm²|mm2|mm|cm|m|kg|g|A|mA|V|kV|W|kW|Ω|Nm|Hz|kHz|MHz|s|ms|µm))?$"
)


# ── Datenmodell ─────────────────────────────────────────────────────────

@dataclass
class MarkdownTable:
    """Geparste Markdown-Tabelle (rein datenhaltend, ohne Streamlit-Bezug)."""

    header: list[str]
    rows: list[list[str]] = field(default_factory=list)
    aligns: list[str] = field(default_factory=list)  # "left" | "right" | "center"

    @property
    def column_count(self) -> int:
        return len(self.header)


@dataclass
class Block:
    """Ein Abschnitt des Antworttextes: entweder Fließtext oder Tabelle."""

    kind: str  # "text" | "table"
    text: str = ""
    table: Optional[MarkdownTable] = None


# ── Zerlegung in Zellen / Erkennung ─────────────────────────────────────

def split_cells(line: str) -> list[str]:
    """Zerlegt eine Tabellenzeile in ihre Zellen.

    Führende/abschließende Pipes sind in Markdown optional — beide
    Schreibweisen (`| a | b |` und `a | b`) werden unterstützt. Mit `\\|`
    maskierte Pipes gelten als Zellinhalt, nicht als Trenner.
    """
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|") and not stripped.endswith("\\|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in _SPLIT_PIPE_RE.split(stripped)]


def _pipe_count(line: str) -> int:
    return len(_SPLIT_PIPE_RE.findall(line))


def is_delimiter_row(line: str) -> bool:
    """True, wenn die Zeile die Trennzeile einer Markdown-Tabelle ist.

    Mindestens zwei Zellen sind Pflicht — sonst würde eine normale
    Markdown-Trennlinie (`---`, horizontale Linie) fälschlich als
    Tabellen-Trennzeile gelten.
    """
    if "-" not in line:
        return False
    cells = split_cells(line)
    if len(cells) < 2:
        return False
    return all(cell and _DELIM_CELL_RE.match(cell) for cell in cells)


def _looks_like_row(line: str) -> bool:
    """Heuristik für eine Tabellen-Datenzeile (ohne Trennzeile zu sein)."""
    stripped = line.strip()
    if not stripped or _pipe_count(stripped) < 2:
        return False
    return len(split_cells(stripped)) >= 2


def _alignments_from_delimiter(line: str) -> list[str]:
    aligns: list[str] = []
    for cell in split_cells(line):
        left = cell.startswith(":")
        right = cell.endswith(":")
        if left and right:
            aligns.append("center")
        elif right:
            aligns.append("right")
        else:
            aligns.append("left")
    return aligns


def _normalize_width(cells: list[str], width: int) -> list[str]:
    """Gleicht abweichende Spaltenzahlen aus (Modelle zählen gern falsch)."""
    if len(cells) < width:
        return cells + [""] * (width - len(cells))
    return cells[:width]


def parse_table(lines: list[str]) -> Optional[MarkdownTable]:
    """Baut aus zusammenhängenden Tabellenzeilen ein `MarkdownTable`.

    Unterstützt beide in der Praxis auftretenden Formen:
      - regulär: Kopfzeile + Trennzeile + Datenzeilen,
      - ohne Trennzeile (kommt bei LLM-Ausgaben vor): erste Zeile wird als
        Kopfzeile interpretiert. Ohne diese Toleranz erschiene genau so
        eine Tabelle wieder als Pipe-Text.
    """
    rows = [ln for ln in lines if ln.strip()]
    if len(rows) < 2:
        return None

    header_cells = split_cells(rows[0])
    aligns: list[str] = []
    body_start = 1

    if is_delimiter_row(rows[1]):
        aligns = _alignments_from_delimiter(rows[1])
        body_start = 2
    elif not _looks_like_row(rows[1]):
        return None

    width = max(len(header_cells), len(aligns) or 0)
    if width < 2:
        return None

    header = _normalize_width(header_cells, width)
    aligns = _normalize_width(aligns, width) if aligns else ["" for _ in range(width)]

    body: list[list[str]] = []
    for line in rows[body_start:]:
        if is_delimiter_row(line):
            continue  # doppelte Trennzeile — ignorieren statt als Daten zu zeigen
        body.append(_normalize_width(split_cells(line), width))

    # Ausrichtung ohne explizite Vorgabe: numerische Spalten rechtsbündig,
    # alles andere linksbündig ("Spalten sauber ausrichten").
    final_aligns: list[str] = []
    for col in range(width):
        if aligns[col]:
            final_aligns.append(aligns[col])
            continue
        values = [row[col].strip() for row in body if row[col].strip()]
        numeric = bool(values) and all(_NUMERIC_RE.match(v) for v in values)
        final_aligns.append("right" if numeric else "left")

    return MarkdownTable(header=header, rows=body, aligns=final_aligns)


# ── Blockzerlegung ──────────────────────────────────────────────────────

def split_blocks(text: str) -> list[Block]:
    """Zerlegt einen Antworttext in Fließtext- und Tabellenblöcke.

    Codeblöcke (``` / ~~~) werden übersprungen: eine Pipe-Tabelle INNERHALB
    eines Codeblocks ist beabsichtigter Quelltext und darf nicht in eine
    HTML-Tabelle umgewandelt werden.
    """
    blocks: list[Block] = []
    buffer: list[str] = []
    lines = (text or "").split("\n")
    i = 0
    in_fence = False

    def flush_text() -> None:
        if buffer:
            chunk = "\n".join(buffer).strip("\n")
            if chunk.strip():
                blocks.append(Block(kind="text", text=chunk))
            buffer.clear()

    while i < len(lines):
        line = lines[i]

        if _FENCE_RE.match(line):
            in_fence = not in_fence
            buffer.append(line)
            i += 1
            continue

        if in_fence or not _looks_like_row(line):
            buffer.append(line)
            i += 1
            continue

        # Kandidat: zusammenhängender Block aus Tabellenzeilen einsammeln.
        start = i
        candidate: list[str] = []
        while i < len(lines) and (
            _looks_like_row(lines[i]) or is_delimiter_row(lines[i])
        ):
            candidate.append(lines[i])
            i += 1

        table = parse_table(candidate)
        if table is None:
            buffer.extend(candidate)
            i = max(i, start + 1)
            continue

        flush_text()
        blocks.append(Block(kind="table", table=table))

    flush_text()
    return blocks


# ── Inline-Formatierung + HTML-Erzeugung ────────────────────────────────

def _inline_to_html(cell: str) -> str:
    """Escaped den Zellinhalt und löst danach begrenzte Inline-Syntax auf."""
    out = html.escape(cell, quote=False)
    out = out.replace("\\|", "|")
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    out = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"(?<![\w*])\*([^*]+)\*(?![\w*])", r"<em>\1</em>", out)
    out = re.sub(r"(?<![\w_])_([^_]+)_(?![\w_])", r"<em>\1</em>", out)
    out = re.sub(
        r"\[([^\]]+)\]\((https?://[^\s)]+)\)",
        r'<a href="\2" target="_blank" rel="noopener noreferrer">\1</a>',
        out,
    )
    out = re.sub(r"&lt;br\s*/?&gt;", "<br/>", out, flags=re.IGNORECASE)
    return out


def table_to_html(table: MarkdownTable) -> str:
    """Rendert die Tabelle als HTML in einem horizontal scrollbaren Rahmen."""
    head = "".join(
        f'<th style="text-align:{align}">{_inline_to_html(cell)}</th>'
        for cell, align in zip(table.header, table.aligns)
    )
    body_rows = []
    for row in table.rows:
        cells = "".join(
            f'<td style="text-align:{align}">{_inline_to_html(cell)}</td>'
            for cell, align in zip(row, table.aligns)
        )
        body_rows.append(f"<tr>{cells}</tr>")
    body = "".join(body_rows)
    return (
        '<div class="petra-table-wrap">'
        '<table class="petra-table">'
        f"<thead><tr>{head}</tr></thead>"
        f"<tbody>{body}</tbody>"
        "</table></div>"
    )


def contains_table(text: str) -> bool:
    """True, wenn der Text mindestens einen erkannten Tabellenblock enthält."""
    return any(block.kind == "table" for block in split_blocks(text))


# ── Streamlit-Rendering ─────────────────────────────────────────────────

def render_markdown(text: str) -> None:
    """Rendert einen Chat-Text; Tabellenblöcke als echte Tabellen.

    Enthält der Text keine Tabelle, verhält sich die Funktion exakt wie
    das bisherige `st.markdown(text)` — normale Antworten, Listen,
    Codeblöcke und Zitate bleiben unverändert.
    """
    if not text:
        st.markdown("")
        return

    blocks = split_blocks(text)
    if not blocks:
        st.markdown(text)
        return

    for block in blocks:
        if block.kind == "table" and block.table is not None:
            st.markdown(table_to_html(block.table), unsafe_allow_html=True)
        elif block.text.strip():
            st.markdown(block.text)
