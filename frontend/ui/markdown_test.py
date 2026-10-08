"""
ui/markdown_test.py — Tests für die Tabellendarstellung in ui/markdown.py
ohne laufendes Streamlit (Streamlit wird durch einen Stub ersetzt).

Ausführen:  python3 ui/markdown_test.py   (oder: python3 run_tests.py)

Geprüft werden das verlustfreie Streaming (services/chat.py: _stream_chunks),
Tabellenerkennung, Spaltenausrichtung, HTML-Ausgabe inkl. Escaping sowie
die unveränderte Darstellung normaler Antworten.
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _install_streamlit_stub() -> None:
    """Minimaler Streamlit-Ersatz (ui/markdown.py importiert streamlit)."""
    if "streamlit" in sys.modules:
        return
    mod = types.ModuleType("streamlit")
    mod.rendered = []
    mod.markdown = lambda body="", **kw: mod.rendered.append((body, kw))
    mod.cache_resource = lambda *a, **k: (lambda fn: fn)
    mod.session_state = {}
    sys.modules["streamlit"] = mod


_install_streamlit_stub()

import streamlit as st  # noqa: E402  (Stub)

from services.chat import _stream_chunks  # noqa: E402
from ui import markdown as md  # noqa: E402

_PASSED = 0
_FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    global _PASSED
    if condition:
        _PASSED += 1
        print(f"  ✅ {name}")
    else:
        _FAILED.append(name)
        print(f"  ❌ {name}" + (f"  →  {detail}" if detail else ""))


TABLE_ANSWER = """Für den Vergleich der Reihenklemmen:

| Produkt  | Querschnitt | Nennstrom | Bemerkung        |
|----------|------------:|----------:|:-----------------|
| WDU 2.5  | 0,5–4 mm²   | 24 A      | **Standard**     |
| WDU 4    | 1,5–6 mm²   | 32 A      | für höhere Last  |

Beide Klemmen sind schraubenlos anschließbar.
"""


def run() -> bool:
    print("\n1) Primärursache: verlustfreies Streaming (services/chat.py)")
    rebuilt = "".join(_stream_chunks(TABLE_ANSWER))
    check("Streaming rekonstruiert den Text zeichengenau", rebuilt == TABLE_ANSWER)
    check(
        "Zeilenumbrüche überleben das Streaming",
        rebuilt.count("\n") == TABLE_ANSWER.count("\n"),
        f"{rebuilt.count(chr(10))} statt {TABLE_ANSWER.count(chr(10))}",
    )
    old_behaviour = " ".join(TABLE_ANSWER.split())
    check(
        "alte Implementierung hätte die Tabelle zerstört (Gegenprobe)",
        "\n" not in old_behaviour and not md.contains_table(old_behaviour),
    )

    print("\n2) Tabellenerkennung")
    blocks = md.split_blocks(TABLE_ANSWER)
    kinds = [b.kind for b in blocks]
    check("Text/Tabelle/Text korrekt getrennt", kinds == ["text", "table", "text"], str(kinds))
    table = blocks[1].table
    check("4 Spalten erkannt", table.column_count == 4, str(table.column_count))
    check("2 Datenzeilen erkannt", len(table.rows) == 2, str(len(table.rows)))
    check("Kopfzeile korrekt", table.header[0] == "Produkt" and table.header[3] == "Bemerkung")
    check("Trennzeile nicht als Datenzeile", all("---" not in c for r in table.rows for c in r))

    print("\n3) Spaltenausrichtung")
    check(
        "Ausrichtung aus Trennzeile übernommen",
        table.aligns == ["left", "right", "right", "left"],
        str(table.aligns),
    )
    auto = md.parse_table(["| Typ | Strom |", "| A | 24 A |", "| B | 32 A |"])
    check("numerische Spalte automatisch rechtsbündig", auto.aligns == ["left", "right"],
          str(auto.aligns))

    print("\n4) HTML-Ausgabe")
    out = md.table_to_html(table)
    check("scrollbarer Rahmen vorhanden", 'class="petra-table-wrap"' in out)
    check("echtes <table> statt Pipe-Text", "<table" in out and "<thead>" in out)
    check("Ausrichtung im HTML gesetzt", 'style="text-align:right"' in out)
    check("Inline-Fettschrift übernommen", "<strong>Standard</strong>" in out)
    check("keine rohen Pipes mehr im Zellinhalt", "|" not in out.replace("\\|", ""))

    print("\n5) Sicherheit / Robustheit")
    evil = md.parse_table(["| A | B |", "|---|---|", "| <script>x()</script> | ok |"])
    evil_html = md.table_to_html(evil)
    check("HTML in Zellen wird escaped", "<script>" not in evil_html and "&lt;script&gt;" in evil_html)
    no_delim = md.parse_table(["| Produkt | Preis |", "| WDU 2.5 | 1,20 € |"])
    check("Tabelle ohne Trennzeile wird toleriert", no_delim is not None and len(no_delim.rows) == 1)
    ragged = md.parse_table(["| A | B | C |", "|---|---|---|", "| 1 | 2 |"])
    check("fehlende Zellen werden aufgefüllt", ragged.rows[0] == ["1", "2", ""], str(ragged.rows))

    print("\n6) Normale Antworten bleiben unangetastet")
    plain = "Die WDU 2.5 ist eine Reihenklemme.\n\n- Punkt A\n- Punkt B\n"
    plain_blocks = md.split_blocks(plain)
    check("Fließtext/Liste bleibt ein Textblock",
          len(plain_blocks) == 1 and plain_blocks[0].kind == "text")
    check("Textinhalt unverändert", plain_blocks[0].text.strip() == plain.strip())

    citation = "Laut Datenblatt [1] gilt: a | b ist kein Tabellentrenner."
    check("Quellenangabe wird nicht als Tabelle interpretiert",
          not md.contains_table(citation))

    hr = "Text davor\n\n---\n\nText danach"
    check("horizontale Linie ist keine Tabelle", not md.contains_table(hr))

    code = "```\n| a | b |\n|---|---|\n| 1 | 2 |\n```"
    check("Tabelle im Codeblock bleibt Code", not md.contains_table(code))

    print("\n7) Rendering über Streamlit (Stub)")
    st.rendered.clear()
    md.render_markdown(TABLE_ANSWER)
    calls = st.rendered
    check("drei Render-Aufrufe (Text, Tabelle, Text)", len(calls) == 3, str(len(calls)))
    check("Tabelle mit unsafe_allow_html gerendert", calls[1][1].get("unsafe_allow_html") is True)
    check("Textblöcke ohne unsafe_allow_html", "unsafe_allow_html" not in calls[0][1])

    st.rendered.clear()
    md.render_markdown(plain)
    check("normale Antwort -> genau ein st.markdown", len(st.rendered) == 1)

    print("\n" + "=" * 62)
    if _FAILED:
        print(f"❌ {len(_FAILED)} Test(s) fehlgeschlagen: {', '.join(_FAILED)}")
        return False
    print(f"✅ Alle {_PASSED} Tests bestanden.")
    return True


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
