# ============================================================
# petra_hybrid/price_rows.py — Preiszeilen-Parser (v3.4-patch)
#
# ANFORDERUNGSBEZUG: UC-04 Hauptszenario 3 ("strukturierte Preisangabe
# aus Chunk"), F-10 (Preisextraktion >= 85 %), UC-04 "Erfinde keine
# Preise", Zyklus 1 (VektordatenbankTool, lokal zuerst).
#
# ============================================================
from __future__ import annotations

import re

ROW_RE = re.compile(
    r"(?<!\d)(?P<art>\d{10})\s+"
    r"(?P<name>.{1,60}?)\s+"
    r"(?P<mbm>\d{1,3}(?:\.\d{3})*)\s+"
    r"(?P<pg>[A-Z]\d{3})\s+"
    r"(?P<preis>\d{1,3}(?:\.\d{3})*,\d{2})\s*EUR"
    r"(?:\s+(?P<pe>\d{1,3}(?:\.\d{3})*)\s*(?P<einh>[A-Z]{1,3})\b)?"
)

EINHEIT = {"PCE": "Stück", "ST": "Stück", "M": "Meter", "MTR": "Meter", "KG": "kg", "SET": "Set"}


def norm_name(s: str) -> str:
    return re.sub(r"\s+", "", (s or "").upper().replace(",", "."))


def _num(s: str) -> float:
    return float(s.replace(".", "").replace(",", "."))


def _de(x: float, stellen: int = 2) -> str:
    return f"{x:,.{stellen}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def parse_rows(text: str) -> list[dict]:
    rows = []
    for m in ROW_RE.finditer(text or ""):
        d = m.groupdict()
        d["name"] = re.sub(r"\s+", " ", d["name"]).strip()
        rows.append(d)
    return rows


def row_to_text(r: dict) -> str:
    """
    Kontexttext für die Preisangabe. Der Stückpreis wird OHNE
    Währungszeichen geschrieben, damit die nachgelagerte Regex-Extraktion
    (price_extraction.py) ihn nicht als zweiten, "widersprüchlichen"
    Preis liest.
    """
    preis = _num(r["preis"])
    pe = int(_num(r["pe"])) if r.get("pe") else None
    einh = EINHEIT.get((r.get("einh") or "").upper(), r.get("einh") or "Stück")
    if pe:
        einheit_txt = f"je {pe} {einh}"
        if pe > 1:
            einheit_txt += f" (entspricht {_de(preis / pe, 3)} pro {einh if einh != 'Stück' else 'Stück'})"
    else:
        einheit_txt = "(Preiseinheit in der Zeile nicht erkennbar)"
    return (f"Preiszeile Artikelnummer {r['art']} | {r['name']}: "
            f"Listenpreis {r['preis']} EUR {einheit_txt}; "
            f"Verpackungseinheit (VPE) {r['mbm']} {einh}, Preisgruppe {r['pg']}.")


def find_price_rows(index, artnrs: list[str] | None = None, name: str = "",
                    filename_substring: str = "preisliste") -> list[dict]:
    """
    Liefert Preiszeilen-Chunks für GENAU das gesuchte Produkt.

    1. Artikelnummer(n) aus dem Produkt-Lexikon -> Zeile mit dieser Nummer.
    2. Sonst: Zeilen, deren normalisierte Bezeichnung GENAU dem Namen
       entspricht (keine Varianten wie "... AGSNO" oder "... BL").
    Leere Liste = Produkt steht nicht (erkennbar) in der Preisliste.
    """
    artnrs = [a for a in (artnrs or []) if a]
    ziel_name = norm_name(name)
    gesucht = artnrs or ([name] if len(ziel_name) >= 4 else [])
    out: list[dict] = []
    seen: set[str] = set()
    for needle in gesucht:
        try:
            hits = index.search_by_filename_substring(
                filename_substring=filename_substring, text_substring=needle, limit=40)
        except TypeError:  # ältere Signatur ohne limit
            hits = index.search_by_filename_substring(
                filename_substring=filename_substring, text_substring=needle)
        for h in hits:
            for r in parse_rows(h.get("text", "")):
                passt = (r["art"] in artnrs) if artnrs else (norm_name(r["name"]) == ziel_name)
                if not passt or r["art"] in seen:
                    continue
                seen.add(r["art"])
                meta = dict(h.get("metadata") or {})
                meta.update({
                    "preiszeile": True,
                    "artikelnummer": r["art"],
                    "produkt": r["name"],
                    "preiseinheit": f"{r.get('pe') or '?'} {r.get('einh') or ''}".strip(),
                    "source_type": "table",
                })
                out.append({
                    "chunk_id": f"{h.get('chunk_id', 'preisliste')}#{r['art']}",
                    "text": row_to_text(r),
                    "metadata": meta,
                    "score": None,
                })
    return out
