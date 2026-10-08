# ============================================================
# petra_hybrid/price_extraction.py — Preisextraktion (UC-04)
#
# Anforderungen:
#   UC-04 HS 3     strukturierte Preisangabe aus Chunks
#   UC-04 HS 4     Ausgabe mit Quelldokument und Dokumentdatum
#   UC-04 A3/F-10  Dokumentdatum > 12 Monate -> Warnhinweis
#   UC-04 A2       widersprüchliche Preise -> alle Quellen ausweisen
#   F-10           Genauigkeit der Preisextraktion >= 85 %
#   UC-04          keine erfundenen Preise; fehlende Daten benennen
#
# Regelbasiert statt LLM: Eine Regex liefert den Betrag unverändert oder
# gar nicht. Ein LLM kann Beträge verfälschen (z. B. Dezimalstelle).
# Die sprachliche Einbettung übernimmt das LLM in Schicht 4.
# ============================================================
from __future__ import annotations

import json as _json
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime

# Grenze aus UC-04 A3 (Wissensaktualitäts-Grenze).
STALE_MONTHS = 12

_PRICE_PATTERNS = (
    # Betrag vor Währung: 12,50 EUR | 12.50 EUR | 1.234,56 EUR | 1,234.56 EUR
    # Tausendertrenner Punkt, Komma oder Leerzeichen; _parse_amount bestimmt
    # das Dezimaltrennzeichen. \b nur nach EUR/Euro: "€" ist kein
    # Wortzeichen, "\b" nach "€ " würde nie greifen.
    re.compile(r"(\d{1,7}(?:[., ]\d{3})*(?:[.,]\d{1,2})?)\s*(EUR\b|Euro\b|€)", re.IGNORECASE),
    # EUR 12,50 | € 12.50
    re.compile(r"(?:EUR\b|Euro\b|€)\s*(\d{1,7}(?:[., ]\d{3})*(?:[.,]\d{1,2})?)\b", re.IGNORECASE),
)

_DATE_PATTERNS = (
    re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b"),      # 2022-03-15
    re.compile(r"\b(\d{2})\.(\d{2})\.(20\d{2})\b"),    # 15.03.2022
    re.compile(r"\b(20\d{2})-(\d{2})\b"),              # 2022-03
)

# Mindestens eines dieser Wörter muss im Chunk vorkommen, damit Zahlen
# als Preis gelten (verhindert z. B. "24 V" als Betrag).
_PRICE_CONTEXT = (
    "preis", "listenpreis", "kosten", "netto", "brutto", "vpe", "stückpreis",
    "price", "list price", "eur", "€", "je stück", "pro stück",
)


@dataclass
class Preisangabe:
    """Belegte Preisangabe mit Quelle, Dokumentdatum und Altersbewertung."""
    betrag: float
    waehrung: str = "EUR"
    chunk_id: str | None = None
    dokument: str | None = None
    seite: int | None = None
    dokumentdatum: str | None = None
    alter_monate: int | None = None
    veraltet: bool = False
    warnhinweis: str | None = None
    textauszug: str = ""

    def to_dict(self) -> dict:
        """JSON-serialisierbare Darstellung fuer n8n und FastAPI."""
        return {
            "betrag": self.betrag,
            "waehrung": self.waehrung,
            "chunk_id": self.chunk_id,
            "dokument": self.dokument,
            "seite": self.seite,
            "dokumentdatum": self.dokumentdatum,
            "alter_monate": self.alter_monate,
            "veraltet": self.veraltet,
            "warnhinweis": self.warnhinweis,
            "textauszug": self.textauszug,
        }


@dataclass
class PreisErgebnis:
    """Gesamtergebnis der Preisextraktion inkl. Widerspruchskennzeichnung."""
    gefunden: bool
    preise: list[Preisangabe] = field(default_factory=list)
    widerspruechlich: bool = False
    web_recherche_empfohlen: bool = False
    meldung: str = ""

    def to_dict(self) -> dict:
        """JSON-serialisierbare Darstellung fuer n8n und FastAPI."""
        return {
            "preis_gefunden": self.gefunden,
            "preise": [p.to_dict() for p in self.preise],
            "widerspruechlich": self.widerspruechlich,
            "web_recherche_empfohlen": self.web_recherche_empfohlen,
            "preis_meldung": self.meldung,
        }


def _parse_amount(raw: str) -> float | None:
    """Wandelt deutsche und englische Zahlformate in einen Float (2 Nachkommastellen).

    "1.234" ist im deutschen Format 1234, im englischen 1.234.

    Returns:
        Betrag oder None, wenn nicht interpretierbar.
    """
    text = raw.strip().replace(" ", "")
    if "," in text and "." in text:
        # Das zuletzt stehende Zeichen ist das Dezimaltrennzeichen.
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        # Ein bis zwei Nachkommastellen -> Dezimalkomma, sonst Tausendertrenner.
        text = text.replace(",", ".") if re.search(r",\d{1,2}$", text) else text.replace(",", "")
    elif re.search(r"\.\d{3}$", text):
        text = text.replace(".", "")
    try:
        return round(float(text), 2)
    except ValueError:
        return None


def extract_document_date(chunk: dict) -> str | None:
    """Dokumentdatum aus Metadaten, ersatzweise aus dem Text.

    Metadaten haben Vorrang, da ein Datum im Text sich auf Normen oder
    Produktionsjahre beziehen kann. `abrufdatum` wird von agent.py für
    Web-Quellen gesetzt.
    """
    meta = chunk.get("metadata") or {}
    for key in ("date", "datum", "document_date", "erstellungsdatum", "abrufdatum", "dokumentdatum"):
        if meta.get(key):
            return str(meta[key])[:10]

    # v3.4-patch P11: Stand undatierter Dokumente (z. B. Preisliste) optional
    # aus /data/dokument_stand.json, Format {"<Dateiname>": "JJJJ-MM-TT"}.
    _doc = str(meta.get("document") or meta.get("filename") or meta.get("source") or "")
    if _doc:
        try:
            with open(os.getenv("PETRA_DOKUMENT_STAND", "/data/dokument_stand.json"), encoding="utf-8") as _fh:
                _stand = (_json.load(_fh) or {}).get(os.path.basename(_doc))
            if _stand:
                return str(_stand)[:10]
        except (OSError, ValueError):
            pass

    text = chunk.get("text") or ""
    for pattern in _DATE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        groups = match.groups()
        if len(groups) == 3 and len(groups[0]) == 4:
            return f"{groups[0]}-{groups[1]}-{groups[2]}"
        if len(groups) == 3:
            return f"{groups[2]}-{groups[1]}-{groups[0]}"
        return f"{groups[0]}-{groups[1]}-01"
    return None


def assess_age(document_date: str | None, heute: date | None = None) -> tuple[int | None, bool, str | None]:
    """Bewertet das Alter einer Preisangabe (UC-04 A3).

    Fehlendes oder ungültiges Datum gilt als veraltet.

    Returns:
        (alter_monate, veraltet, warnhinweis)
    """
    heute = heute or date.today()
    if not document_date:
        return None, True, (
            "Listenpreis aus einem undatierten Dokument: Die Aktualität ist nicht "
            "belegbar. Aktuelle Händlerpreise siehe Web-Recherche."  # v3.4-patch P11
        )
    try:
        text = document_date[:10]
        parsed = datetime.fromisoformat(text if len(text) == 10 else f"{text}-01").date()
    except ValueError:
        return None, True, f"Dokumentdatum '{document_date}' ist nicht interpretierbar."

    monate = (heute.year - parsed.year) * 12 + (heute.month - parsed.month)
    if monate > STALE_MONTHS:
        return monate, True, (
            f"Preisangabe stammt aus einem älteren Dokument (Stand {parsed.isoformat()}, "
            f"{monate} Monate alt). Der aktuelle Preis kann abweichen."
        )
    return monate, False, None


def extract_prices(chunks: list[dict], heute: date | None = None) -> PreisErgebnis:
    """
    Extrahiert strukturierte Preisangaben aus Retrieval-Chunks (UC-04).

    Returns:
        PreisErgebnis; Preise sortiert nach (veraltet, betrag).
        `web_recherche_empfohlen` ist True, wenn nichts gefunden wurde oder
        alle Preise veraltet sind.
    """
    preise: list[Preisangabe] = []

    for chunk in chunks or []:
        text = chunk.get("text") or ""
        lower = text.lower()
        if not any(marker in lower for marker in _PRICE_CONTEXT):
            continue

        meta = chunk.get("metadata") or {}
        dokumentdatum = extract_document_date(chunk)
        monate, veraltet, warnung = assess_age(dokumentdatum, heute)

        for pattern in _PRICE_PATTERNS:
            for match in pattern.finditer(text):
                betrag = _parse_amount(match.group(1))
                if betrag is None or betrag <= 0:
                    continue
                start = max(0, match.start() - 60)
                preise.append(
                    Preisangabe(
                        betrag=betrag,
                        waehrung="EUR",
                        chunk_id=chunk.get("chunk_id"),
                        dokument=(meta.get("document") or meta.get("dateiname") or meta.get("filename") or meta.get("source")),
                        seite=meta.get("page") or meta.get("seite"),
                        dokumentdatum=dokumentdatum,
                        alter_monate=monate,
                        veraltet=veraltet,
                        warnhinweis=warnung,
                        textauszug=text[start: match.end() + 40].strip(),
                    )
                )
            if preise:
                break  # zweites Muster nur, wenn das erste nichts gefunden hat

    if not preise:
        # Fehlende Daten explizit melden, keinen Preis schätzen (UC-04 A1)
        return PreisErgebnis(
            gefunden=False,
            web_recherche_empfohlen=True,
            meldung=(
                "In der lokalen Dokumentenbasis ist für diese Anfrage keine Preisangabe "
                "enthalten. Es wird kein Preis geschätzt."
            ),
        )

    betraege = {p.betrag for p in preise}
    widerspruechlich = len(betraege) > 1
    veraltet_alle = all(p.veraltet for p in preise)

    if widerspruechlich:
        # UC-04 A2: alle Quellen ausweisen, Entscheidung beim Benutzer
        meldung = (
            f"Es wurden {len(betraege)} unterschiedliche Preisangaben gefunden "
            f"({', '.join(f'{b:.2f} EUR' for b in sorted(betraege))}). Alle Quellen sind "
            "ausgewiesen; bitte prüfen Sie, welche für Ihren Fall maßgeblich ist."
        )
    elif veraltet_alle:
        meldung = preise[0].warnhinweis or "Preisangabe möglicherweise veraltet."
    else:
        meldung = "Preisangabe aus aktueller lokaler Dokumentenbasis."

    return PreisErgebnis(
        gefunden=True,
        preise=sorted(preise, key=lambda p: (p.veraltet, p.betrag)),
        widerspruechlich=widerspruechlich,
        web_recherche_empfohlen=veraltet_alle,
        meldung=meldung,
    )