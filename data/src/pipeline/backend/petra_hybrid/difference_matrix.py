# ============================================================
# petra_hybrid/difference_matrix.py — Differenzmatrix (UC-03)
#
# Anforderungen:
#   UC-03 HS 2  Normalisierung von Einheiten, Schreibweisen, Formaten
#   UC-03 HS 3  dimensionsweise Gegenüberstellung
#   UC-03 HS 4  Klassifikation: identisch / ähnlich / signifikant unterschiedlich
#   UC-03 HS 5  Ausgabe als Matrix (Dict/JSON)
#   UC-03 A1    nicht normalisierbar -> Originalwert
#   UC-03 A2    Dimension nicht dokumentiert -> "Nicht dokumentiert"
#   UC-03       jeder Wert mit chunk_id verknüpft
#   UC-02 A2 / F-09  fehlende Werte als "N/A"
#
# Architektur: Kap. 4.4.1 (Q2), Kombination C (TAG als Datenbasis).
#
# Regelbasiert auf serialisierten Tabellenzeilen und "Attribut: Wert"-
# Mustern: Ein LLM könnte fehlende Werte erfinden (UC-03 verbietet
# Inferenz über Nicht-Vorhandenes). Nicht gefundene Dimensionen erscheinen
# als N/A. Die sprachliche Ausformulierung übernimmt das LLM (Schicht 4).
# ============================================================
from __future__ import annotations

import re
from dataclasses import dataclass, field

NICHT_VERFUEGBAR = "N/A"
NICHT_DOKUMENTIERT = "Nicht dokumentiert"

KLASSE_IDENTISCH = "identisch"
KLASSE_AEHNLICH = "ähnlich"
KLASSE_UNTERSCHIEDLICH = "signifikant unterschiedlich"
KLASSE_UNVOLLSTAENDIG = "nicht vergleichbar"

# Einheit (klein geschrieben) -> (Basiseinheit, Faktor).
_UNIT_FACTORS = {
    # Länge -> Meter
    "mm": ("m", 0.001), "cm": ("m", 0.01), "m": ("m", 1.0), "km": ("m", 1000.0),
    # Spannung -> Volt
    "mv": ("V", 0.001), "v": ("V", 1.0), "kv": ("V", 1000.0),
    # Strom -> Ampere
    "ma": ("A", 0.001), "a": ("A", 1.0),
    # Querschnitt -> mm²
    "mm²": ("mm²", 1.0), "mm2": ("mm²", 1.0),
    # Temperatur bleibt unangetastet (Offset-Umrechnung, kein Faktor)
    "°c": ("°C", 1.0),
    # Gewicht -> Gramm
    "g": ("g", 1.0), "kg": ("g", 1000.0),
}

_VALUE_RE = re.compile(
    r"(-?\d+(?:[.,]\d+)?)\s*(mm²|mm2|mm|cm|km|m|mV|kV|V|mA|A|°C|kg|g)\b",
    re.IGNORECASE,
)
_RANGE_RE = re.compile(
    r"(-?\d+(?:[.,]\d+)?)\s*(?:\.\.\.|\.\.|–|-|bis|to)\s*(-?\d+(?:[.,]\d+)?)\s*"
    r"(mm²|mm2|mm|cm|km|m|mV|kV|V|mA|A|°C|kg|g)?\b",
    re.IGNORECASE,
)
_IP_RE = re.compile(r"\bIP\s?(\d{2}[Kk]?)\b")

# Standarddimensionen des Use Case (UC-02) mit Suchbegriffen für den Textabschnitt.
STANDARD_DIMENSIONEN: dict[str, tuple[str, ...]] = {
    "Schutzart": ("schutzart", "protection", "ip-schutz", "degree of protection"),
    "Messbereich": ("messbereich", "measuring range", "erfassungsbereich", "range"),
    "Betriebsspannung": ("betriebsspannung", "versorgungsspannung", "operating voltage", "supply voltage"),
    "Schnittstelle": ("schnittstelle", "interface", "io-link", "kommunikation", "protocol"),
    "Umgebungstemperatur": ("umgebungstemperatur", "ambient temperature", "temperaturbereich"),
    "Anschlussquerschnitt": ("querschnitt", "leiterquerschnitt", "cross section", "conductor"),
    "Bemessungsstrom": ("bemessungsstrom", "nennstrom", "rated current", "strombelastbarkeit"),
    "Material": ("material", "gehäusematerial", "housing"),
}


@dataclass
class Zellwert:
    """Ein Wert einer Dimension für genau ein Produkt — mit Beleg."""

    roh: str = NICHT_VERFUEGBAR
    normalisiert: float | None = None
    einheit: str | None = None
    chunk_id: str | None = None      # UC-03: Verknüpfung Differenzwert -> Chunk
    dokument: str | None = None
    seite: int | None = None
    verfuegbar: bool = False

    def to_dict(self) -> dict:
        """JSON-serialisierbare Darstellung fuer n8n und FastAPI."""
        return {
            "wert": self.roh,
            "normalisiert": self.normalisiert,
            "einheit": self.einheit,
            "chunk_id": self.chunk_id,
            "dokument": self.dokument,
            "seite": self.seite,
            "verfuegbar": self.verfuegbar,
        }


@dataclass
class DimensionsVergleich:
    """Zeile der Differenzmatrix: eine Dimension über alle Produkte."""
    dimension: str
    werte: dict[str, Zellwert] = field(default_factory=dict)
    klassifikation: str = KLASSE_UNVOLLSTAENDIG
    hinweis: str = ""

    def to_dict(self) -> dict:
        """JSON-serialisierbare Darstellung fuer n8n und FastAPI."""
        return {
            "dimension": self.dimension,
            "werte": {p: z.to_dict() for p, z in self.werte.items()},
            "klassifikation": self.klassifikation,
            "hinweis": self.hinweis,
        }


def normalize_value(text: str) -> tuple[float | None, str | None, str]:
    """Extrahiert einen numerischen Wert samt normalisierter Einheit.

    Returns:
    (normalisierter Wert, Basiseinheit, Rohtext). Ist keine
    Normalisierung möglich, kommt (None, None, Rohtext) zurück — UC-03 A1:
    "bei Unsicherheit Originalwerte unverändert ausgeben".
    """
    raw = (text or "").strip()
    if not raw:
        return None, None, NICHT_VERFUEGBAR

    # Schutzart ist eine Klasse, keine Messgröße (IP67 nicht als 67 vergleichen).
    ip = _IP_RE.search(raw)
    if ip:
        return None, "IP", f"IP{ip.group(1).upper()}"

    # Bereiche zuerst: "0.2...2 m" darf nicht als Einzelwert "0.2" gelesen werden.
    span = _RANGE_RE.search(raw)
    if span and span.group(3):
        lo = float(span.group(1).replace(",", "."))
        hi = float(span.group(2).replace(",", "."))
        base, factor = _UNIT_FACTORS.get(span.group(3).lower(), (span.group(3), 1.0))
        return None, base, f"{lo * factor:g}–{hi * factor:g} {base}"

    single = _VALUE_RE.search(raw)
    if single:
        value = float(single.group(1).replace(",", "."))
        base, factor = _UNIT_FACTORS.get(single.group(2).lower(), (single.group(2), 1.0))
        return value * factor, base, f"{value * factor:g} {base}"

    return None, None, raw


def _extract_dimension(chunks: list[dict], suchbegriffe: tuple[str, ...]) -> Zellwert:
    """Sucht den Wert einer Dimension in den Chunks eines Produkts.

    Tabellen-Chunks werden zuerst durchsucht (eindeutige Zuordnung
    Attribut). Erster Treffer gewinnt.
    """
    sortiert = sorted(
        chunks,
        key=lambda c: 0 if (c.get("metadata") or {}).get("source_type") == "table" else 1,
    )

    for chunk in sortiert:
        text = chunk.get("text") or ""
        lower = text.lower()
        for begriff in suchbegriffe:
            pos = lower.find(begriff)
            if pos < 0:
                continue
            # 160 Zeichen ab dem Begriff; der Wert folgt nach ":" oder "|".
            fenster = text[pos: pos + 160]
            nach_trenner = re.split(r"[:|]\s*", fenster, maxsplit=1)
            kandidat = nach_trenner[1] if len(nach_trenner) > 1 else fenster
            wert, einheit, roh = normalize_value(kandidat)
            if roh == NICHT_VERFUEGBAR:
                continue
            meta = chunk.get("metadata") or {}
            return Zellwert(
                roh=roh, normalisiert=wert, einheit=einheit,
                chunk_id=chunk.get("chunk_id"),
                dokument=meta.get("document") or meta.get("dateiname"),
                seite=meta.get("page") or meta.get("seite"),
                verfuegbar=True,
            )
    return Zellwert()


def _klassifiziere(werte: list[Zellwert]) -> tuple[str, str]:
    """Klassifiziert eine Dimension (UC-03 HS 4).

    Returns:
        (Klasse, Hinweistext)
    """
    verfuegbar = [w for w in werte if w.verfuegbar]
    if len(verfuegbar) < 2:
        fehlend = len(werte) - len(verfuegbar)
        return KLASSE_UNVOLLSTAENDIG, (
            f"Für {fehlend} von {len(werte)} Produkten ist kein belegter Wert vorhanden "
            f"(als {NICHT_VERFUEGBAR} ausgewiesen)."
        )

    rohwerte = {w.roh for w in verfuegbar}
    if len(rohwerte) == 1:
        return KLASSE_IDENTISCH, "Identischer Wert in allen Quellen."

    numerisch = [w for w in verfuegbar if w.normalisiert is not None]
    if len(numerisch) == len(verfuegbar) and len({w.einheit for w in numerisch}) == 1:
        werte_num = [w.normalisiert for w in numerisch]
        lo, hi = min(werte_num), max(werte_num)
        if lo == hi:
            return KLASSE_IDENTISCH, "Identischer Wert nach Einheitennormalisierung."
        # Schwelle 10 % relative Abweichung (Lesehilfe; die Absolutwerte
        # werden immer mit ausgegeben).
        abweichung = abs(hi - lo) / max(abs(hi), 1e-9)
        if abweichung <= 0.10:
            return KLASSE_AEHNLICH, f"Relative Abweichung {abweichung:.1%} (Schwelle 10 %)."
        return KLASSE_UNTERSCHIEDLICH, f"Relative Abweichung {abweichung:.1%} (Schwelle 10 %)."

    return KLASSE_UNTERSCHIEDLICH, "Unterschiedliche Angaben; numerischer Vergleich nicht möglich."


def build_difference_matrix(
    produkt_chunks: dict[str, list[dict]],
    dimensionen: list[str] | None = None,
) -> dict:
    """Erzeugt die Differenzmatrix (UC-03).

    Args:
        produkt_chunks: Produktbezeichnung -> Chunks (aus Dual-Retrieval, UC-02).
        dimensionen: Angefragte Dimensionen. Ohne Angabe werden die
            Standarddimensionen geprüft; nur belegte werden aufgenommen.

    Returns:
        JSON-serialisierbare Matrix mit Werten, Belegen (chunk_id,
        Dokument, Seite), Klassifikation und Belegquote.
    """
    produkte = list(produkt_chunks)
    if len(produkte) < 2:
        return {
            "produkte": produkte,
            "dimensionen": [],
            "vergleichbar": False,
            # UC-02 A1: nur ein Produkt in der Datenbank vorhanden
            "hinweis": (
                "Für einen Vergleich werden mindestens zwei Produkte mit "
                "Dokumentenbasis benötigt."
            ),
        }

    if dimensionen:
        zu_pruefen = {
            d: STANDARD_DIMENSIONEN.get(d, (d.lower(),)) for d in dimensionen
        }
    else:
        zu_pruefen = STANDARD_DIMENSIONEN

    ergebnisse: list[DimensionsVergleich] = []
    for dimension, begriffe in zu_pruefen.items():
        vergleich = DimensionsVergleich(dimension=dimension)
        for produkt in produkte:
            vergleich.werte[produkt] = _extract_dimension(produkt_chunks[produkt], begriffe)

        belegt = [w for w in vergleich.werte.values() if w.verfuegbar]
        if not belegt:
            if dimensionen:
                # UC-03 A2: explizit angefragte Dimension, nirgends dokumentiert
                vergleich.klassifikation = KLASSE_UNVOLLSTAENDIG
                vergleich.hinweis = NICHT_DOKUMENTIERT
                ergebnisse.append(vergleich)
            # Ohne explizite Anfrage wird eine leere Standarddimension
            # weggelassen — eine Tabelle voller N/A-Zeilen ist keine Hilfe.
            continue

        vergleich.klassifikation, vergleich.hinweis = _klassifiziere(
            list(vergleich.werte.values())
        )
        ergebnisse.append(vergleich)

    return {
        "produkte": produkte,
        "dimensionen": [e.to_dict() for e in ergebnisse],
        "vergleichbar": bool(ergebnisse),
        "anzahl_dimensionen": len(ergebnisse),
        "belegquote": _belegquote(ergebnisse, len(produkte)),
        "hinweis": "" if ergebnisse else (
            "Keine der geprüften Vergleichsdimensionen ist in den vorliegenden "
            "Dokumenten belegt."
        ),
    }


def _belegquote(ergebnisse: list[DimensionsVergleich], anzahl_produkte: int) -> float:
    """Anteil belegter Zellen (Messgröße F-09: Quellennachweis für ≥ 90 % der Dimensionen)."""
    zellen = len(ergebnisse) * anzahl_produkte
    if not zellen:
        return 0.0
    belegt = sum(
        1 for e in ergebnisse for w in e.werte.values() if w.verfuegbar
    )
    return round(belegt / zellen, 3)


def to_markdown(matrix: dict) -> str:
    """Rendert die Matrix als Markdown-Tabelle mit Quellenangaben für den Prompt.

    Das LLM ordnet die belegten Werte ein, statt sie aus Fließtext zu
    extrahieren (Halluzinationsschutz, F-13).
    """
    produkte = matrix.get("produkte", [])
    dimensionen = matrix.get("dimensionen", [])
    if not dimensionen:
        return matrix.get("hinweis", "Keine vergleichbaren Dimensionen gefunden.")

    kopf = "| Dimension | " + " | ".join(produkte) + " | Einordnung |"
    trenner = "|" + "---|" * (len(produkte) + 2)
    zeilen = [kopf, trenner]
    for eintrag in dimensionen:
        zellen = []
        for produkt in produkte:
            wert = eintrag["werte"].get(produkt, {})
            text = wert.get("wert", NICHT_VERFUEGBAR)
            if wert.get("verfuegbar") and wert.get("dokument"):
                text += f" [Quelle: {wert['dokument']}, S. {wert.get('seite', '?')}]"
            zellen.append(text)
        zeilen.append(
            f"| {eintrag['dimension']} | " + " | ".join(zellen) + f" | {eintrag['klassifikation']} |"
        )
    return "\n".join(zeilen)
