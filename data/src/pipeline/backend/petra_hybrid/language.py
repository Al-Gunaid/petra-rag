# ============================================================
# petra_hybrid/language.py — Spracherkennung der Anfrage
#
# Anforderungen:
#   NF-08  Antwortsprache = Anfragesprache; Deutsch und Englisch.
#   UC-01 A5  nicht unterstützte Sprache -> Fehlermeldung.
#   F-08   Sprachinstruktion als {language}-Variable im Prompt.
#
# Regelbasiert (Funktionswörter und Zeichenstatistik): Ein zusätzlicher
# LLM-Aufruf im kritischen Pfad kostet 0,8–1,5 s des NF-03-Budgets.
# Bibliotheken wie langdetect entfallen, da keine zusätzlichen Pakete
# benötigt werden sollen (NF-02, Docker-Build ohne Netz).
#
# Bei Unsicherheit wird Deutsch angenommen: Der Bestand ist
# deutschsprachig, und kurze Anfragen ("Schutzart WDU 2.5?") sollen keine
# Fehlermeldung auslösen.
# ============================================================
from __future__ import annotations

import re

SUPPORTED = ("Deutsch", "Englisch")
DEFAULT_LANGUAGE = "Deutsch"

# Funktionswörter technischer Kurzanfragen. Fachbegriffe ("Sensor",
# "IP67") sind in beiden Sprachen gleich und daher nicht enthalten.
_DE_MARKERS = {
    "welche", "welcher", "welches", "wie", "was", "ist", "hat", "der", "die", "das",
    "und", "oder", "für", "mit", "von", "kostet", "vergleiche", "unterschied",
    "eignet", "sich", "bei", "kann", "nicht", "ein", "eine", "einen", "zwischen",
}
_EN_MARKERS = {
    "which", "what", "how", "is", "are", "has", "have", "the", "and", "or",
    "for", "with", "of", "cost", "costs", "compare", "difference", "between",
    "suitable", "can", "not", "a", "an", "does", "do", "price",
}

# Umlaute/ß sprechen für Deutsch; ihr Fehlen ist kein Hinweis auf Englisch.
_GERMAN_CHARS = re.compile(r"[äöüßÄÖÜ]")

# Zeichen außerhalb des lateinischen Bereichs deuten auf eine nicht
# unterstützte Sprache hin (kyrillisch, CJK, arabisch, griechisch).
_NON_LATIN = re.compile(r"[\u0400-\u04FF\u0590-\u05FF\u0600-\u06FF\u0370-\u03FF\u4E00-\u9FFF\u3040-\u30FF]")

_TOKEN_RE = re.compile(r"[a-zA-ZäöüßÄÖÜ]+")


def detect_language(query: str) -> dict:
    """Erkennt die Anfragesprache.

    Returns:
        dict mit ``language`` (aus SUPPORTED), ``supported`` (bool),
        ``confidence`` [0,1] und ``reason``. Bei ``supported = False``
        löst der Workflow UC-01 A5 aus: Hinweis statt Antwortversuch.
    """
    text = (query or "").strip()
    if not text:
        return {
            "language": DEFAULT_LANGUAGE, "supported": True, "confidence": 0.0,
            "reason": "Leere Anfrage — Standardsprache angenommen.",
        }

    non_latin = len(_NON_LATIN.findall(text))
    if non_latin >= 3 and non_latin / max(len(text), 1) > 0.15:
        return {
            "language": "unbekannt", "supported": False, "confidence": 0.9,
            "reason": (
                "Anfrage enthält überwiegend nicht-lateinische Schriftzeichen. "
                "Unterstützt werden Deutsch und Englisch (NF-08)."
            ),
        }

    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    de_hits = sum(1 for t in tokens if t in _DE_MARKERS)
    en_hits = sum(1 for t in tokens if t in _EN_MARKERS)

    # Tokens, die in beiden Listen stehen ("is", "or"), zählen doppelt und
    # heben sich damit auf — genau das gewünschte Verhalten.
    if _GERMAN_CHARS.search(text):
        de_hits += 2

    total = de_hits + en_hits
    if total == 0:
        return {
            "language": DEFAULT_LANGUAGE, "supported": True, "confidence": 0.3,
            "reason": (
                "Keine sprachspezifischen Funktionswörter gefunden (typisch für "
                "reine Produktcode-Anfragen). Standardsprache angenommen."
            ),
        }

    if de_hits >= en_hits:
        return {
            "language": "Deutsch", "supported": True,
            "confidence": round(de_hits / total, 2),
            "reason": f"{de_hits} deutsche vs. {en_hits} englische Marker.",
        }
    return {
        "language": "Englisch", "supported": True,
        "confidence": round(en_hits / total, 2),
        "reason": f"{en_hits} englische vs. {de_hits} deutsche Marker.",
    }


def unsupported_language_message() -> str:
    """Meldung für UC-01 A5."""
    return (
        "Diese Anfrage scheint in einer nicht unterstützten Sprache verfasst zu sein. "
        "Das System beantwortet Anfragen auf Deutsch und Englisch. "
        "Bitte formulieren Sie die Frage in einer dieser Sprachen."
    )
