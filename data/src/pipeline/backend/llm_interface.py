"""llm_interface — Fassade der Generierungs- und Validierungsschicht (NF-10).

Prompt-Komposition [4a], Antwortschema, Quellenliste [5b] und
Sprachbehandlung (NF-08). Die LLM-Aufrufe selbst laufen über n8n-Nodes
gegen Ollama (Kap. 4.7.1); dieses Modul enthält Vor- und Nachbereitung
und ist ohne laufendes Modell testbar.
"""
from backend.petra_hybrid.language import (
    DEFAULT_LANGUAGE, SUPPORTED, detect_language, unsupported_language_message,
)
from backend.petra_hybrid.prompt_builder import (
    MAX_CONTEXT_TOKENS, SYSTEM_INSTRUCTION, build_context, build_prompt,
    estimate_tokens,
)
from backend.petra_hybrid.response_schema import (
    Source, StructuredAnswer, build_no_context_answer, build_sources_from_chunks,
)

__all__ = [
    "build_prompt", "build_context", "estimate_tokens", "SYSTEM_INSTRUCTION",
    "MAX_CONTEXT_TOKENS", "StructuredAnswer", "Source",
    "build_sources_from_chunks", "build_no_context_answer", "detect_language",
    "unsupported_language_message", "SUPPORTED", "DEFAULT_LANGUAGE",
    "FAITHFULNESS_THRESHOLD", "flag_unverified",
]

# F-13: unterhalb dieses Werts wird ein Warnhinweis angezeigt.
FAITHFULNESS_THRESHOLD = 0.75


def flag_unverified(faithfulness_score: float | None) -> dict:
    """Kennzeichnet nicht belegte Antworten (F-13, UC-08 A3).

    Ein nicht bewertbarer Score fuehrt bewusst NICHT zu ``verified=False``:
    "nicht bewertet" und "nicht belegt" sind unterschiedliche Aussagen,
    und ein falscher Warnhinweis auf einer korrekten Antwort kostet
    Vertrauen.

    Returns:
        Dict mit ``verified``, ``warnhinweis`` und ``hinweis``.
    """
    if faithfulness_score is None:
        return {
            "verified": None,
            "warnhinweis": None,
            "hinweis": "Faithfulness konnte nicht bewertet werden.",
        }
    verified = faithfulness_score >= FAITHFULNESS_THRESHOLD
    return {
        "verified": verified,
        "warnhinweis": None if verified else "Nicht verifiziert – bitte Dokument konsultieren",
        "hinweis": None,
    }
