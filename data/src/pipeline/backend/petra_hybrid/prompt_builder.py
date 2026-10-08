# ============================================================
# petra_hybrid/prompt_builder.py — Prompt-Komposition
#
# Token-Budget 7.168 ([4a]); bei Überschreitung werden Chunks
# nach Rang verworfen. Ohne Budget überschreiten 15 Chunks als 512 Token das
# Kontextfenster von Llama 3.1 8B (8.192 Token), und Ollama kürzt den
# Prompt stillschweigend. Zu langer Kontext verschlechtert zudem die
# Nutzung mittlerer Chunks ("Lost in the Middle", Liu et al. 2023).
#
# ============================================================
from __future__ import annotations

# Konservative Schätzung für deutschsprachigen Fachtext mit Llama-Tokenizer:
# ~3.3 Zeichen pro Token. Bewusst niedriger angesetzt als der oft zitierte
# Wert von 4, weil Produktcodes und Einheiten überdurchschnittlich viele
# Token je Zeichen erzeugen. Eine Unterschätzung des Budgets kostet einen
# Chunk, eine Überschätzung kostet die Antwort.
CHARS_PER_TOKEN = 3.3

MAX_CONTEXT_TOKENS = 7168        # [4a]
RESERVED_OUTPUT_TOKENS = 512     # für die generierte Antwort
SYSTEM_PROMPT_TOKEN_ESTIMATE = 320

SYSTEM_INSTRUCTION = """Du bist ein technischer Produktassistent für industrielle Automatisierungskomponenten.

Regeln:
- Antworte AUSSCHLIESSLICH auf Basis des folgenden Kontexts.
- Erfinde KEINE technischen Daten, Produktnummern, Einheiten oder Hersteller.
- Übernimm Produktnummern, Messwerte und Einheiten EXAKT aus dem Kontext.
- Nenne den Hersteller genau so, wie er in der jeweiligen Quellenangabe steht. Ordne Produkte niemals einem Hersteller zu, der nicht in der Quelle genannt ist.
- Zitiere für jede fachliche Aussage die Quelle im Format [Quelle: Dokument, S. X].
- Wenn die Information nicht im Kontext enthalten ist, antworte exakt: "Ich weiß es nicht."
"""
# Zusätzliche Anweisung je Anfrageklasse (classification.ROUTING_MATRIX).
CLASS_INSTRUCTIONS = {
    "produktvergleich": (
        "- Diese Anfrage ist ein Produktvergleich. Stelle die Produkte dimensionsweise "
        "in einer Markdown-Tabelle gegenüber (eine Zeile je Attribut, eine Spalte je "
        "Produkt). Attribute ohne Beleg im Kontext markierst du mit 'nicht belegt'.\n"
    ),
    "produktspezifikation": (
        "- Diese Anfrage fragt einen konkreten Attributwert ab. Antworte knapp und "
        "nenne den Wert samt Einheit, dann die Quelle.\n"
    ),
    "produktsuche": (
        "- Diese Anfrage sucht passende Produkte. Nenne nur Produkte, die im Kontext "
        "vorkommen, und begründe die Eignung anhand belegter Attribute.\n"
    ),
    "allgemeine_technische_frage": (
        "- Diese Anfrage ist begrifflich/konzeptuell. Erkläre knapp auf Basis des "
        "Kontexts, ohne Produktempfehlung.\n"
    ),
}


def estimate_tokens(text: str) -> int:
    """Konservative Token-Schaetzung fuer das Kontextbudget (Kap. 4.2 [4a])."""
    return int(len(text or "") / CHARS_PER_TOKEN) + 1


def format_chunk(chunk: dict) -> str:
    """Formatiert einen Chunk mit Quellenannotation fuer den Prompt."""
    meta = chunk.get("metadata") or {}
    document = meta.get("document") or meta.get("dateiname") or meta.get("filename") or meta.get("source") or "unbekannt"
    page = meta.get("page") or meta.get("seite") or "?"
    manufacturer = meta.get("manufacturer") or meta.get("hersteller")
    source_type = meta.get("source_type", "text")

    header = f"[Quelle: {document}, S. {page}"
    if manufacturer:
        header += f", Hersteller: {manufacturer}"
    if source_type == "table":
        header += ", Typ: Tabelle"
    header += "]"
    return f"{header}\n{(chunk.get('text') or '').strip()}"


def build_context(
    chunks: list[dict],
    max_tokens: int = MAX_CONTEXT_TOKENS,
    reserved_output: int = RESERVED_OUTPUT_TOKENS,
    query: str = "",
) -> tuple[str, list[dict], dict]:
    """Baut den Kontextblock innerhalb des Token-Budgets.

    Tabellen-Chunks (TAG) stehen vorn, da ihre Serialisierung
    Attributwerte eindeutig Produkten zuordnet. Innerhalb der Gruppen bleibt
    die Eingangsreihenfolge (Reranker bzw. RRF) erhalten. Das Budget beträgt
    mindestens 512 Token.

    Returns:
        (Kontexttext, verwendete Chunks, Budget-Info)
    """
    budget = max_tokens - reserved_output - SYSTEM_PROMPT_TOKEN_ESTIMATE - estimate_tokens(query)
    budget = max(budget, 512)

    ordered = sorted(
        chunks or [],
        key=lambda c: 0 if (c.get("metadata") or {}).get("source_type") == "table" else 1,
    )

    used: list[dict] = []
    parts: list[str] = []
    consumed = 0
    dropped = 0

    for chunk in ordered:
        block = format_chunk(chunk)
        cost = estimate_tokens(block)
        if consumed + cost > budget:
            if not used:
                # Erster Chunk zu lang: kürzen statt leeren Kontext liefern.
                allowed_chars = int(budget * CHARS_PER_TOKEN)
                parts.append(block[:allowed_chars])
                used.append(chunk)
                consumed = budget
            else:
                dropped += 1
            continue
        parts.append(block)
        used.append(chunk)
        consumed += cost

    info = {
        "context_tokens_estimate": consumed,
        "context_token_budget": budget,
        "chunks_used": len(used),
        "chunks_dropped": dropped,
        "budget_exceeded": dropped > 0,
    }
    return "\n\n".join(parts), used, info


def build_prompt(
    query: str,
    chunks: list[dict],
    query_class: str = "",
    language: str = "Deutsch",
    max_tokens: int = MAX_CONTEXT_TOKENS,
) -> dict:
    """Setzt den vollständigen Prompt zusammen [4a]."""
    context, used, info = build_context(chunks, max_tokens=max_tokens, query=query)

    prompt = (
        SYSTEM_INSTRUCTION
        + CLASS_INSTRUCTIONS.get(query_class, "")
        + f"- Antworte in der Sprache: {language}\n"
        + f"\nAnfrage-Klasse: {query_class or 'unbekannt'}\n"
        + f"\nKontext:\n{context}\n"
        + f"\nBenutzeranfrage: {query}"
    )
    return {"prompt": prompt, "used_chunks": used, **info}
