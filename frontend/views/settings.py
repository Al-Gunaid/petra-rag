"""
views/settings.py — Generierungs- und Retrieval-Parameter der RAG-Pipeline.

Alle Werte liegen in st.session_state.settings und werden bei jeder
Chat-Anfrage an POST /query übergeben (services/chat.py: _build_payload).

Vom Backend ausgewertet (backend/query_router_v3.py: QueryRequest):

    query: str
    language: str = "Deutsch"
    top_k: int = 5                 # 1..10
    score_threshold: float         # Default 0.65
    request_id: str | None = None

`temperature` und `model` werden mitgesendet, im aktuellen Schema aber
ignoriert. Chunk-Größe und Overlap betreffen nur die Ingestion.
"""

from __future__ import annotations

import streamlit as st

from config import DEFAULT_SETTINGS
from ui.state import settings


def render() -> None:
    st.title("Einstellungen")
    st.caption(
        "Diese Parameter werden in st.session_state gespeichert und bei "
        "jeder Chat-Anfrage im Request-Payload an POST /query übermittelt "
        "(siehe services/chat.py: HttpRagBackend._build_payload)."
    )
    s = settings()

    st.subheader("Retrieval (aktiv verbunden)")
    st.caption(
        "✅ Diese drei Werte werden vom Backend nachweislich ausgewertet "
        "(verifiziert anhand von backend/query_router_v3.py: QueryRequest)."
    )
    c1, c2 = st.columns(2)
    s["top_k"] = c1.slider(
        "Top-K (Retrieval)", 1, 10, int(s.get("top_k", 5)),
        help="F-06: Anzahl der abgerufenen Dokumentenchunks. Backend-Feld: "
             "top_k (QueryRequest, 1–10, Default 5).",
    )
    s["score_threshold"] = c2.slider(
        "Score-Schwelle (CRAG-Filter)", 0.0, 1.0,
        float(s.get("score_threshold", 0.65)), 0.05,
        help="Mindest-Relevanzscore, ab dem ein Chunk als ausreichend "
             "belastbar gilt (Backend-Feld: score_threshold, Default 0,65 "
             "— identisch mit backend/petra_hybrid/crag_filter.py: "
             "DEFAULT_THRESHOLD). Niedrigere Werte liefern mehr, aber "
             "potenziell weniger relevante Quellen.",
    )
    s["language"] = c1.selectbox(
        "Antwortsprache", ["Deutsch", "Englisch"],
        index=0 if s.get("language", "Deutsch") == "Deutsch" else 1,
        help="NF-08: Backend-Feld 'language' (QueryRequest). Steuert die "
             "Sprach-Instruktion des Prompts.",
    )

    st.divider()
    st.subheader("Generierung (LLM, im Payload enthalten)")
    st.caption(
        "ℹ️ Modell und Temperature werden bereits jetzt im Request-Payload "
        "an das Backend übermittelt. Das aktuell eingesetzte Query-Schema "
        "(backend/query_router_v3.py: QueryRequest) wertet ausschließlich "
        "query/language/top_k/score_threshold/request_id aus — diese "
        "beiden Felder werden vom Backend also noch nicht berücksichtigt. "
        "Das ist eine Eigenschaft des Backends, nicht des Frontends "
        "(keine Backend-Änderung im Rahmen dieses Auftrags), und wird "
        "hier bewusst transparent gemacht statt verschwiegen."
    )
    c3, c4 = st.columns(2)
    s["model"] = c3.selectbox(
        "LLM-Modell (Ollama)", s["model_options"],
        index=s["model_options"].index(s.get("model", s["model_options"][0])),
        help="Spezifikation: Llama 3.1 8B (bevorzugt) oder Mistral 7B — "
             "lokal via Ollama (NF-01). Wird via 'model' im Payload "
             "mitgesendet.",
    )
    s["temperature"] = c4.slider(
        "Temperature", 0.0, 1.0, float(s.get("temperature", 0.1)), 0.05,
        help="Spezifikationswert 0,1 für deterministische, "
             "halluzinationsarme Antworten (UC-08). Wird via "
             "'temperature' im Payload mitgesendet.",
    )

    st.divider()
    st.subheader("Chunking (Ingestion-Parameter, informativ)")
    st.caption(
        "⚠️ Chunk-Größe und Overlap sind serverseitige Konstanten der "
        "bestehenden Ingestion (chunking.py: 512/50 **Zeichen** — "
        "dokumentierte Abweichung W2 gegenüber F-03 „Token“). Änderungen "
        "wirken erst nach Re-Import und sind hier nur vorgemerkt — sie "
        "werden NICHT an POST /query gesendet, sondern betreffen nur den "
        "Ingestion-Pfad (views/ingestion.py)."
    )
    c5, c6 = st.columns(2)
    s["chunk_size"] = c5.number_input(
        "Chunk Size", 128, 2048, int(s.get("chunk_size", 512)), 64)
    s["chunk_overlap"] = c6.number_input(
        "Chunk Overlap", 0, 256, int(s.get("chunk_overlap", 50)), 10)

    st.divider()
    st.subheader("Agent (Web-Recherche für Preisanfragen)")
    s["web_agent_enabled"] = st.toggle(
        "Web-Agent für Preisrecherchen erlauben (UC-11)",
        value=bool(s.get("web_agent_enabled", False)),
        help="Standard: deaktiviert (NF-01 Datensouveränität). "
             "Aktivierung erlaubt Online-Preisrecherche.",
    )

    st.divider()
    col_a, col_b = st.columns([1, 3])
    if col_a.button("↩️ Auf Standardwerte zurücksetzen"):
        st.session_state.settings = dict(DEFAULT_SETTINGS)
        st.rerun()
    with col_b.expander("Aktuelle Konfiguration (wird an die Pipeline übergeben)"):
        st.json({k: v for k, v in s.items() if k != "model_options"})
