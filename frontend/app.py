"""
app.py — Einstiegspunkt des PETRA-RAG-Frontends.

Start:  streamlit run app.py

Angebundene Dienste (nur localhost): FastAPI :8001, ChromaDB :8000,
n8n :5678, Ollama :11434.

Schichten:  views (Seiten) -> ui (Darstellung/State) ->
            services (fachliche Aggregation) -> api (Transport).
"""

from __future__ import annotations

import time

import streamlit as st

from config import CONFIG
from services.system_status import cached_health_only
from ui.components import status_badge
from ui.state import init_state
from ui.theme import apply_theme
from views import chat as view_chat
from views import dashboard as view_dashboard
from views import ingestion as view_ingestion
from views import settings as view_settings
from views import evaluation as view_evaluation
from views import evaluation_skript as view_evaluation_skript  # v3.7

# Die Sidebar wird bei jedem Rerun vor der Seite gerendert. Sie führt daher
# kein automatisches I/O aus: GET /status-summary hält serverseitig den
# CHROMA_DB_LOCK und würde während eines Imports die gesamte UI blockieren.
# Dokument-/Chunk-Kennzahlen zeigt ausschließlich das Dashboard.


def _render_sidebar() -> None:
    """
    Statische Sidebar mit Health-Check auf Knopfdruck.

    Beim Rendern wird nur st.session_state gelesen. Der Health-Check prüft
    ausschließlich die Erreichbarkeit der Dienste (ohne CHROMA_DB_LOCK).
    """
    st.markdown("### 🏭 PETRA-RAG")
    st.caption("Weidmüller-Automatisierungskomponenten")
    st.divider()

    cached = st.session_state.get("sidebar_health")
    if cached is None:
        st.caption("ℹ️ Dienststatus noch nicht abgerufen.")
    else:
        for svc in cached["services"]:
            status_badge(svc)
        st.caption(f"Stand: {cached['fetched_at']} Uhr")

    if st.button(
        "🔄 Health-Check ausführen", use_container_width=True,
        key="btn_health_check",
        help="Prüft NUR die Erreichbarkeit von FastAPI/ChromaDB/Ollama/n8n "
             "(reiner Heartbeat, KEINE Chunk-/Dokumentenabfrage, KEIN "
             "CHROMA_DB_LOCK). Läuft ausschließlich bei Klick — niemals "
             "automatisch bei Seitenwechseln oder Einstellungsänderungen.",
    ):
        try:
            with st.spinner("Health-Check läuft …"):
                services = cached_health_only()
            st.session_state["sidebar_health"] = {
                "services": services,
                "fetched_at": time.strftime("%H:%M:%S"),
            }
        except Exception:  # noqa: BLE001 — Sidebar darf nie blockieren (NF-07)
            st.warning("Health-Check derzeit nicht möglich.")
        st.rerun()

    st.divider()
    st.caption("Dokument- und Chunk-Kennzahlen: siehe Dashboard "
               "(dort ebenfalls nur auf Klick, siehe „🔄 Aktualisieren“).")


def main() -> None:
    st.set_page_config(
        page_title=f"{CONFIG.app_title} · {CONFIG.version}",
        page_icon="🏭",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    apply_theme()
    init_state()

    pages = [
        st.Page(view_dashboard.render, title="Dashboard", icon="📊",
                url_path="dashboard", default=True),
        st.Page(view_ingestion.render, title="Ingestion", icon="📥",
                url_path="ingestion"),
        st.Page(view_chat.render, title="Produktberatung", icon="💬",
                url_path="chat"),
        st.Page(view_evaluation_skript.render, title="Evaluation", icon="🧪", 
                url_path="evaluation"),
        st.Page(view_settings.render, title="Einstellungen", icon="⚙️",
                url_path="settings"),
    ]
    nav = st.navigation(pages)

    with st.sidebar:
        _render_sidebar()
        st.divider()
        st.caption(f"{CONFIG.app_subtitle}")
        st.caption(f"{CONFIG.version} · nur localhost (NF-11)")

    nav.run()


if __name__ == "__main__":
    main()
