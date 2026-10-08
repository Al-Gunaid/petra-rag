"""
ui/state.py — Initialisierung und Zugriff auf den Sitzungszustand.

Kapselt st.session_state, damit Views keine Schlüsselnamen duplizieren.
Der Chatverlauf ist sitzungsbezogen und wird nicht persistiert.
"""

from __future__ import annotations

import streamlit as st

from config import DEFAULT_SETTINGS
from services.chat import get_rag_backend


def init_state() -> None:
    """Legt fehlende Schlüssel in st.session_state mit Standardwerten an."""
    if "settings" not in st.session_state:
        st.session_state.settings = dict(DEFAULT_SETTINGS)
    if "chat_history" not in st.session_state:
        # Liste von {"role": "user"|"assistant", "content": str,
        #            "answer": ChatAnswer | None}
        st.session_state.chat_history = []
    if "rag_backend" not in st.session_state:
        st.session_state.rag_backend = get_rag_backend()
    if "ingestion_run" not in st.session_state:
        st.session_state.ingestion_run = None


def settings() -> dict:
    """Einstellungen der aktuellen Sitzung (wird direkt verändert, keine Kopie)."""
    return st.session_state.settings
