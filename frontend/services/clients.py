"""
services/clients.py — Prozessweit geteilte Instanzen der HTTP-Clients.

st.cache_resource erzeugt genau eine Instanz pro Python-Prozess. Die
zugrunde liegende requests.Session (Connection-Pooling, Keep-Alive) bleibt
damit über alle Reruns und Streamlit-Sitzungen erhalten. Die Clients halten
keinen Benutzerzustand.

Hinweis: requests.Session ist nicht vollständig thread-sicher. Für die hier
verwendeten GET/POST-Aufrufe ohne Änderung von Headern oder Auth zur Laufzeit
ist die gemeinsame Nutzung unkritisch; bei hoher paralleler Last wäre eine
Session pro Worker-Thread robuster.
"""

from __future__ import annotations

import streamlit as st

from api.client import InfraClient, N8nClient, PetraApiClient


@st.cache_resource(show_spinner=False)
def get_petra_client() -> PetraApiClient:
    """Geteilter Client für den Ingestion-Service (FastAPI, Port 8001)."""
    return PetraApiClient()


@st.cache_resource(show_spinner=False)
def get_infra_client() -> InfraClient:
    """Geteilter Client für die Health-Checks von ChromaDB, Ollama und n8n."""
    return InfraClient()


@st.cache_resource(show_spinner=False)
def get_n8n_client() -> N8nClient:
    """Geteilter Client für den Webhook-Trigger des Ingestion-Workflows."""
    return N8nClient()
