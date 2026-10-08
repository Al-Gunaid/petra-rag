"""views/dashboard.py — Systemstatus, Bestandskennzahlen und Dokumentliste.

Daten werden nur auf Klick auf "🔄 Aktualisieren" geladen, da GET /status
und GET /status-summary serverseitig den CHROMA_DB_LOCK halten. Der zuletzt
abgerufene Stand bleibt mit Zeitstempel in st.session_state erhalten.
"""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from services.system_status import cached_full_status, invalidate_status_caches
from ui.components import metric_card, status_badge
from config import CONFIG


def render() -> None:
    head_l, head_r = st.columns([4, 1])
    head_l.title("Dashboard")
    refresh_clicked = head_r.button(
        "🔄 Aktualisieren", use_container_width=True,
        help="Ruft Health-Checks, Chunk-Kennzahlen (GET /status-summary) "
             "und die Dokumentliste (GET /status) ab. Läuft AUSSCHLIESSLICH "
             "bei diesem Klick — niemals automatisch beim Öffnen der Seite.",
    )
    st.caption("Live-Status des PETRA-RAG-Dienstverbunds (nur localhost).")

    if refresh_clicked:
        # Cache vor dem Abruf verwerfen, damit kein veralteter Stand
        # innerhalb der TTL zurückkommt (z. B. direkt nach einem Import).
        invalidate_status_caches()
        with st.spinner("Systemstatus wird abgefragt …"):
            status = cached_full_status()
        st.session_state["dashboard_status"] = {
            "status": status,
            "fetched_at": time.strftime("%H:%M:%S"),
        }

    cached = st.session_state.get("dashboard_status")
    if cached is None:
        st.info(
            "Noch keine Daten geladen — auf „🔄 Aktualisieren“ klicken, um "
            "Systemstatus sowie Chunk- und Dokumentenzahlen abzurufen.",
            icon="ℹ️",
        )
        return

    status = cached["status"]
    st.caption(f"Stand: {cached['fetched_at']} Uhr")

    # ── Dienststatus ─────────────────────────────────────────
    st.subheader("Systemstatus")
    cols = st.columns(4)
    for col, svc in zip(cols, status.services):
        with col, st.container(border=True):
            status_badge(svc)

    if not status.service("Ollama").ok:
        st.info(
            "Ollama ist nicht erreichbar. Das LLM-Modell kann nicht geladen werden."
        )

    # ── Kennzahlen ───────────────────────────────────────────
    st.subheader("Wissensbasis")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        metric_card("Dokumente", status.total_documents,
                    "Anzahl indexierter PDF-Dateien (GET /status, "
                    "Pro-Datei-Statistik)", error=status.documents_error)
    with k2:
        metric_card("Chunks gesamt", status.total_chunks,
                    "Summe aus text_chunks + tag_chunks (GET /status-summary)",
                    error=status.chunks_error)
    with k3:
        metric_card("Embedding-Modell", status.embedding_model,
                    "Ist-Stand laut MODEL_NAME (docker-compose); "
                    "Abweichung W1 dokumentiert")
    with k4:
        metric_card("Letzter Import", status.last_import,
                    "Quelle: vorgeschlagener Endpunkt GET /import-log")

    if status.collections:
        c1, c2 = st.columns(2)
        c1.caption(f"{CONFIG.collection_text}: "
                   f"{status.collections.get(CONFIG.collection_text, 0):,}"
                   .replace(",", "."))
        c2.caption(f"{CONFIG.collection_tag}: "
                   f"{status.collections.get(CONFIG.collection_tag, 0):,}"
                   .replace(",", "."))

    # ── Dokumentliste ────────────────────────────────────────
    st.subheader("Indexierte Dokumente")
    if status.files:
        df = pd.DataFrame(status.files).rename(columns={
            "datei": "Datei", "chunks_gesamt": "Chunks",
            "text_chunks": "Text", "tabellen": "Tabellen (TAG)",
            "ocr_chunks": "OCR", "seiten": "Seiten", "status": "Status",
        })
        st.dataframe(df, use_container_width=True, hide_index=True)
    elif status.documents_error:
        # Fehlgeschlagener Abruf ist nicht gleichbedeutend mit leerem Bestand.
        st.error(f"Dokumentliste konnte nicht geladen werden: {status.documents_error}")
    elif status.service("FastAPI").ok:
        st.caption("Noch keine Dokumente indexiert — Import über die "
                   "Seite „Ingestion“ starten.")
    else:
        st.error("Der Ingestion-Service ist nicht erreichbar; "
                 "Bestandsdaten können nicht geladen werden (NF-07-Fallback).")
