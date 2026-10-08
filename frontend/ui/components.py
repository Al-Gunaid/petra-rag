"""
ui/components.py — Wiederverwendbare UI-Bausteine.

Zustandslose Darstellungskomponenten; Daten kommen aus den Services.
"""

from __future__ import annotations

import base64
import json

import streamlit as st

from services.chat import ChatAnswer, Source
from services.ingestion import IngestionRun, StageState
from services.system_status import ServiceState, SystemStatus
from config import CONFIG

_STATE_ICON = {
    StageState.PENDING: "⚪",
    StageState.RUNNING: "🔵",
    StageState.DONE: "✅",
    StageState.SKIPPED: "⏭️",
    StageState.ERROR: "❌",
}


def status_badge(state: ServiceState) -> None:
    """Kompakter Dienststatus (Sidebar/Dashboard)."""
    icon = "🟢" if state.ok else "🔴"
    st.markdown(
        f"{icon} **{state.name}** &nbsp;·&nbsp; "
        f"<span style='color:var(--petra-muted);font-size:0.85em'>{state.detail}</span>",
        unsafe_allow_html=True,
    )


def metric_card(label: str, value, help_text: str = "", error: str = "") -> None:
    """Kennzahl mit drei Zuständen: Wert, Fehler (mit Meldung) oder noch nicht geladen."""
    if value is None and error:
        st.metric(label, "⚠️ Fehler", help=f"{help_text}\n\n{error}".strip())
        st.caption(f"⚠️ {error}")
    elif value is None:
        st.metric(label, "wird geladen …", help=help_text or None)
    else:
        st.metric(label, value, help=help_text or None)


def pipeline_progress(run: IngestionRun) -> None:
    """Fortschrittsbalken und Stufenleiste eines Ingestion-Laufs."""
    if run.files_total:
        st.progress(
            run.files_done / max(run.files_total, 1),
            text=f"Dokumente: {run.files_done}/{run.files_total}"
                 + (f" — aktuell: {run.current_file}" if run.current_file else ""),
        )
    cols = st.columns(len(run.stages))
    for col, stage in zip(cols, run.stages):
        with col:
            st.markdown(
                f"<div class='petra-stage petra-{stage.state.value}'>"
                f"{_STATE_ICON[stage.state]}<br/><b>{stage.name}</b><br/>"
                f"<span class='petra-stage-detail'>{stage.detail or '&nbsp;'}</span>"
                "</div>",
                unsafe_allow_html=True,
            )


def source_block(sources: list[Source]) -> None:
    """Aufklappbare Quellenliste, absteigend nach Score sortiert (UC-09, F-11)."""
    if not sources:
        st.caption("Keine Quellen verfügbar.")
        return
    st.markdown(
        f"<div class='petra-sources-title'>📎 Quellen "
        f"({len(sources)}) — nach Relevanz sortiert</div>",
        unsafe_allow_html=True,
    )
    for i, s in enumerate(sorted(sources, key=lambda x: x.score, reverse=True), 1):
        title = f"{i}. {s.document} · Seite {s.page} · Relevanz {s.score_percent}"
        with st.expander(title, expanded=False):
            meta1, meta2, meta3, meta4 = st.columns(4)
            meta1.caption(f"**Hersteller:** {s.manufacturer}")
            meta2.caption(f"**Dokumenttyp:** {s.doc_type}")
            meta3.caption(f"**Dokumentdatum:** {s.doc_date}")
            meta4.caption(f"**Score:** {s.score:.2f}")
            if s.url:
                st.caption(f"**Web-Quelle:** {s.url} (abgerufen {s.retrieved_at or '—'})")
            st.code(s.preview or "(keine Vorschau verfügbar)", language=None)


def answer_footer(answer: ChatAnswer) -> None:
    """Status- und Qualitätszeile unter einer Antwort (F-13, F-14).

    Die Sprachausgabe rendert render_answer_audio() separat, da dafür der
    Nachrichtenindex benötigt wird.
    """
    # F-13: Warnung bei fehlender Verifikation oder Faithfulness < 0,75.
    if not answer.verified or (
        answer.faithfulness is not None and answer.faithfulness < 0.75
    ):
        st.warning("⚠️ Nicht verifiziert — bitte Dokument konsultieren.")
    tags = []
    tags.append("🧪 Mock-Antwort (Phase 2a)" if answer.mock
                else "🖥️ lokale Pipeline" if answer.agent_path == "lokal"
                else "🌐 Web-Recherche (Agent)")
    if answer.faithfulness is not None:
        tags.append(f"Faithfulness {answer.faithfulness:.2f}")
    if answer.latency_s is not None:
        tags.append(f"{answer.latency_s:.1f} s")
    tags.append("⚡ Streaming: Token-für-Token" if answer.streamed_real
                else "⏳ Streaming: nachgebildet (wortweise)")
    st.caption(" · ".join(tags))



def render_answer_audio(msg_index: int, answer: ChatAnswer) -> None:
    """Gibt die Sprachausgabe einer Antwort aus (F-15).

    Beim ersten Rendern einer Nachricht wird ein `<audio autoplay>`-Element
    eingefügt; danach erscheint ein normaler Player. Bereits abgespielte
    Indizes stehen in st.session_state["tts_autoplayed"], damit beim
    Neuzeichnen des Verlaufs nicht alle Antworten erneut starten.

    Browser können Autoplay ohne vorherige Nutzerinteraktion blockieren;
    der Player bleibt dann als Fallback.
    """
    if not answer.audio_wav:
        if answer.tts_error:
            st.caption(f"🔇 Sprachausgabe nicht verfügbar: {answer.tts_error} "
                       "(läuft lokal über pyttsx3, siehe services/speech.py)")
        return

    played_indices: set[int] = st.session_state.setdefault("tts_autoplayed", set())
    if msg_index not in played_indices:
        b64_audio = base64.b64encode(answer.audio_wav).decode("ascii")
        st.markdown(
            f'<audio autoplay style="width:100%;height:32px" '
            f'src="data:audio/wav;base64,{b64_audio}">'
            "Ihr Browser unterstützt kein Audio-Element.</audio>",
            unsafe_allow_html=True,
        )
        played_indices.add(msg_index)
    else:
        st.audio(answer.audio_wav, format="audio/wav")


def export_chat_markdown(chat_history: list[dict]) -> str:
    """Exportiert den Chatverlauf als lesbares Markdown-Dokument."""
    lines = ["# PETRA-RAG — Chatverlauf-Export", ""]
    for msg in chat_history:
        role = "**Benutzer**" if msg["role"] == "user" else "**Assistent**"
        lines.append(f"## {role}")
        lines.append(msg.get("content", ""))
        answer = msg.get("answer")
        if answer and answer.sources:
            lines.append("")
            lines.append("**Quellen:**")
            for s in answer.sources:
                lines.append(f"- {s.document} · Seite {s.page} · "
                              f"Score {s.score:.2f} · Hersteller: {s.manufacturer}")
        if answer and answer.faithfulness is not None:
            lines.append(f"\n*Faithfulness: {answer.faithfulness:.2f} · "
                          f"Agentenpfad: {answer.agent_path}*")
        lines.append("")
    return "\n".join(lines)


def export_chat_json(chat_history: list[dict]) -> str:
    """Exportiert den Chatverlauf als strukturiertes JSON (ohne Audiodaten)."""
    records = []
    for msg in chat_history:
        answer = msg.get("answer")
        record = {"role": msg["role"], "content": msg.get("content", "")}
        if answer:
            record["sources"] = [
                {
                    "document": s.document, "page": s.page,
                    "manufacturer": s.manufacturer, "doc_type": s.doc_type,
                    "score": s.score, "doc_date": s.doc_date,
                }
                for s in answer.sources
            ]
            record["faithfulness"] = answer.faithfulness
            record["verified"] = answer.verified
            record["agent_path"] = answer.agent_path
            record["latency_s"] = answer.latency_s
        records.append(record)
    return json.dumps(records, ensure_ascii=False, indent=2)


def sidebar_summary(status: SystemStatus) -> None:
    """Sidebar mit Kennzahlen, Dienststatus und Importstatistik.

    Unterscheidet "noch nicht abgerufen" (leere services, kein Fehlertext)
    von einem tatsächlichen Fehler. Die Dokumentanzahl erscheint nur, wenn
    GET /status abgefragt wurde (status.documents_fetched).
    """
    st.markdown("### 🏭 PETRA-RAG")
    st.caption("Frontend-Prototyp v3")
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        if status.documents_fetched:
            if status.total_documents is not None:
                st.metric("Dokumente", status.total_documents)
            else:
                st.metric("Dokumente", "⚠️ Fehler", help=status.documents_error)
        else:
            st.metric("Dokumente", "→ Dashboard")
            st.caption("Details im Dashboard")
    with c2:
        if status.total_chunks is not None:
            c2.metric("Chunks", f"{status.total_chunks:,}".replace(",", "."))
        elif status.chunks_error:
            c2.metric("Chunks", "⚠️ Fehler", help=status.chunks_error)
        else:
            # Noch nie abgerufen (kein Fehler, einfach kein Aufruf erfolgt) —
            # siehe Docstring-Ergänzung oben.
            c2.metric("Chunks", "—")
    if status.chunks_error:
        st.caption(f"⚠️ Chunks: {status.chunks_error}")

    st.caption(f"**Embedding-Modell:** `{status.embedding_model}`")
    if not status.services:
        # Noch kein Health-Check durchgeführt (siehe Docstring-Ergänzung
        # oben) — klar von einem tatsächlich festgestellten Ausfall
        # unterscheiden, statt fälschlich "🔴 gestört" zu zeigen.
        pipeline_label = "⚪ Status unbekannt (noch nicht abgerufen)"
    else:
        health_ok = status.service("FastAPI").ok and status.service("ChromaDB").ok
        if not health_ok:
            pipeline_label = "🔴 gestört"
        elif status.chunks_error:
            # KORREKTUR: Health-Checks (FastAPI/ChromaDB-Heartbeat) sagen nichts
            # darüber aus, ob die Bestandsabfrage (/status-summary) gerade
            # tatsächlich funktioniert — z. B. blockiert durch CHROMA_DB_LOCK
            # während eines laufenden Imports. "bereit" wäre hier irreführend.
            pipeline_label = "🟡 eingeschränkt (Bestandsdaten aktuell nicht abrufbar)"
        else:
            pipeline_label = "🟢 bereit"
    st.caption(f"**Pipeline:** {pipeline_label}")
    st.divider()

    st.markdown("**Dienste**")
    if not status.services:
        st.caption("Noch nicht abgerufen — „🔄 Status aktualisieren“ klicken.")
    for s in status.services:
        status_badge(s)
    st.divider()

    if status.collections:
        st.markdown("**Importstatistik**")
        for name, count in status.collections.items():
            label = "Text-Chunks" if name == CONFIG.collection_text else \
                    "Tabellen-Chunks (TAG)" if name == CONFIG.collection_tag else name
            st.caption(f"{label}: {count:,}".replace(",", "."))
        st.caption(f"Letzter Import: {status.last_import}")
