"""views/chat.py — Chatbereich der Produktberatung (PETRA-RAG, Weidmüller).

Die Seite spricht ausschließlich über das RagBackend-Protokoll mit dem
aktiven Backend (services/chat.py: HttpRagBackend) — kein Mock-Modus, keine
Platzhalterdaten. Layout der Quellenanzeige gemäß UC-09/F-11/NF-06.

VEREINFACHUNG (siehe DOKUMENTATION.md, Punkt 1): Der manuelle Hersteller-/
Dokumenttyp-Filter (Expander "Filter (Hersteller / Dokumenttyp)") wurde
vollständig entfernt. Das System verarbeitet ausschließlich Weidmüller-
Komponenten — ein Filter, dessen einzig sinnvoller Wert ohnehin immer
"Weidmüller" gewesen wäre, war überflüssiger Overhead ohne Nutzen für den
Nutzer (zusätzlicher UI-Zustand, zusätzlicher `cached_filter_options()`-
Aufruf pro Seitenaufruf) und wurde ersatzlos gestrichen. Die Anzeige-
Normalisierung einzelner Quellenangaben (z. B. "unbekannt" → "Weidmüller"
in den Quellen einer Chat-Antwort, siehe services/chat.py:
_normalize_manufacturer/_normalize_doc_type) ist davon unabhängig und
bleibt bestehen.

"""

from __future__ import annotations

import streamlit as st

from services import live_stt
from ui.components import (
    answer_footer,
    export_chat_json,
    export_chat_markdown,
    render_answer_audio,
    source_block,
)
from ui.markdown import render_markdown
from ui.state import settings


def _render_export() -> None:
    if not st.session_state.chat_history:
        return
    c1, c2, _ = st.columns([1, 1, 3])
    c1.download_button(
        "⬇️ Export (Markdown)",
        export_chat_markdown(st.session_state.chat_history),
        file_name="petra_rag_chatverlauf.md",
        mime="text/markdown",
        use_container_width=True,
    )
    c2.download_button(
        "⬇️ Export (JSON)",
        export_chat_json(st.session_state.chat_history),
        file_name="petra_rag_chatverlauf.json",
        mime="application/json",
        use_container_width=True,
    )


def _prefill_composer() -> None:
    """
    Schreibt vorgemerkten Text ins Eingabefeld — VOR dessen Erzeugung.

    """
    pending = st.session_state.pop("pending_transcript", "") or ""
    restore = st.session_state.pop("composer_restore", "") or ""
    if not pending and not restore:
        return

    current = st.session_state.get("composer_text", "") or ""
    parts: list[str] = []
    for part in (current, restore, pending):
        part = part.strip()
        if part and part not in parts:
            parts.append(part)
    st.session_state["composer_text"] = " ".join(parts)


def _apply_transcript(transcribed: str, stt_error: str) -> None:
    """
    Merkt einen erkannten Text für das Sendefeld vor.

    """
    if transcribed:
        st.session_state["pending_transcript"] = transcribed
        st.session_state["stt_error"] = ""
    else:
        st.session_state["stt_error"] = (
            stt_error or "Es konnte keine verständliche Sprache erkannt werden."
        )
    st.rerun()


def _handle_voice_recording() -> None:
    """Spracheingabe (Punkt 7/10 im Modul-Docstring).

    `live_stt.render_microphone()` kapselt Aufnahme UND lokale
    Transkription (faster-whisper) und wirft niemals eine Exception nach
    außen; zurück kommt immer (text, fehlermeldung). Ein `st.rerun()`
    erfolgt nur, wenn tatsächlich etwas passiert ist — sonst würde die
    laufende Aufnahme bei jedem Durchlauf neu gestartet.
    """
    transcript, stt_error = live_stt.render_microphone(
        language=settings().get("language", "Deutsch"),
        key="petra_stt",
    )
    if not transcript and not stt_error:
        return  # keine neue Aufnahme — nichts zu tun
    _apply_transcript(transcript, stt_error)


def _stream_answer(backend, prompt: str) -> str:
    """Zeigt die Antwort während der Erzeugung an und rendert sie final.

    Ablauf:
      1. Ein `st.empty()`-Platzhalter zeigt den wachsenden Text inkl.
         Cursor — die gewohnte Streaming-Optik bleibt also erhalten.
      2. Nach dem letzten Häppchen wird derselbe Platzhalter durch die
         finale, tabellenfähige Darstellung ERSETZT (kein zweiter Block).

    Maßgeblich für die finale Darstellung ist `backend.last_answer().text`
    (der unveränderte Originaltext des Backends); der gestreamte Text
    dient nur als Rückfallebene.
    """
    placeholder = st.empty()
    parts: list[str] = []
    for chunk in backend.stream_answer(prompt, settings()):
        parts.append(chunk)
        placeholder.markdown("".join(parts) + " ▌")

    streamed = "".join(parts)
    final_text = (getattr(backend.last_answer(), "text", "") or "").strip() or streamed
    with placeholder.container():
        render_markdown(final_text)
    return final_text


def _process_prompt(prompt: str, backend) -> None:
    """Verarbeitet eine neu abgeschickte Anfrage vollständig.

    """
    st.session_state.chat_history.append(
        {"role": "user", "content": prompt, "answer": None}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Anfrage wird verarbeitet …"):
                text = _stream_answer(backend, prompt)
            answer = backend.last_answer()
            source_block(answer.sources)
            answer_footer(answer)
            render_answer_audio(len(st.session_state.chat_history), answer)
        except Exception as exc:  # noqa: BLE001 — NF-07: kein UI-Absturz
            text, answer = "", None
            st.error(
                "Die Anfrage konnte nicht verarbeitet werden. Bitte erneut "
                f"versuchen oder Anfrage vereinfachen. (Details: {exc})"
            )

    st.session_state.chat_history.append(
        {"role": "assistant", "content": text or "", "answer": answer}
    )


def render() -> None:
    _prefill_composer()

    st.title("Produktberatung")
    st.caption("KI-Assistent für Weidmüller-Automatisierungskomponenten "
               "(Klemmen, I/O-Module, Sensoren) auf Basis lokal indexierter "
               "Datenblätter.")
    backend = st.session_state.rag_backend
    st.caption(f"Backend: {backend.name}")

    settings()["tts_enabled"] = st.toggle(
        "🔊 Sprachausgabe (Coqui TTS)", value=bool(settings().get("tts_enabled", False)),
        help="F-15: Lokale Sprachausgabe (pyttsx3, kein Backend-Aufruf, "
             "siehe services/speech.py). Schlägt sie fehl (z. B. fehlendes "
             "Paket/Systemabhängigkeit), bleibt der Chattext unverändert "
             "nutzbar und der Fehler wird angezeigt.",
    )

    # ── Verlauf rendern ──────────────────────────────────────
    for idx, msg in enumerate(st.session_state.chat_history):
        with st.chat_message(msg["role"]):
            # 
            render_markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("answer"):
                source_block(msg["answer"].sources)
                answer_footer(msg["answer"])
                # Punkt 9 (Modul-Docstring): automatisches Abspielen statt
                # Klick auf einen Player — Index steuert, dass jede
                # Antwort nur EINMAL automatisch abgespielt wird.
                render_answer_audio(idx, msg["answer"])

    # ── Eingabe (Textfeld + Senden + Mikrofon in EINER Zeile) ──
    # Beispielanfrage bewusst am Weidmüller-Produktsortiment orientiert
    # (z. B. WDU 2.5 Reihenklemme), da das System AUSSCHLIESSLICH für
    # Weidmüller-Datenblätter ausgelegt ist (kein generisches Beispiel).
    st.divider()
    if st.session_state.get("stt_error"):
        st.caption(f"🔇 Spracherkennung: {st.session_state['stt_error']}")
    else:
        st.caption(live_stt.recorder_hint())

    # ── Eingabezeile: [ Textfeld ] [ 🎤 ] [ ➤ ] ───────────────────────
    st.session_state.setdefault("mic_panel_open", False)

    with st.form("composer_form", clear_on_submit=True):
        with st.container(key="petra_enter_submit"):
            enter_pressed = st.form_submit_button(
                "Senden", use_container_width=True,
            )

        text_col, mic_col, send_col = st.columns(
            [10, 1.2, 1.2], vertical_alignment="bottom",
        )
        with text_col:
            # st.text_input statt st.text_area — nur so löst Streamlit den
            # Form-Submit auch beim Drücken von Enter aus (bei text_area
            # wäre Enter immer ein Zeilenumbruch, auch im Formular).
            typed_text = st.text_input(
                "Nachricht",
                key="composer_text",
                label_visibility="collapsed",
                placeholder="Frage zu Weidmüller-Automatisierungskomponenten stellen … "
                            "(z. B. „Welche Anschlussquerschnitte unterstützt die WDU 2.5?“)",
            )
        with mic_col:
            # Zwei Container-Keys je nach Öffnungszustand (siehe
            # ui/theme.py) — Basis-Look für beide, Akzentrahmen nur für
            # „geöffnet“.
            mic_container_key = (
                "petra_mic_toggle_open" if st.session_state["mic_panel_open"]
                else "petra_mic_toggle"
            )
            with st.container(key=mic_container_key):
                mic_clicked = st.form_submit_button(
                    "🎤", help="Sprachnachricht aufnehmen",
                    use_container_width=True,
                )
        with send_col:
            with st.container(key="petra_send_btn"):
                send_clicked = st.form_submit_button(
                    "➤ Senden", use_container_width=True,
                )

    if mic_clicked:
        st.session_state["mic_panel_open"] = not st.session_state["mic_panel_open"]
        # Entwurf sichern: Der Klick war ein Form-Submit; ohne dies ginge
        # ein bereits getippter Text beim Öffnen des Mikrofons verloren.
        st.session_state["composer_restore"] = typed_text or ""
        st.rerun()

    if st.session_state["mic_panel_open"]:
        with st.container(border=True):
            _handle_voice_recording()


    prompt = (typed_text or "").strip()
    submitted = bool(send_clicked or enter_pressed)
    if submitted and prompt:
        _process_prompt(prompt, backend)
    elif submitted and not prompt:
        st.warning("Bitte zuerst eine Nachricht eingeben oder per Mikrofon aufnehmen.")

    c1, _ = st.columns([1, 5])
    if c1.button("🗑️ Verlauf löschen"):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    _render_export()
