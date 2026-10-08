"""
services/live_stt.py — Mikrofonaufnahme im Browser und lokale Transkription.

Aufnahme:
  - Primär `st.audio_input()` (Streamlit >= 1.41): MediaRecorder-API des
    Browsers, Übertragung über die bestehende Streamlit-Verbindung, ohne
    zusätzlichen Port oder blockierenden Thread.
  - Fallback `streamlit-mic-recorder` (JS-Komponente) für ältere Versionen.

Kein WebRTC: Eine WebRTC-Lösung blockierte den Skriptthread während der
Aufnahme, benötigte eine zweite Verbindung (ICE/STUN) und verlor Frames.

Transkription lokal mit faster-whisper (services/speech.py); Segmente werden
während der Erkennung fortlaufend angezeigt.

Mikrofonzugriff erfordert im Browser einen sicheren Kontext
(`https://…` oder `http://localhost`). Über `http://<IP>:<port>` verweigert
der Browser den Zugriff.
"""

from __future__ import annotations

import hashlib

import streamlit as st

from services.speech import (
    SpeechError,
    stt_available,
    transcribe_bytes_fallback,
    transcribe_stream,
)

try:  # Fallback-Recorder für Streamlit < 1.41
    from streamlit_mic_recorder import mic_recorder

    _MIC_RECORDER_AVAILABLE = True
except Exception:  # noqa: BLE001 — fehlendes Paket darf die App nicht stoppen
    mic_recorder = None  # type: ignore[assignment]
    _MIC_RECORDER_AVAILABLE = False


def native_recorder_available() -> bool:
    """True, wenn `st.audio_input` existiert (Streamlit >= 1.41)."""
    return hasattr(st, "audio_input")


def recorder_mode() -> str:
    """Aktiver Aufnahmeweg: "native" | "mic_recorder" | "none"."""
    if native_recorder_available():
        return "native"
    if _MIC_RECORDER_AVAILABLE:
        return "mic_recorder"
    return "none"


def recorder_available() -> bool:
    return recorder_mode() != "none"


def _transcribe_with_live_view(
    audio: bytes, language: str, placeholder, *, beam_size: int = 5,
) -> tuple[str, str]:
    """Transkribiert und aktualisiert `placeholder` mit jedem neuen Segment.

    Returns:
        (text, fehlermeldung)
    """
    parts: list[str] = []
    try:
        for segment in transcribe_stream(audio, language, beam_size=beam_size):
            parts.append(segment)
            placeholder.markdown("🎙️ " + " ".join(parts) + " ▌")
    except SpeechError as exc:
        if not parts:
            # Fallback über temporäre Datei (faster-whisper ohne BinaryIO-Unterstützung).
            text, file_error = transcribe_bytes_fallback(bytes(audio), language)
            if text:
                placeholder.markdown("🎙️ " + text)
                return text, ""
            return "", file_error or str(exc)
        return "", str(exc)

    text = " ".join(parts).strip()
    if not text:
        return "", "Es konnte keine verständliche Sprache erkannt werden."
    placeholder.markdown("🎙️ " + text)
    return text, ""


def _digest(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def _render_native(language: str, key: str) -> tuple[str, str]:
    """Aufnahme über `st.audio_input`.

    Der Widget-Wert bleibt über Reruns erhalten. Der SHA-1-Hash der zuletzt
    verarbeiteten Aufnahme in st.session_state verhindert eine erneute
    Transkription derselben Aufnahme.
    """
    digest_key = f"{key}_digest"

    audio = st.audio_input(
        "Sprachnachricht aufnehmen",
        key=f"{key}_native",
        help="Aufnahme starten/stoppen. Die Verarbeitung erfolgt vollständig "
             "lokal mit faster-whisper — es werden keine Audiodaten versendet.",
    )
    if audio is None:
        # Aufnahme verworfen: nächste Aufnahme wieder zulassen.
        st.session_state.pop(digest_key, None)
        return "", ""

    try:
        data = audio.getvalue()
    except Exception as exc:  # noqa: BLE001 — NF-07
        return "", f"Aufnahme konnte nicht gelesen werden: {exc}"

    if not data:
        return "", "Keine Audiodaten empfangen."

    digest = _digest(data)
    if st.session_state.get(digest_key) == digest:
        return "", ""  # bereits transkribiert
    st.session_state[digest_key] = digest

    placeholder = st.empty()
    placeholder.markdown("🎙️ *Transkription läuft …*")
    with st.spinner("Spracherkennung läuft (erster Aufruf lädt einmalig das "
                    "lokale Modell) …"):
        text, error = _transcribe_with_live_view(data, language, placeholder)
    placeholder.empty()
    return text, error


def _render_mic_recorder(language: str, key: str) -> tuple[str, str]:
    """Aufnahme über streamlit-mic-recorder (Fallback)."""
    try:
        audio = mic_recorder(
            start_prompt="🎤 Aufnahme starten",
            stop_prompt="⏹️ Aufnahme beenden",
            just_once=True,
            format="wav",
            use_container_width=True,
            key=f"{key}_recorder",
        )
    except Exception as exc:  # noqa: BLE001 — Mikrofon darf den Chat nicht blockieren
        return "", f"Sprachaufnahme nicht verfügbar: {exc}"

    if not audio or not audio.get("bytes"):
        return "", ""

    placeholder = st.empty()
    with st.spinner("Spracherkennung läuft …"):
        text, error = _transcribe_with_live_view(audio["bytes"], language, placeholder)
    placeholder.empty()
    return text, error


def render_microphone(language: str = "Deutsch", key: str = "petra_stt") -> tuple[str, str]:
    """Rendert die Mikrofonaufnahme und transkribiert lokal.

    Returns:
        ("", "")      keine neue Aufnahme
        (text, "")    Transkription dieses Durchlaufs
        ("", fehler)  Fehlermeldung für die Oberfläche

    Wirft keine Exception (NF-07).
    """
    if not stt_available():
        return "", ("Paket „faster-whisper“ ist nicht installiert — "
                    "Spracheingabe deaktiviert (siehe requirements-audio.txt). "
                    "Die Texteingabe funktioniert unabhängig davon normal.")

    mode = recorder_mode()
    if mode == "none":
        return "", ("Dieses Streamlit ist älter als 1.41 (kein `st.audio_input`) "
                    "und „streamlit-mic-recorder“ ist nicht installiert. Bitte "
                    "Streamlit aktualisieren (siehe requirements-audio.txt).")

    try:
        if mode == "native":
            return _render_native(language, key)
        return _render_mic_recorder(language, key)
    except Exception as exc:  # noqa: BLE001 — NF-07
        return "", f"Spracheingabe fehlgeschlagen: {exc}"


def recorder_hint() -> str:
    """Hinweistext für die Chat-Oberfläche je nach Aufnahmeweg."""
    mode = recorder_mode()
    if mode == "native":
        return ("🎤 Aufnahme starten, sprechen, Aufnahme beenden — der Text "
                "erscheint fortlaufend und landet danach zur Kontrolle im "
                "Eingabefeld. Verarbeitung vollständig lokal (faster-whisper).")
    if mode == "mic_recorder":
        return ("🎤 Aufnehmen → stoppen → der erkannte Text landet im "
                "Eingabefeld. Verarbeitung vollständig lokal (faster-whisper).")
    return "🔇 Spracheingabe in dieser Umgebung nicht verfügbar."
