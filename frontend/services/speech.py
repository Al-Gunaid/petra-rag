"""
services/speech.py — Lokale Spracherkennung (STT) und Sprachausgabe (TTS).

Das Backend stellt keine /stt- oder /tts-Endpoints bereit. Beide Funktionen
laufen im Streamlit-Prozess; Audiodaten verlassen den Rechner nicht (NF-01).

  - STT: faster-whisper (CTranslate2). Das Modell wird beim ersten Aufruf
    heruntergeladen (~/.cache) und per st.cache_resource im Speicher gehalten.
    Audio wird als io.BytesIO übergeben; die Dekodierung übernimmt PyAV
    (Abhängigkeit von faster-whisper, kein separates ffmpeg nötig).
    transcribe_stream() liefert Segmente einzeln für die Live-Anzeige.
  - TTS: pyttsx3 mit der Sprachsynthese des Betriebssystems
    (unter Linux espeak-ng), offline ohne Modell-Download.
"""

from __future__ import annotations

import io
import os
import tempfile
from typing import Iterator, Optional, Union

import streamlit as st

from config import CONFIG


class SpeechError(RuntimeError):
    """Fehler der lokalen Sprachverarbeitung mit anzeigbarer Meldung."""


# ISO-639-1-Codes für faster-whisper; Schlüssel entsprechen settings()["language"].
_LANGUAGE_TO_WHISPER_CODE = {"deutsch": "de", "englisch": "en"}

# Whisper arbeitet intern mit 16 kHz.
SAMPLE_RATE = 16000


def whisper_language_code(language: str) -> Optional[str]:
    """Wandelt „Deutsch“/„Englisch“ in den ISO-Code; None aktiviert die automatische Spracherkennung."""
    return _LANGUAGE_TO_WHISPER_CODE.get((language or "").strip().lower())


@st.cache_resource(show_spinner="Spracherkennungsmodell wird geladen …")
def _get_whisper_model():
    """Lädt das Whisper-Modell einmal pro Prozess.

    Konfiguration über PETRA_STT_MODEL_SIZE, PETRA_STT_DEVICE und
    PETRA_STT_COMPUTE_TYPE (config.py).
    """
    from faster_whisper import WhisperModel  # optionale Abhängigkeit

    return WhisperModel(
        CONFIG.stt_model_size,
        device=CONFIG.stt_device,
        compute_type=CONFIG.stt_compute_type,
    )


def stt_available() -> bool:
    """True, wenn `faster-whisper` importierbar ist. Lädt kein Modell."""
    try:
        import faster_whisper  # noqa: F401
    except Exception:  # noqa: BLE001 — jeder Importfehler gilt als „nicht verfügbar“
        return False
    return True


def load_model() -> tuple[Optional[object], str]:
    """Lädt das Modell.

    Returns:
        (modell, fehlermeldung); genau einer der beiden Werte ist gesetzt (NF-07).
    """
    try:
        return _get_whisper_model(), ""
    except ImportError:
        return None, (
            "Paket „faster-whisper“ ist nicht installiert. Bitte "
            "`pip install faster-whisper` ausführen (siehe "
            "requirements-audio.txt im Projektstamm)."
        )
    except Exception as exc:  # noqa: BLE001 — darf den Chat nicht blockieren
        return None, f"Spracherkennungsmodell konnte nicht geladen werden: {exc}"


def _as_model_input(audio):
    """Bytes werden zu io.BytesIO; numpy-Arrays (16 kHz Mono) bleiben unverändert."""
    if isinstance(audio, (bytes, bytearray, memoryview)):
        return io.BytesIO(bytes(audio))
    return audio


def transcribe_stream(
    audio: Union[bytes, object],
    language: str = "Deutsch",
    *,
    beam_size: int = 5,
    vad_filter: bool = True,
) -> Iterator[str]:
    """Transkribiert lokal und liefert die Segmente einzeln.

    Args:
        audio: Audio-Bytes (WAV/WebM/OGG) oder float32-numpy-Array, 16 kHz Mono.
        language: Klartext-Sprachname, siehe whisper_language_code().

    Raises:
        SpeechError: bei Modell- oder Transkriptionsfehlern.
    """
    model, error = load_model()
    if error or model is None:
        raise SpeechError(error or "Spracherkennungsmodell nicht verfügbar.")

    try:
        # transcribe() liefert einen Generator; die Berechnung erfolgt beim Iterieren.
        segments, _info = model.transcribe(
            _as_model_input(audio),
            language=whisper_language_code(language),
            beam_size=beam_size,
            vad_filter=vad_filter,
        )
        for segment in segments:
            text = (getattr(segment, "text", "") or "").strip()
            if text:
                yield text
    except Exception as exc:  # noqa: BLE001 — NF-07
        raise SpeechError(f"Transkription fehlgeschlagen: {exc}") from exc


def transcribe_local(audio_bytes: bytes, language: str = "Deutsch") -> tuple[str, str]:
    """Transkribiert eine Aufnahme vollständig.

    Returns:
        (text, fehlermeldung); bei Erfolg ist die Fehlermeldung leer.
    """
    if not audio_bytes:
        return "", "Keine Audiodaten empfangen."

    try:
        parts = list(transcribe_stream(audio_bytes, language))
    except SpeechError as exc:
        # Fallback über temporäre Datei für faster-whisper-Versionen ohne BinaryIO-Unterstützung.
        text, file_error = transcribe_bytes_fallback(audio_bytes, language)
        if text:
            return text, ""
        return "", file_error or str(exc)

    text = " ".join(parts).strip()
    if not text:
        return "", "Es konnte keine verständliche Sprache erkannt werden."
    return text, ""


def transcribe_bytes_fallback(
    audio_bytes: bytes, language: str = "Deutsch"
) -> tuple[str, str]:
    """Transkription über eine temporäre WAV-Datei (Fallback)."""
    model, error = load_model()
    if error or model is None:
        return "", error or "Spracherkennungsmodell nicht verfügbar."

    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        segments, _info = model.transcribe(
            tmp_path,
            language=whisper_language_code(language),
            beam_size=5,
            vad_filter=True,
        )
        text = " ".join((getattr(seg, "text", "") or "").strip() for seg in segments).strip()
    except Exception as exc:  # noqa: BLE001 — NF-07
        return "", f"Transkription fehlgeschlagen: {exc}"
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    if not text:
        return "", "Es konnte keine verständliche Sprache erkannt werden."
    return text, ""


_LANGUAGE_VOICE_HINTS = {"deutsch": ("german", "de"), "englisch": ("english", "en")}


def synthesize_local(text: str, language: str = "Deutsch") -> tuple[bytes, str]:
    """Erzeugt eine WAV-Sprachausgabe lokal mit pyttsx3.

    Pro Aufruf wird eine neue Engine erzeugt, da pyttsx3-Engines nicht
    über mehrere Aufrufe oder Threads hinweg wiederverwendbar sind.
    Unter Linux wird das Systempaket espeak-ng benötigt.

    Returns:
        (wav_bytes, fehlermeldung); bei Erfolg ist die Fehlermeldung leer.
    """
    if not text.strip():
        return b"", "Kein Text zum Vorlesen vorhanden."

    try:
        import pyttsx3
    except ImportError:
        return b"", (
            "Paket „pyttsx3“ ist nicht installiert. Bitte `pip install "
            "pyttsx3` ausführen (siehe requirements-audio.txt); unter "
            "Linux wird zusätzlich das Systempaket „espeak-ng“ benötigt."
        )

    tmp_path: Optional[str] = None
    try:
        engine = pyttsx3.init()
        # Stimme anhand von Name oder ID der Sprache auswählen.
        hint_name, hint_id = _LANGUAGE_VOICE_HINTS.get(
            language.strip().lower(), ("", "")
        )
        if hint_name or hint_id:
            for voice in engine.getProperty("voices") or []:
                name = (getattr(voice, "name", "") or "").lower()
                vid = (getattr(voice, "id", "") or "").lower()
                if (hint_name and hint_name in name) or (hint_id and hint_id in vid):
                    engine.setProperty("voice", voice.id)
                    break

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        engine.save_to_file(text, tmp_path)
        engine.runAndWait()
        try:
            engine.stop()
        except Exception:  # noqa: BLE001
            pass

        with open(tmp_path, "rb") as f:
            wav_bytes = f.read()
    except Exception as exc:  # noqa: BLE001 — NF-07
        return b"", f"Sprachausgabe fehlgeschlagen: {exc}"
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    if not wav_bytes:
        return b"", "Sprachausgabe lieferte keine Audiodaten."
    return wav_bytes, ""
