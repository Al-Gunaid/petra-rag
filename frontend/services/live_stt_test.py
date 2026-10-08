"""
services/live_stt_test.py — Tests für services/live_stt.py ohne Mikrofon,
Browser, Streamlit oder Whisper-Modell (Streamlit wird durch einen Stub ersetzt).

Ausführen:  python3 services/live_stt_test.py   (oder: python3 run_tests.py)

Geprüft werden:
  - Auswahl des Aufnahmewegs (st.audio_input / streamlit-mic-recorder)
  - keine erneute Transkription derselben Aufnahme bei Reruns (Hash-Schutz)
  - fortlaufende Anzeige der Segmente
  - kein WebRTC-Code mehr im Modul
  - keine Exception aus `render_microphone()` (NF-07)
  - Übergabe des Transkripts an das Eingabefeld der Chat-Seite
"""

import os
import sys
import types
import re
from contextlib import contextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_PASSED = 0
_FAILED = []


def check(name, condition, detail=""):
    global _PASSED
    if condition:
        _PASSED += 1
        print(f"  ✅ {name}")
    else:
        _FAILED.append(name)
        print(f"  ❌ {name}" + (f"  →  {detail}" if detail else ""))


class _Placeholder:
    """Ersatz für st.empty(); protokolliert alle Ausgaben in `history`."""

    def __init__(self):
        self.history = []

    def markdown(self, body="", **kw):
        self.history.append(body)

    def caption(self, body="", **kw):
        self.history.append(body)

    def empty(self):
        self.history.append(None)

    def container(self):
        return self


def _install_streamlit_stub():
    """Registriert ein minimales `streamlit`-Modul in sys.modules."""
    if "streamlit" in sys.modules:
        return sys.modules["streamlit"]
    mod = types.ModuleType("streamlit")
    mod.placeholders = []
    mod.session_state = {}
    mod.audio_input_return = None
    mod.checkbox_return = False

    def cache_resource(*a, **k):
        def deco(fn):
            return fn
        return deco

    def empty():
        ph = _Placeholder()
        mod.placeholders.append(ph)
        return ph

    @contextmanager
    def spinner(*a, **k):
        yield

    mod.cache_resource = cache_resource
    mod.cache_data = cache_resource
    mod.empty = empty
    mod.spinner = spinner
    mod.markdown = lambda *a, **k: None
    mod.caption = lambda *a, **k: None
    mod.error = lambda *a, **k: None
    mod.warning = lambda *a, **k: None
    mod.checkbox = lambda *a, **k: mod.checkbox_return
    mod.audio_input = lambda *a, **k: mod.audio_input_return
    mod.rerun = lambda: None
    sys.modules["streamlit"] = mod
    return mod


class _FakeAudio:
    """Ersatz für den Rückgabewert von st.audio_input (UploadedFile)."""

    def __init__(self, data: bytes):
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


def run() -> bool:
    st = _install_streamlit_stub()
    import importlib

    live_stt = importlib.import_module("services.live_stt")
    speech = importlib.import_module("services.speech")
    chat_view = importlib.import_module("views.chat")

    print("\n1) WebRTC ist vollständig entfernt")
    import inspect
    source = inspect.getsource(live_stt)
    code_lines = [ln for ln in source.split("\n")
                  if ln.strip().startswith(("import ", "from "))]
    check("kein Import von streamlit_webrtc",
          not any("webrtc" in ln for ln in code_lines), str(code_lines))
    check("kein Import von av/aiortc",
          not any(re.search(r"\b(av|aiortc)\b", ln) for ln in code_lines),
          str(code_lines))
    for gone in ("webrtc_streamer", "WebRtcMode", "rtc_configuration",
                 "PETRA_WEBRTC_STUN_URL", "iceServers", "audio_receiver"):
        check(f"Symbol {gone!r} nicht mehr im Code", gone not in source)
    check("keine blockierende Aufnahmeschleife mehr",
          "while ctx.state.playing" not in source)
    check("kein Hintergrundthread mehr nötig", "threading" not in source)

    print("\n2) Auswahl des Aufnahmewegs")
    has_native = hasattr(st, "audio_input")
    check("natives st.audio_input wird erkannt", live_stt.native_recorder_available() is has_native)
    check("Modus = native", live_stt.recorder_mode() == "native", live_stt.recorder_mode())
    check("Aufnahme verfügbar", live_stt.recorder_available() is True)
    check("Hinweistext passt zum Modus", "lokal" in live_stt.recorder_hint())

    print("\n3) Transkription: fortlaufende Anzeige der Segmente")
    calls = {}

    def fake_stream(audio, language="Deutsch", **kw):
        calls["audio"] = audio
        calls["language"] = language
        calls["beam_size"] = kw.get("beam_size")
        for part in ["Welche Anschlussquerschnitte", "unterstützt die WDU 2.5?"]:
            yield part

    live_stt.transcribe_stream = fake_stream
    ph = _Placeholder()
    text, err = live_stt._transcribe_with_live_view(b"RIFFfake", "Deutsch", ph)
    check("Text vollständig zusammengesetzt",
          text == "Welche Anschlussquerschnitte unterstützt die WDU 2.5?", text)
    check("kein Fehler gemeldet", err == "")
    check("Audio unverändert an faster-whisper übergeben", calls["audio"] == b"RIFFfake")
    check("Sprache durchgereicht", calls["language"] == "Deutsch")
    check("Zwischenstände wurden angezeigt (Live-Transkription)",
          len([h for h in ph.history if h and "▌" in h]) == 2,
          str(ph.history))

    print("\n4) Schutz vor Mehrfach-Transkription bei Reruns")
    st.session_state.clear()
    st.audio_input_return = _FakeAudio(b"RIFFaufnahme-1")
    runs = []

    def counting_stream(audio, language="Deutsch", **kw):
        runs.append(audio)
        yield "erste Aufnahme"

    live_stt.transcribe_stream = counting_stream
    live_stt.stt_available = lambda: True

    t1, e1 = live_stt.render_microphone("Deutsch", key="t")
    t2, e2 = live_stt.render_microphone("Deutsch", key="t")  # Rerun mit gleicher Aufnahme
    check("erste Auswertung liefert Text", t1 == "erste Aufnahme" and e1 == "", f"{t1!r}/{e1!r}")
    check("Rerun transkribiert NICHT erneut", (t2, e2) == ("", ""), f"{t2!r}/{e2!r}")
    check("faster-whisper genau einmal aufgerufen", len(runs) == 1, str(len(runs)))

    st.audio_input_return = _FakeAudio(b"RIFFaufnahme-2")
    t3, _ = live_stt.render_microphone("Deutsch", key="t")
    check("neue Aufnahme wird wieder transkribiert", t3 == "erste Aufnahme" and len(runs) == 2,
          str(len(runs)))

    st.audio_input_return = None  # Aufnahme verworfen
    t4, e4 = live_stt.render_microphone("Deutsch", key="t")
    check("verworfene Aufnahme -> nichts zu tun", (t4, e4) == ("", ""))
    check("Hash-Merker zurückgesetzt", "t_digest" not in st.session_state)

    print("\n5) Fehlertoleranz (NF-07)")
    st.audio_input_return = _FakeAudio(b"RIFFkaputt")
    st.session_state.clear()

    def boom(audio, language="Deutsch", **kw):
        raise speech.SpeechError("Modell konnte nicht geladen werden")
        yield  # pragma: no cover

    live_stt.transcribe_stream = boom
    live_stt.transcribe_bytes_fallback = lambda *a, **k: ("", "auch Dateiweg fehlgeschlagen")
    text, err = live_stt.render_microphone("Deutsch", key="err")
    check("Fehler wird als Text gemeldet, nicht geworfen", text == "" and bool(err), f"{err!r}")

    def explode(*a, **k):
        raise RuntimeError("Widget kaputt")

    st.audio_input = explode
    text, err = live_stt.render_microphone("Deutsch", key="err2")
    check("unerwartete Exception wird abgefangen",
          text == "" and "Widget kaputt" in err, f"{err!r}")

    live_stt.stt_available = lambda: False
    text, err = live_stt.render_microphone("Deutsch", key="err3")
    check("fehlendes faster-whisper wird klar gemeldet",
          text == "" and "faster-whisper" in err, f"{err!r}")

    print("\n6) Sprachzuordnung für faster-whisper")
    check("Deutsch -> de", speech.whisper_language_code("Deutsch") == "de")
    check("Englisch -> en", speech.whisper_language_code("englisch") == "en")
    check("unbekannt -> None (Autoerkennung)", speech.whisper_language_code("Klingonisch") is None)

    print("\n7) Regressionstest: Absturz nach der Aufnahme "
          "(StreamlitWidgetAlreadyInstantiatedError)")
    # Das Eingabefeld "composer_text" existiert in diesem Durchlauf bereits,
    # danach trifft das Transkript ein.
    st.session_state.clear()
    st.session_state["composer_text"] = "Bereits getippter Entwurf"
    before = st.session_state["composer_text"]

    chat_view._apply_transcript("Welche Querschnitte hat die WDU 2.5", "")
    check("`composer_text` wird NICHT nach der Widget-Erzeugung überschrieben",
          st.session_state["composer_text"] == before,
          repr(st.session_state.get("composer_text")))
    check("Text wird stattdessen neutral vorgemerkt",
          st.session_state.get("pending_transcript") == "Welche Querschnitte hat die WDU 2.5")
    check("Fehlermeldung zurückgesetzt", st.session_state.get("stt_error") == "")

    # Folgedurchlauf: _prefill_composer() läuft vor der Widget-Erzeugung.
    chat_view._prefill_composer()
    check("Entwurf + Transkript werden zusammengeführt",
          st.session_state["composer_text"]
          == "Bereits getippter Entwurf Welche Querschnitte hat die WDU 2.5",
          repr(st.session_state["composer_text"]))
    check("Vormerkung ist verbraucht", "pending_transcript" not in st.session_state)

    unchanged = st.session_state["composer_text"]
    chat_view._prefill_composer()
    check("ohne Vormerkung keine Änderung", st.session_state["composer_text"] == unchanged)

    # Der Form-Submit beim Mikrofon-Klick leert das Feld (clear_on_submit);
    # der Entwurf muss über "composer_restore" wiederhergestellt werden.
    st.session_state.clear()
    st.session_state["composer_text"] = ""
    st.session_state["composer_restore"] = "Halber Satz"
    chat_view._prefill_composer()
    check("Entwurf überlebt den Mikrofon-Klick",
          st.session_state["composer_text"] == "Halber Satz")

    st.session_state.clear()
    st.session_state["composer_text"] = "Doppelt"
    st.session_state["composer_restore"] = "Doppelt"
    chat_view._prefill_composer()
    check("keine Verdopplung bei identischem Entwurf",
          st.session_state["composer_text"] == "Doppelt",
          repr(st.session_state["composer_text"]))

    print("\n" + "=" * 62)
    if _FAILED:
        print(f"❌ {len(_FAILED)} Test(s) fehlgeschlagen: {', '.join(_FAILED)}")
        return False
    print(f"✅ Alle {_PASSED} Tests bestanden.")
    return True


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
