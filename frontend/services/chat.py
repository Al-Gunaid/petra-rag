"""
services/chat.py — Chat-Service der PETRA-RAG-Oberfläche.

Verbindet die Streamlit-Oberfläche mit dem Backend:

    Streamlit -> FastAPI POST /query -> n8n-RAG-Workflow -> JSON-Antwort

Aufgaben:
- Request-Payload aus den Einstellungen aufbauen (top_k, score_threshold, language)
- Backend-Antwort in ChatAnswer/Source überführen
- Hersteller/Dokumenttyp für die Anzeige normalisieren (kein Filter)
- Antwort wortweise an die Oberfläche streamen
- optionale lokale Sprachausgabe (F-15, services/speech.py)

Backend-Schema (backend/query_router_v3.py):

    class QueryRequest(BaseModel):
        query: str
        language: str = "Deutsch"
        top_k: int = 5              # 1..10
        score_threshold: float      # Default 0.65 (crag_filter.DEFAULT_THRESHOLD)
        request_id: str | None = None

Pydantic ignoriert unbekannte Felder. `temperature` und `model` werden
mitgesendet, vom Backend in diesem Stand aber nicht ausgewertet.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Protocol

import streamlit as st

from config import CONFIG
from services.clients import get_petra_client


@dataclass
class Source:
    """
    Repräsentiert eine Quelle, die zur Beantwortung einer Anfrage verwendet wurde.

    Die Klasse bildet das einheitliche Quellenschema der Chat-Oberfläche ab.
    Sie unterstützt sowohl lokale Dokumentquellen als auch optionale Webquellen.
    """

    document: str
    page: Optional[int] = None
    manufacturer: str = "—"
    doc_type: str = "—"
    doc_date: str = "—"
    score: float = 0.0
    preview: str = ""
    url: Optional[str] = None
    retrieved_at: Optional[str] = None

    @property
    def score_percent(self) -> str:
        """
        Gibt den Relevanzwert der Quelle als gerundeten Prozentwert zurück.
        """

        return f"{round(self.score * 100)} %"


@dataclass
class ChatAnswer:
    """
    Enthält die vollständige Antwort der PETRA-RAG-Pipeline.

    Neben dem generierten Antworttext werden die verwendeten Quellen,
    der Verifikationsstatus, der Agentenpfad und die gemessene Laufzeit
    gespeichert.
    """

    text: str = ""
    sources: list[Source] = field(default_factory=list)
    faithfulness: Optional[float] = None
    verified: bool = False
    agent_path: str = "lokal"
    latency_s: Optional[float] = None
    mock: bool = False
    streamed_real: bool = False       # True, wenn echtes Token-Streaming genutzt wurde
    audio_wav: Optional[bytes] = None  # F-15: TTS-Ausgabe, sofern erzeugt
    tts_error: str = ""               # Klartext statt stillem Fehlschlag (NF-07)


class RagBackend(Protocol):
    """
    Definiert die Schnittstelle eines austauschbaren RAG-Backends.

    Die Chat-Oberfläche arbeitet ausschließlich gegen dieses Protokoll.
    Dadurch kann die Backend-Implementierung ausgetauscht werden, ohne
    die Darstellungsschicht anzupassen.
    """

    name: str

    def stream_answer(
        self,
        query: str,
        settings: dict,
    ) -> Iterator[str]:
        """
        Verarbeitet eine Benutzerfrage und liefert den Antworttext schrittweise.
        """
        ...

    def last_answer(self) -> ChatAnswer:
        """
        Gibt das vollständige Ergebnis der zuletzt verarbeiteten Anfrage zurück.
        """
        ...


def _sources_from_payload(data: dict) -> list[Source]:
    """Überführt das Backend-Quellenschema in Source-Objekte.

    Das Backend (response_schema.SourceModel) liefert deutsche Feldnamen
    ("hersteller", "dokumenttyp", ...). Englische Feldnamen werden als
    Fallback ebenfalls akzeptiert.
    """
    sources: list[Source] = []
    for source_data in data.get("sources", []) or []:
        sources.append(
            Source(
                document=source_data.get("dokument", source_data.get("dateiname", source_data.get("document", "unbekannt"))),
                page=source_data.get("seite", source_data.get("page")),
                manufacturer=_normalize_manufacturer(
                    source_data.get("hersteller", source_data.get("manufacturer"))
                ),
                doc_type=_normalize_doc_type(
                    source_data.get("dokumenttyp", source_data.get("doc_type"))
                ),
                doc_date=source_data.get("datum", source_data.get("doc_date", "—")) or "—",
                score=float(source_data.get("score") or 0.0),
                preview=source_data.get("textauszug", source_data.get("textvorschau", source_data.get("preview", ""))) or "",
                url=source_data.get("url"),
                retrieved_at=source_data.get("retrieved_at"),
            )
        )
    return sources


# Platzhalterwerte für fehlende Metadaten (Backend-Default "unbekannt").
_UNKNOWN_MANUFACTURER_VALUES = {"unbekannt", "unknown", "n/a", "none", "", "—", "-"}
_UNKNOWN_DOC_TYPE_VALUES = {"unbekannt", "unknown", "n/a", "none", "", "—", "-"}


def _normalize_manufacturer(value: Any) -> str:
    """Ersetzt fehlende Herstellerangaben durch CONFIG.default_manufacturer.

    Die Ingestion schreibt kein Herstellerfeld; der Korpus enthält
    ausschließlich Weidmüller-Datenblätter. Befüllte Werte haben Vorrang.
    Wirkt nur auf die Anzeige.
    """
    text = str(value).strip() if value else ""
    if not text or text.lower() in _UNKNOWN_MANUFACTURER_VALUES:
        return CONFIG.default_manufacturer
    return text


def _normalize_doc_type(value: Any) -> str:
    """Ersetzt fehlende Dokumenttypen durch CONFIG.default_doc_type.

    Der Backend-Default "text" ist ein gültiger Chunk-Typ und bleibt erhalten.
    """
    text = str(value).strip() if value else ""
    if not text or text.lower() in _UNKNOWN_DOC_TYPE_VALUES:
        return CONFIG.default_doc_type
    return text


# Verlustfreie Zerlegung: "".join(chunks) == text. Zeilenumbrüche bleiben
# erhalten, damit Markdown-Tabellen beim Streaming nicht zerfallen.
_STREAM_CHUNK_RE = re.compile(r"\s*\S+|\s+")


def _stream_chunks(text: str) -> Iterator[str]:
    """Zerlegt `text` verlustfrei in wortweise Streaming-Häppchen."""
    for match in _STREAM_CHUNK_RE.finditer(text or ""):
        yield match.group(0)


def _latency_from_payload(data: dict, start_time: float) -> float:
    latency_ms = data.get("latency_ms")
    if latency_ms is not None:
        return round(float(latency_ms) / 1000, 2)
    return round(time.time() - start_time, 2)


class HttpRagBackend:
    """
    HTTP-Anbindung an die PETRA-RAG-Pipeline.

    Standardpfad: POST /query liefert die vollständige JSON-Antwort; das
    Streaming wird clientseitig wortweise nachgebildet.

    Optional (CONFIG.enable_query_stream_probe): Abfrage von
    PETRA_QUERY_STREAM_PATH (Standard /query/stream) mit SSE- oder
    NDJSON-Zeilen ("delta"/"token"). Bei Fehler oder unbekanntem Format
    erfolgt der Fallback auf POST /query.
    """

    name = "PETRA-RAG Pipeline"

    def __init__(self) -> None:
        # st.cache_resource: geteilte, langlebige Client-Instanz (Connection-
        # Pooling) statt einer neuen PetraApiClient()/requests.Session() pro
        # Chat-Backend-Instanz. Die HttpRagBackend-Instanz selbst bleibt wie
        # zuvor pro Streamlit-Sitzung in st.session_state.rag_backend
        # gehalten (benutzerspezifischer Zustand: self._answer), NICHT die
        # zustandslose Transportschicht (siehe services/clients.py).
        self._api = get_petra_client()
        self._answer = ChatAnswer()

    # ── Payload-Aufbau ──────────────────────────────────────────
    @staticmethod
    def _build_payload(query: str, settings: dict) -> dict:
        """Baut den Request-Payload für POST /query aus st.session_state.settings.

        Ausgewertet werden `top_k`, `score_threshold` und `language`.
        `temperature` und `model` werden mitgesendet, aber vom
        QueryRequest-Schema in diesem Stand ignoriert.
        """
        payload = {
            "query": query,
            "language": settings.get("language", "Deutsch"),
            "top_k": int(settings.get("top_k", 5)),
            "score_threshold": float(settings.get("score_threshold", 0.65)),
            "temperature": float(settings.get("temperature", 0.1)),
            "model": settings.get("model", "llama3.1:8b"),
        }
        return payload

    # ── Echtes Streaming (NICHT VERIFIZIERT) ───────────────────
    def _try_real_stream(self, payload: dict, start_time: float) -> Optional[Iterator[str]]:
        """
        Versucht Token-Streaming über query_stream_raw().

        Returns:
            Generator über Text-Tokens oder None, wenn der Endpoint nicht
            verfügbar ist. Wirft keine Exception (NF-07).
        """
        resp = self._api.query_stream_raw(payload)
        if resp is None:
            return None

        def _gen() -> Iterator[str]:
            text_parts: list[str] = []
            final_data: dict[str, Any] = {}
            saw_any_line = False
            try:
                for raw_line in resp.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue
                    line = raw_line[6:] if raw_line.startswith("data: ") else raw_line
                    line = line.strip()
                    if not line or line == "[DONE]":
                        continue
                    try:
                        obj = json.loads(line)
                    except (ValueError, TypeError):
                        # Kein JSON — als reinen Text-Token behandeln.
                        saw_any_line = True
                        text_parts.append(raw_line)
                        yield raw_line
                        continue
                    if not isinstance(obj, dict):
                        continue
                    delta = obj.get("delta", obj.get("token"))
                    if delta:
                        saw_any_line = True
                        text_parts.append(delta)
                        yield delta
                    # Abschluss-Objekt erkennt man an "sources"/"faithfulness"/"answer".
                    if any(k in obj for k in ("sources", "faithfulness", "answer", "verified")):
                        final_data = obj
            finally:
                resp.close()

            if not saw_any_line:
                self._answer = ChatAnswer(streamed_real=False)
                return

            answer_text = final_data.get("answer") or "".join(text_parts)
            self._answer = ChatAnswer(
                text=answer_text,
                sources=_sources_from_payload(final_data),
                faithfulness=final_data.get("faithfulness"),
                verified=bool(final_data.get("verified", False)),
                agent_path=final_data.get("agent_path", "lokal"),
                latency_s=_latency_from_payload(final_data, start_time),
                mock=False,
                streamed_real=True,
            )

        return _gen()

    # ── Fallback: vollständige JSON-Antwort + simuliertes Streaming ─
    def _fallback_full_response(self, payload: dict, start_time: float) -> Iterator[str]:
        result = self._api.query(payload)

        if not result.ok:
            message = (
                f"Die PETRA-RAG-Pipeline ist derzeit nicht erreichbar oder hat einen "
                f"Fehler gemeldet: {result.error}"
            )
            self._answer = ChatAnswer(
                text=message, verified=False,
                latency_s=round(time.time() - start_time, 2),
            )
            yield message
            return

        data = result.data if isinstance(result.data, dict) else {}
        answer_text = data.get("answer") or "Ich weiß es nicht."

        self._answer = ChatAnswer(
            text=answer_text,
            sources=_sources_from_payload(data),
            faithfulness=data.get("faithfulness"),
            verified=bool(data.get("verified", False)),
            agent_path=data.get("agent_path", "lokal"),
            latency_s=_latency_from_payload(data, start_time),
            mock=False,
            streamed_real=False,
        )

        # Kein str.split(): Zeilenumbrüche müssen für Markdown-Tabellen
        # erhalten bleiben (siehe _stream_chunks, ui/markdown.py).
        chunks = list(_stream_chunks(answer_text))
        # Sehr lange Antworten (z. B. große Vergleichstabellen) sollen nicht
        # allein durch die Streaming-Optik spürbar verzögert werden.
        delay = 0.01 if len(chunks) <= 300 else 0.0
        for chunk in chunks:
            yield chunk
            if delay:
                time.sleep(delay)

    # ── Öffentliche Schnittstelle ───────────────────────────────
    def stream_answer(self, query: str, settings: dict) -> Iterator[str]:
        start_time = time.time()
        payload = self._build_payload(query, settings)

        # Kontrollanzeige der übertragenen Parameter. Bewusste Ausnahme von
        # der Regel "keine UI-Logik in services/".
        st.sidebar.caption(
            f"🛠️ Request-Parameter: top_k={payload['top_k']} · "
            f"score_threshold={payload['score_threshold']} · "
            f"language={payload['language']}"
        )

        # Standardmäßig deaktiviert: Das Backend stellt keinen Streaming-
        # Endpoint bereit; jeder Versuch kostet einen zusätzlichen
        # Roundtrip. Aktivierung über PETRA_ENABLE_QUERY_STREAM_PROBE=true.
        if CONFIG.enable_query_stream_probe:
            real_stream = self._try_real_stream(payload, start_time)
            if real_stream is not None:
                got_output = False
                for token in real_stream:
                    got_output = True
                    yield token
                if got_output and self._answer.text:
                    self._maybe_synthesize_tts(settings)
                    return
                # Endpoint erreichbar, aber ohne verwertbare Ausgabe: Fallback.

        yield from self._fallback_full_response(payload, start_time)
        self._maybe_synthesize_tts(settings)

    # ── F-15: Sprachausgabe (lokal, pyttsx3) ─────────────────────
    def _maybe_synthesize_tts(self, settings: dict) -> None:
        """
        Erzeugt optional eine lokale Sprachausgabe der Antwort (F-15).

        Synthese lokal über services/speech.py (pyttsx3), ohne Backend-Aufruf.
        Fehler werden in ChatAnswer.tts_error abgelegt; der Antworttext
        bleibt nutzbar (NF-07).
        """
        if not settings.get("tts_enabled") or not self._answer.text:
            return
        from services.speech import synthesize_local  # lokaler Import: optionale Abhängigkeit
        try:
            wav_bytes, error = synthesize_local(
                self._answer.text, settings.get("language", "Deutsch"),
            )
        except Exception as exc:  # noqa: BLE001 — NF-07
            self._answer.tts_error = f"Sprachausgabe fehlgeschlagen: {exc}"
            return
        if error:
            self._answer.tts_error = error
            return
        self._answer.audio_wav = wav_bytes

    def last_answer(self) -> ChatAnswer:
        return self._answer


def get_rag_backend() -> RagBackend:
    """Factory für das konfigurierte RAG-Backend."""

    return HttpRagBackend()