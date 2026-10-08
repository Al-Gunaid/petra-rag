"""
api/client.py — HTTP-Client des Frontends.

Dünne Transportschicht für die Endpoints des Ingestion-Service
(api_server.py) und die Health-Endpoints von ChromaDB, Ollama und n8n.
Alle Aufrufe liefern ein ApiResult; Netzwerkfehler werden nicht als
Exception weitergegeben (NF-07).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import requests

from config import CONFIG


@dataclass
class ApiResult:
    """Einheitliches Ergebnisobjekt aller Aufrufe."""

    ok: bool
    data: Any = None
    error: str = ""
    status_code: Optional[int] = None
    timeout: bool = False  # unterscheidet "Zeitüberschreitung" von anderen Fehlern

    @property
    def not_found(self) -> bool:
        return self.status_code == 404


class PetraApiClient:
    """Client für den PETRA-RAG Ingestion-Service (FastAPI, Port 8001)."""

    def __init__(self) -> None:
        self.base = CONFIG.endpoints.api
        self._session = requests.Session()

    # ── interne Transporthelfer ───────────────────────────────
    def _request(
        self,
        method: str,
        url: str,
        json_body: Any = None,
        timeout: float = CONFIG.timeout_short,
    ) -> ApiResult:
        try:
            resp = self._session.request(method, url, json=json_body, timeout=timeout)
        except requests.exceptions.ConnectionError:
            return ApiResult(False, error="Dienst nicht erreichbar")
        except requests.exceptions.Timeout:
            return ApiResult(False, error=f"Zeitüberschreitung (>{timeout:.0f}s)", timeout=True)
        except requests.exceptions.RequestException as exc:  # noqa: BLE001
            return ApiResult(False, error=str(exc))

        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text[:300])
            except ValueError:
                detail = resp.text[:300]
            return ApiResult(False, error=str(detail), status_code=resp.status_code)

        try:
            return ApiResult(True, data=resp.json(), status_code=resp.status_code)
        except ValueError:
            return ApiResult(True, data=resp.text, status_code=resp.status_code)

    def _get(self, path: str, timeout: float = CONFIG.timeout_short) -> ApiResult:
        return self._request("GET", f"{self.base}{path}", timeout=timeout)

    def _post(self, path: str, body: Any = None,
              timeout: float = CONFIG.timeout_pipeline) -> ApiResult:
        return self._request("POST", f"{self.base}{path}", json_body=body, timeout=timeout)

    # ── Bestehende Endpunkte (Ist-Stand api_server.py v1.1.0) ─
    def health(self) -> ApiResult:
        return self._get("/health", timeout=CONFIG.timeout_health)

    def health_deep(self) -> ApiResult:
        """GET /ops/health/deep — prüft u. a., ob die Ollama-Modelle vorhanden sind.

        Liefert im Fehlerfall den passenden `ollama pull`-Hinweis. Fehlt der
        Endpoint, wird nur /health verwendet.
        """
        return self._get("/ops/health/deep", timeout=CONFIG.timeout_short)

    def list_pdfs(self) -> ApiResult:
        """GET /list-pdfs — scannt den Watch-Folder.

        Jede PDF wird für die Duplikatprüfung gehasht (SHA-256); die Laufzeit
        wächst mit Anzahl und Größe der Dateien.
        """
        return self._get("/list-pdfs", timeout=CONFIG.timeout_list_pdfs)

    def extract(self, filename: str) -> ApiResult:
        return self._post(f"/extract/{filename}")

    def ocr(self, filename: str) -> ApiResult:
        return self._post(f"/ocr/{filename}")

    def chunk(self, chunks: list[dict], filename: str) -> ApiResult:
        return self._post("/chunk", {"chunks": chunks, "filename": filename})

    def save_to_chromadb(self, chunks: list[dict], filename: str, file_hash: str) -> ApiResult:
        return self._post(
            "/save-to-chromadb",
            {"chunks": chunks, "filename": filename, "hash": file_hash},
        )

    def status_summary(self) -> ApiResult:
        """GET /status-summary — Chunk-Anzahlen.

        Antwort:
            {"status": "...", "gesamt_chunks": int,
             "text_chunks": int, "tag_chunks": int}

        Keine Dokumentanzahl (siehe status_details()). Kann während eines
        Imports auf den CHROMA_DB_LOCK warten.
        """
        return self._get("/status-summary", timeout=CONFIG.timeout_status_summary)

    def status_details(self) -> ApiResult:
        """GET /status — Chunk-Statistik pro Quelldatei (chromadb_status.py).

        services/system_status.py akzeptiert mehrere Feldnamen, da das
        Antwortschema nicht festgelegt ist. Kann während eines Imports auf
        den CHROMA_DB_LOCK warten.
        """
        return self._get("/status", timeout=CONFIG.timeout_status_details)

    def export_report(self) -> ApiResult:
        return self._get("/export-report", timeout=120.0)

    # ── Vorgeschlagene Endpunkte (Probe, degradiert bei 404) ──
    def import_log(self) -> ApiResult:
        """GET /import-log — Vorschlag: liefert import_status.jsonl."""
        return self._get("/import-log", timeout=CONFIG.timeout_short)

    def query(self, payload: dict) -> ApiResult:
        """POST /query — bestehender, vollständiger (nicht-streamender)
        Anfrage-Endpunkt. Wird als Fallback verwendet, falls der
        Streaming-Endpunkt (query_stream) nicht verfügbar ist.
        """
        return self._post("/query", payload, timeout=CONFIG.timeout_pipeline)

    def query_stream_raw(self, payload: dict):
        """POST {query_stream_path} — optionaler Streaming-Endpoint (SSE/NDJSON).

        Returns:
            Offenes `requests.Response` im Stream-Modus (Aufrufer schließt es)
            oder None bei Fehler bzw. HTTP-Status >= 400.
        """
        url = f"{self.base}{CONFIG.query_stream_path}"
        try:
            resp = self._session.post(
                url, json=payload, timeout=CONFIG.timeout_query_stream, stream=True,
            )
        except requests.exceptions.RequestException:
            return None
        if resp.status_code >= 400:
            resp.close()
            return None
        return resp

    def tts(self, text: str, language: str = "Deutsch") -> ApiResult:
        """POST {tts_path} — nicht verwendet.

        Der Endpoint existiert im Backend nicht. Die Sprachausgabe läuft
        lokal über services/speech.py: synthesize_local().
        """
        return self._post(
            CONFIG.tts_path,
            {"text": text, "language": language},
            timeout=CONFIG.timeout_tts,
        )
    def stt(
        self,
        audio_bytes: bytes,
        filename: str = "sprachnachricht.wav",
        mime_type: str = "audio/wav",
        language: str = "Deutsch",
        ) -> ApiResult:
        """POST {stt_path} — nicht verwendet.

        Der Endpoint existiert im Backend nicht. Die Spracherkennung läuft
        lokal über services/speech.py: transcribe_local().
        Upload als multipart/form-data, Feldname "audio".
        """
        url = f"{self.base}{CONFIG.stt_path}"
        files = {"audio": (filename, audio_bytes, mime_type)}
        data = {"language": language}
        try:
            resp = self._session.post(
                url, files=files, data=data, timeout=CONFIG.timeout_stt,
            )
        except requests.exceptions.ConnectionError:
            return ApiResult(False, error="Dienst nicht erreichbar")
        except requests.exceptions.Timeout:
            return ApiResult(
                False,
                error=f"Zeitüberschreitung (>{CONFIG.timeout_stt:.0f}s)",
                timeout=True,
            )
        except requests.exceptions.RequestException as exc:  # noqa: BLE001
            return ApiResult(False, error=str(exc))

        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text[:300])
            except ValueError:
                detail = resp.text[:300]
            return ApiResult(False, error=str(detail), status_code=resp.status_code)

        try:
            return ApiResult(True, data=resp.json(), status_code=resp.status_code)
        except ValueError:
            return ApiResult(True, data=resp.text, status_code=resp.status_code)


class N8nClient:
    """Startet den n8n-Workflow "PDF-Pipline-Parsing" über dessen Webhook-Trigger.

    Der Webhook-Trigger-Node ("Webhook Trigger (petra-ingest)") mündet
    parallel zum Manual-Trigger in denselben Ablauf.
    """

    def __init__(self) -> None:
        self.url = CONFIG.endpoints.n8n_ingest_webhook
        self._session = requests.Session()

    def trigger(self) -> ApiResult:
        try:
            resp = self._session.post(self.url, json={}, timeout=CONFIG.timeout_webhook)
        except requests.exceptions.ConnectionError:
            return ApiResult(False, error="n8n nicht erreichbar")
        except requests.exceptions.Timeout:
            return ApiResult(False, error="Zeitüberschreitung beim Webhook-Aufruf", timeout=True)
        except requests.exceptions.RequestException as exc:  # noqa: BLE001
            return ApiResult(False, error=str(exc))

        if resp.status_code == 404:
            return ApiResult(
                False, status_code=404,
                error="Webhook nicht registriert (Workflow inaktiv oder Pfad "
                      f"'{CONFIG.endpoints.n8n_ingest_webhook_path}' weicht ab)."
            )
        if resp.status_code >= 400:
            return ApiResult(False, status_code=resp.status_code, error=resp.text[:300])
        try:
            return ApiResult(True, data=resp.json(), status_code=resp.status_code)
        except ValueError:
            return ApiResult(True, data=resp.text, status_code=resp.status_code)


class InfraClient:
    """Health-Checks für ChromaDB, Ollama und n8n."""

    def __init__(self) -> None:
        self.e = CONFIG.endpoints

    @staticmethod
    def _probe(url: str) -> ApiResult:
        try:
            resp = requests.get(url, timeout=CONFIG.timeout_health)
            if resp.status_code < 400:
                try:
                    return ApiResult(True, data=resp.json(), status_code=resp.status_code)
                except ValueError:
                    return ApiResult(True, data=resp.text, status_code=resp.status_code)
            return ApiResult(False, error=f"HTTP {resp.status_code}",
                             status_code=resp.status_code)
        except requests.exceptions.RequestException:
            return ApiResult(False, error="Dienst nicht erreichbar")

    def chromadb(self) -> ApiResult:
        """ChromaDB-Heartbeat (API v2, Fallback v1 für ältere Images)."""
        res = self._probe(f"{self.e.chroma}/api/v2/heartbeat")
        if res.ok:
            return res
        return self._probe(f"{self.e.chroma}/api/v1/heartbeat")

    def ollama(self) -> ApiResult:
        """Ollama-Modellliste (GET /api/tags) als Verfügbarkeitsprüfung."""
        return self._probe(f"{self.e.ollama}/api/tags")

    def n8n(self) -> ApiResult:
        """n8n-Health-Endpoint (GET /healthz)."""
        return self._probe(f"{self.e.n8n}/healthz")
