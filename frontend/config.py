"""
config.py — Zentrale Konfiguration des PETRA-RAG-Frontends.

Alle Dienst-URLs sind über Umgebungsvariablen konfigurierbar und zeigen
standardmäßig auf localhost-Ports des docker-compose-Verbunds (NF-11).

Endpoints des Ingestion-Service (api_server.py):
    GET  /health              GET  /list-pdfs
    POST /extract/{filename}  POST /ocr/{filename}
    POST /chunk               POST /save-to-chromadb
    GET  /status-summary      GET  /export-report
    GET  /status              POST /split-pdf

GET /status-summary liefert nur Chunk-Summen ("text_chunks", "tag_chunks").
Die Dokumentanzahl wird aus GET /status (Aufschlüsselung pro Datei,
chromadb_status.py) abgeleitet.

Es gibt keinen Upload-Endpoint. PDFs werden direkt in den Watch-Folder
(WATCH_FOLDER, UC-05) geschrieben, den auch api_server.py verwendet.

Ingestion-Start: POST {n8n}/webhook/petra-ingest startet den n8n-Workflow
"PDF-Pipline-Parsing". Optionale Endpoints werden zur Laufzeit geprüft;
fehlen sie, arbeitet das Frontend eingeschränkt weiter (NF-07).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).rstrip("/")


@dataclass(frozen=True)
class Endpoints:
    """URLs des bestehenden Dienstverbunds (docker-compose, nur localhost)."""

    api: str = _env("PETRA_API_URL", "http://localhost:8001")
    chroma: str = _env("PETRA_CHROMA_URL", "http://localhost:8000")
    ollama: str = _env("PETRA_OLLAMA_URL", "http://localhost:11434")
    n8n: str = _env("PETRA_N8N_URL", "http://localhost:5678")

    # Webhook-Pfad des n8n-Workflows "PDF-Pipline-Parsing". Muss mit dem
    # "path"-Parameter des Webhook-Trigger-Node im Workflow übereinstimmen.
    n8n_ingest_webhook_path: str = _env("PETRA_N8N_INGEST_WEBHOOK_PATH",
                                         "petra-ingest")

    @property
    def n8n_ingest_webhook(self) -> str:
        return f"{self.n8n}/webhook/{self.n8n_ingest_webhook_path}"


@dataclass(frozen=True)
class RagasSettings:
    """Konfiguration der RAGAs-Evaluation.

    Judge-Provider: lokales Ollama (Standard) oder OpenAI.
    Variablennamen entsprechen der Projekt-.env.
    """

    judge_provider: str = _env("RAGAS_JUDGE_PROVIDER", "ollama")  # ollama | openai
    judge_model: str = _env("PETRA_RAGAS_JUDGE_MODEL", "qwen2.5:14b")
    embedding_model: str = _env("PETRA_EMBEDDING_MODEL", "bge-m3")


@dataclass
class AppConfig:
    endpoints: Endpoints = field(default_factory=Endpoints)
    ragas: RagasSettings = field(default_factory=RagasSettings)

    # Anzeige-Metadaten (Ist-Stand: MODEL_NAME aus docker-compose)
    embedding_model: str = _env("MODEL_NAME", "BAAI/bge-m3")

    # Collection-Namen wie in api_server.py (COLLECTION_TEXT/COLLECTION_TAG).
    collection_text: str = _env("CHROMA_COLLECTION_TEXT", "petra_text_chunks")
    collection_tag: str = _env("CHROMA_COLLECTION_TAG", "petra_tag_chunks")

    # Muss im Container auf denselben Pfad zeigen wie WATCH_FOLDER der
    # ingestion-api (gleicher Bind-Mount /mnt/petra-rag:/data).
    watch_folder: str = _env("WATCH_FOLDER", "/data/pdfs")

    # # Timeouts in Sekunden.
    # /list-pdfs hasht bei jedem Aufruf alle PDFs (SHA-256, Duplikatprüfung).
    # /status, /status-summary und /export-report können warten, solange
    # api_server.py den CHROMA_DB_LOCK für einen laufenden Import hält.
    timeout_health: float = 2.0
    timeout_short: float = 15.0
    timeout_list_pdfs: float = 90.0       
    timeout_status_summary: float = 60.0  
    timeout_status_details: float = 60.0  
    timeout_pipeline: float = 360.0       # entspricht run_script() im api_server
    timeout_webhook: float = 10.0         # nur Annahmebestätigung des Webhooks
    timeout_query_stream: float = 180.0   
    timeout_tts: float = 60.0
    timeout_stt: float = 60.0             

    # Streaming-Endpoint; im Backend nicht vorhanden (siehe enable_query_stream_probe).
    query_stream_path: str = _env("PETRA_QUERY_STREAM_PATH", "/query/stream")

    # /tts und /stt existieren im Backend nicht. STT/TTS laufen lokal
    # (services/speech.py); die Pfade werden nicht verwendet.
    tts_path: str = _env("PETRA_TTS_PATH", "/tts")  
    stt_path: str = _env("PETRA_STT_PATH", "/stt")  

    # ── Lokale Sprachverarbeitung (services/speech.py) ────────────────
    # Lokale Spracherkennung (faster-whisper, services/speech.py).
    # "small": Kompromiss aus Genauigkeit und CPU-Laufzeit.
    stt_model_size: str = _env("PETRA_STT_MODEL_SIZE", "small")
    stt_device: str = _env("PETRA_STT_DEVICE", "cpu")
    stt_compute_type: str = _env("PETRA_STT_COMPUTE_TYPE", "int8")

    # Streaming-Versuch vor POST /query. Standardmäßig aus, da der Endpoint
    # fehlt und jeder Versuch einen zusätzlichen Roundtrip kostet.
    enable_query_stream_probe: bool = _env(
        "PETRA_ENABLE_QUERY_STREAM_PROBE", "false"
    ).lower() in ("1", "true", "yes")

    # Anzeige-Fallbacks für fehlende Metadaten. Die Ingestion schreibt kein
    # Hersteller-/Dokumenttyp-Feld; der Korpus enthält ausschließlich
    # Weidmüller-Datenblätter. Befüllte Werte haben Vorrang
    # (services/chat.py: _normalize_manufacturer/_normalize_doc_type).
    default_manufacturer: str = _env("PETRA_DEFAULT_MANUFACTURER", "Weidmüller")
    default_doc_type: str = _env("PETRA_DEFAULT_DOC_TYPE", "Datenblatt")

    app_title: str = "PETRA-RAG"
    app_subtitle: str = "Technische Produktberatung für die Automatisierungstechnik"
    version: str = "Frontend-Prototyp v3"


# Standardwerte der Einstellungsseite.
# Generator via Ollama, Temperatur 0,1 (UC-08), Top-k 5, max. 10 (F-06).
# Chunking 512/50 in Zeichen (F-03).

DEFAULT_SETTINGS: dict = {
    "model": "llama3.1:8b",
    "model_options": ["llama3.1:8b", "mistral:7b"],
    "temperature": 0.1,
    "top_k": 5,
    "score_threshold": 0.65,     # = crag_filter.DEFAULT_THRESHOLD im Backend
                                  
    "chunk_size": 512,
    "chunk_overlap": 50,
    "language": "Deutsch",
    "web_agent_enabled": False,  # UC-11: optional, standardmäßig aus (NF-01)
    "tts_enabled": False,        # F-15: lokale Sprachausgabe
}

CONFIG = AppConfig()
