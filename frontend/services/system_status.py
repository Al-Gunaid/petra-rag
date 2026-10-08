"""
services/system_status.py — Aggregation des Systemzustands.

Fasst Health-Checks und ChromaDB-Bestandsstatistik zu einem Statusmodell
zusammen. Enthält keine UI-Logik.

Zwei Abfrageebenen:
  - collect_health() / cached_health_only(): nur Erreichbarkeit
    (FastAPI /health, ChromaDB-Heartbeat, Ollama /api/tags, n8n /healthz).
    Ohne CHROMA_DB_LOCK, auch während eines Imports unkritisch.
  - collect_status() / cached_full_status(): zusätzlich GET /status-summary
    und GET /status. Beide warten serverseitig auf den CHROMA_DB_LOCK und
    werden nur auf Knopfdruck im Dashboard aufgerufen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import streamlit as st

from api.client import InfraClient, PetraApiClient
from config import CONFIG
from services.clients import get_infra_client, get_petra_client


@dataclass
class ServiceState:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class SystemStatus:
    services: list[ServiceState] = field(default_factory=list)

    total_documents: Optional[int] = None
    documents_fetched: bool = False       # True, sobald GET /status versucht wurde
    documents_error: str = ""             # Fehlermeldung statt Platzhalterwert

    total_chunks: Optional[int] = None
    text_chunks: Optional[int] = None
    tag_chunks: Optional[int] = None
    chunks_error: str = ""

    collections: dict[str, int] = field(default_factory=dict)
    files: list[dict[str, Any]] = field(default_factory=list)
    embedding_model: str = CONFIG.embedding_model
    last_import: str = "—"
    ollama_models: list[str] = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        return bool(self.services) and all(s.ok for s in self.services)

    def service(self, name: str) -> ServiceState:
        for s in self.services:
            if s.name == name:
                return s
        return ServiceState(name, False, "unbekannt")


def collect_health(api: PetraApiClient, infra: InfraClient) -> list[ServiceState]:
    """Erreichbarkeit aller Dienste; die Ollama-Modellliste wird als Attribut `models` angehängt."""
    states: list[ServiceState] = []

    r = api.health()
    detail = f"v{r.data.get('version', '?')}" if r.ok and isinstance(r.data, dict) else r.error
    states.append(ServiceState("FastAPI", r.ok, detail))

    r = infra.chromadb()
    states.append(ServiceState("ChromaDB", r.ok, "Heartbeat ok" if r.ok else r.error))

    r = infra.ollama()
    if r.ok and isinstance(r.data, dict):
        models = [m.get("name", "?") for m in r.data.get("models", [])]
        detail = f"{len(models)} Modell(e)" if models else "keine Modelle geladen"
    else:
        models, detail = [], r.error
    states.append(ServiceState("Ollama", r.ok, detail))
    states[-1].models = models  # type: ignore[attr-defined]

    r = infra.n8n()
    states.append(ServiceState("n8n", r.ok, "healthz ok" if r.ok else r.error))
    return states


def collect_chunk_stats(api: PetraApiClient) -> dict[str, Any]:
    """GET /status-summary — Chunk-Anzahlen je Collection.

    Returns:
        Dict mit total_chunks, text_chunks, tag_chunks, collections und
        einer Fehlermeldung in "error" (leer bei Erfolg).
    """
    r = api.status_summary()
    out: dict[str, Any] = {
        "total_chunks": None, "text_chunks": None, "tag_chunks": None,
        "collections": {}, "error": "",
    }
    if r.ok and isinstance(r.data, dict):
        out["text_chunks"] = r.data.get("text_chunks")
        out["tag_chunks"] = r.data.get("tag_chunks")
        out["total_chunks"] = r.data.get(
            "gesamt_chunks",
            (out["text_chunks"] or 0) + (out["tag_chunks"] or 0)
            if out["text_chunks"] is not None or out["tag_chunks"] is not None
            else None,
        )
        if out["text_chunks"] is not None:
            out["collections"][CONFIG.collection_text] = out["text_chunks"]
        if out["tag_chunks"] is not None:
            out["collections"][CONFIG.collection_tag] = out["tag_chunks"]
    elif r.not_found:
        out["error"] = "Endpunkt /status-summary nicht gefunden (Backend-Version prüfen)."
    elif r.timeout:
        out["error"] = ("Zeitüberschreitung bei /status-summary — möglicherweise wartet "
                        "der Server auf die ChromaDB-Sperre, weil gerade ein Import läuft.")
    else:
        out["error"] = r.error or "Unbekannter Fehler bei /status-summary."
    return out


def collect_document_stats(api: PetraApiClient) -> dict[str, Any]:
    """GET /status — Dokumentanzahl und Liste pro Datei (chromadb_status.py).

    Langsam bei großen Beständen (serverseitig bis 1800 s) und blockiert
    während eines Imports am CHROMA_DB_LOCK. Nur gezielt aufrufen.

    Erwartetes Format: Liste unter "dateien", "files" oder "documents",
    optional "gesamt_dateien".
    """
    r = api.status_details()
    out: dict[str, Any] = {"total_documents": None, "files": [], "error": ""}
    if r.ok and isinstance(r.data, dict):
        files = (
            r.data.get("dateien") or r.data.get("files") or r.data.get("documents") or []
        )
        if isinstance(files, list):
            out["files"] = files
            out["total_documents"] = r.data.get("gesamt_dateien", len(files))
        else:
            out["total_documents"] = r.data.get("gesamt_dateien")
        if out["total_documents"] is None:
            out["error"] = ("GET /status hat geantwortet, aber weder eine Pro-Datei-Liste "
                            "noch 'gesamt_dateien' im erwarteten Format geliefert.")
    elif r.not_found:
        out["error"] = "Endpunkt /status nicht gefunden (Backend-Version prüfen)."
    elif r.timeout:
        out["error"] = ("Zeitüberschreitung bei /status — bei großen Beständen kann die "
                        "Pro-Datei-Aggregation lange dauern, zusätzlich blockiert die "
                        "ChromaDB-Sperre während eines laufenden Imports.")
    else:
        out["error"] = r.error or "Unbekannter Fehler bei /status."
    return out


def collect_status(include_documents: bool = True) -> SystemStatus:
    """Vollständiger Systemstatus.

    Args:
        include_documents: False überspringt GET /status und liefert nur
            Health-Checks und Chunk-Anzahlen.
    """
    api, infra = get_petra_client(), get_infra_client()
    status = SystemStatus(services=collect_health(api, infra))

    ollama_state = status.service("Ollama")
    status.ollama_models = getattr(ollama_state, "models", [])

    if status.service("FastAPI").ok:
        chunks = collect_chunk_stats(api)
        status.total_chunks = chunks["total_chunks"]
        status.text_chunks = chunks["text_chunks"]
        status.tag_chunks = chunks["tag_chunks"]
        status.collections = chunks["collections"]
        status.chunks_error = chunks["error"]

        if include_documents:
            docs = collect_document_stats(api)
            status.total_documents = docs["total_documents"]
            status.files = docs["files"]
            status.documents_error = docs["error"]
            status.documents_fetched = True

        # Letzter Import über den optionalen Endpoint /import-log; bei 404
        # bleibt das Feld mit Hinweis unbelegt (NF-07).
        r = api.import_log()
        if r.ok and isinstance(r.data, dict):
            records = r.data.get("records", [])
            if records:
                status.last_import = str(records[-1].get("datei", "—"))
        elif r.not_found and status.last_import == "—":
            status.last_import = "n. v. (Endpunkt /import-log vorgeschlagen)"
    else:
        status.chunks_error = "Ingestion-Service (FastAPI) nicht erreichbar."
        status.documents_error = "Ingestion-Service (FastAPI) nicht erreichbar."

    return status


# st.cache_data, da reine Daten zurückgegeben werden. Die TTL verhindert
# keinen synchronen Netzwerkaufruf bei Ablauf; die Funktionen werden daher
# nur aus Button-Handlern aufgerufen (app.py, views/dashboard.py).

@st.cache_data(ttl=10, show_spinner=False)
def cached_health_only() -> list[ServiceState]:
    """Health-Checks ohne Chunk-/Dokumentabfrage.

    ChromaDB wird nur über den Heartbeat geprüft (ohne CHROMA_DB_LOCK);
    ein laufender Import hat daher keinen Einfluss.
    """
    return collect_health(get_petra_client(), get_infra_client())


@st.cache_data(ttl=15, show_spinner=False)
def cached_full_status() -> SystemStatus:
    """Vollständiger Status für das Dashboard (GET /status und /status-summary).

    Die TTL schützt nur vor schnell aufeinanderfolgenden Klicks.
    """
    return collect_status(include_documents=True)


def invalidate_status_caches() -> None:
    """Verwirft den Cache von cached_full_status() nach einem Import.

    cached_health_only() bleibt unverändert, da ein Import die
    Erreichbarkeit der Dienste nicht beeinflusst.
    """
    cached_full_status.clear()
