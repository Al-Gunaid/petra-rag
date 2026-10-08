"""
services/ingestion.py — Anbindung der Ingestion-Pipeline.

Standardpfad: Start des n8n-Workflows "PDF-Pipline-Parsing" über
POST {n8n}/webhook/petra-ingest (n8n_workflows/PDF-Pipline-Parsing.json).
Der Fortschritt wird über GET /status-summary (Delta der Text- und
Tabellen-Chunks) gepollt; der Workflow bietet keinen eigenen
Fortschritts-Endpoint, und GET /status ist für Polling zu langsam.

Fallback ohne n8n (Direktmodus, run_direct): Aufruf der FastAPI-Endpoints
in der Reihenfolge des Workflows:
list-pdfs -> extract -> [ocr] -> chunk -> save-to-chromadb.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterator, Optional

from api.client import ApiResult, PetraApiClient
from config import CONFIG
from services.clients import get_n8n_client
from services.system_status import invalidate_status_caches

# Anzeigestufen entsprechend dem n8n-Workflow.
STAGE_NAMES: list[str] = [
    "PDF-Erkennung",
    "Parsing (PyMuPDF)",
    "OCR (Tesseract)",
    "Chunking",
    "Embedding",
    "Speicherung (ChromaDB)",
    "Fertig",
]


class StageState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class Stage:
    name: str
    state: StageState = StageState.PENDING
    detail: str = ""


@dataclass
class IngestionRun:
    """Zustand eines Importlaufs für die Fortschrittsanzeige."""

    stages: list[Stage] = field(
        default_factory=lambda: [Stage(n) for n in STAGE_NAMES]
    )
    current_file: str = ""
    files_total: int = 0
    files_done: int = 0
    log: list[str] = field(default_factory=list)
    finished: bool = False
    error: str = ""

    def set(self, name: str, state: StageState, detail: str = "") -> None:
        """Setzt Zustand und Detailtext einer Stufe; Details werden protokolliert."""
        for s in self.stages:
            if s.name == name:
                s.state, s.detail = state, detail
        if detail:
            self.log.append(f"[{name}] {detail}")

    def reset_file_stages(self) -> None:
        """Setzt die dateibezogenen Stufen für das nächste Dokument zurück."""
        for s in self.stages:
            if s.name not in ("PDF-Erkennung", "Fertig"):
                s.state, s.detail = StageState.PENDING, ""


ProgressCb = Callable[[IngestionRun], None]


def check_new_pdfs(api: PetraApiClient) -> ApiResult:
    """PDF-Liste inkl. Duplikat-Kennzeichnung (list_pdfs.py)."""
    return api.list_pdfs()


def try_start_via_n8n() -> ApiResult:
    """Startet den Ingestion-Workflow über den Webhook (geteilter N8nClient)."""
    return get_n8n_client().trigger()


# PDF-Upload (UC-05): Es gibt keinen Upload-Endpoint. Dateien werden direkt
# in den Watch-Folder (WATCH_FOLDER=/data/pdfs) geschrieben. Der
# frontend-Container benötigt dafür denselben Bind-Mount wie ingestion-api
# (/mnt/petra-rag:/data, docker-compose.yml).

ALLOWED_SUFFIX = ".pdf"
MAX_UPLOAD_MB = 50  # Obergrenze aus Eingangsprüfung I1


@dataclass
class UploadResult:
    filename: str
    ok: bool
    detail: str = ""


def save_uploaded_files(uploaded_files: list[Any]) -> list[UploadResult]:
    """Schreibt Streamlit-UploadedFile-Objekte in den Watch-Folder.

    Prüft Dateiendung und Größe (I1). Schreibfehler werden je Datei im
    UploadResult gemeldet.
    """
    results: list[UploadResult] = []
    target_dir = CONFIG.watch_folder

    try:
        os.makedirs(target_dir, exist_ok=True)
    except OSError as exc:
        return [UploadResult(f.name, False,
                              f"Watch-Folder '{target_dir}' nicht beschreibbar: {exc}")
                for f in uploaded_files]

    for f in uploaded_files:
        name = f.name
        if not name.lower().endswith(ALLOWED_SUFFIX):
            results.append(UploadResult(name, False, "Nur .pdf-Dateien werden akzeptiert."))
            continue
        size_mb = f.size / (1024 * 1024)
        if size_mb > MAX_UPLOAD_MB:
            results.append(UploadResult(
                name, False,
                f"{size_mb:.1f} MB überschreitet die Obergrenze von "
                f"{MAX_UPLOAD_MB} MB (I1). Große PDFs werden serverseitig "
                f"über /split-pdf verarbeitet, sofern bereits im "
                f"Watch-Folder abgelegt."))
            continue

        dest_path = os.path.join(target_dir, name)
        try:
            with open(dest_path, "wb") as out:
                out.write(f.getbuffer())
            results.append(UploadResult(name, True, f"nach {dest_path} geschrieben"))
        except OSError as exc:
            results.append(UploadResult(name, False, f"Schreibfehler: {exc}"))

    return results


def poll_status_summary(api: PetraApiClient) -> ApiResult:
    """Chunk-Anzahlen für die Fortschrittsanzeige nach dem Webhook-Start."""
    return api.status_summary()


def _new_files(payload: Any) -> list[dict]:
    """Filtert neue, gültige PDFs aus der Antwort von /list-pdfs.

    Akzeptiert die Listenschlüssel pdfs/files/dateien/neu und die
    Flags neu/new und valid.
    """
    if not isinstance(payload, dict):
        return []
    items = (
        payload.get("pdfs")
        or payload.get("files")
        or payload.get("dateien")
        or payload.get("neu")
        or []
    )
    result = []
    for it in items:
        if not isinstance(it, dict):
            continue
        if it.get("neu", it.get("new", True)) and it.get("valid", True):
            result.append(it)
    return result


def run_direct(api: PetraApiClient, on_update: Optional[ProgressCb] = None
               ) -> Iterator[IngestionRun]:
    """Direktmodus ohne n8n: Workflow-Schritte über die FastAPI-Endpoints.

    Generator; liefert das Laufmodell nach jedem Zustandswechsel.
    Fehler einer Datei brechen nur diese Datei ab.
    """
    run = IngestionRun()

    def emit() -> IngestionRun:
        if on_update:
            on_update(run)
        return run

    # PDF-Erkennung
    run.set("PDF-Erkennung", StageState.RUNNING, "Watch-Folder wird geprüft …")
    yield emit()
    r = api.list_pdfs()
    if not r.ok:
        run.set("PDF-Erkennung", StageState.ERROR, r.error)
        run.error, run.finished = r.error, True
        yield emit()
        return

    files = _new_files(r.data)
    run.files_total = len(files)
    if not files:
        run.set("PDF-Erkennung", StageState.DONE, "Keine neuen PDFs gefunden.")
        run.set("Fertig", StageState.DONE, "Nichts zu importieren.")
        run.finished = True
        yield emit()
        return
    run.set("PDF-Erkennung", StageState.DONE, f"{len(files)} neue PDF(s) erkannt.")
    yield emit()

    # Je Datei: Parsing -> [OCR] -> Chunking -> Embedding + Speicherung
    for f in files:
        filename = f.get("filename") or f.get("name") or ""
        file_hash = f.get("hash", "")
        run.current_file = filename
        run.reset_file_stages()
        yield emit()

        run.set("Parsing (PyMuPDF)", StageState.RUNNING, filename)
        yield emit()
        r = api.extract(filename)
        if not r.ok:
            run.set("Parsing (PyMuPDF)", StageState.ERROR, r.error)
            run.error = f"{filename}: {r.error}"
            continue
        blocks = (r.data or {}).get("chunks", [])
        has_text = bool((r.data or {}).get("has_text", True))
        run.set("Parsing (PyMuPDF)", StageState.DONE, f"{len(blocks)} Blöcke")
        yield emit()

        if has_text:
            run.set("OCR (Tesseract)", StageState.SKIPPED,
                    "Textschicht vorhanden — OCR nicht erforderlich.")
        else:
            run.set("OCR (Tesseract)", StageState.RUNNING, filename)
            yield emit()
            r = api.ocr(filename)
            if not r.ok:
                run.set("OCR (Tesseract)", StageState.ERROR, r.error)
                run.error = f"{filename}: {r.error}"
                continue
            blocks = (r.data or {}).get("chunks", [])
            run.set("OCR (Tesseract)", StageState.DONE, f"{len(blocks)} OCR-Blöcke")
        yield emit()

        run.set("Chunking", StageState.RUNNING, "RecursiveCharacterTextSplitter …")
        yield emit()
        r = api.chunk(blocks, filename)
        if not r.ok:
            run.set("Chunking", StageState.ERROR, r.error)
            run.error = f"{filename}: {r.error}"
            continue
        final_chunks = (r.data or {}).get("chunks", [])
        run.set("Chunking", StageState.DONE,
                f"{(r.data or {}).get('total_chunks', len(final_chunks))} Chunks")
        yield emit()

        # Embedding und Speicherung laufen serverseitig in einem Endpoint
        # (save_to_chromadb.py); die Anzeige trennt sie nur visuell.
        run.set("Embedding", StageState.RUNNING, "bge-m3 Batch-Embedding …")
        run.set("Speicherung (ChromaDB)", StageState.RUNNING, "Upsert …")
        yield emit()
        r = api.save_to_chromadb(final_chunks, filename, file_hash)
        if not r.ok:
            run.set("Embedding", StageState.ERROR, r.error)
            run.set("Speicherung (ChromaDB)", StageState.ERROR, r.error)
            run.error = f"{filename}: {r.error}"
            continue
        stored = (r.data or {}).get("chunks_saved",
                                    (r.data or {}).get("stored", "?"))
        run.set("Embedding", StageState.DONE, "abgeschlossen")
        run.set("Speicherung (ChromaDB)", StageState.DONE, f"{stored} gespeichert")
        run.files_done += 1
        yield emit()

    ok = run.files_done == run.files_total and not run.error
    run.set(
        "Fertig",
        StageState.DONE if ok else StageState.ERROR,
        f"{run.files_done}/{run.files_total} Dokument(e) importiert.",
    )
    run.finished = True
    if run.files_done:
        # Status-Cache nur verwerfen, wenn sich der Bestand geändert hat.
        invalidate_status_caches()
    yield emit()
