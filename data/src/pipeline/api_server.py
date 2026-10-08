#!/usr/bin/env python3
# ============================================================
# api_server.py — PETRA-RAG FastAPI-Server (Ingestion und Query)
#
# Stellt die Ingestion-Skripte als HTTP-Endpoints für n8n bereit und
# bindet die Query-/Retrieval-Router aus backend/query_router_v3.py ein.
# Port    : 8001 (8000 = ChromaDB, 5678 = n8n)
# Start   : python3 api_server.py
#           oder: uvicorn api_server:app --host 0.0.0.0 --port 8001
# ============================================================

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from chunking import build_splitter, process_chunks
from save_to_chromadb import save_chunks_to_chromadb
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import threading
import chromadb

from split_pdf import split_pdf
import status_cache   # Dashboard: Dokumentstatistik im Hintergrund

from backend.query_router_v3 import (
    router as query_router, retrieval_router, cache_router,
    agent_router, telemetry_router, ops_router,
)



# ── Logging konfigurieren ─────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] api_server – %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Konfiguration über Umgebungsvariablen ───────────────────────────────
VENV_PYTHON = os.environ.get("PETRA_VENV_PYTHON", sys.executable)


PIPELINE_DIR = Path(
    os.environ.get("PETRA_PIPELINE_DIR", "/app/src/pipeline")
)

PDF_FOLDER = Path(
    os.environ.get("WATCH_FOLDER", "/data/pdfs")
).expanduser()

CHROMA_HOST = os.environ.get("CHROMA_HOST", "chromadb")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8000"))

COLLECTION_TEXT = os.environ.get(
    "CHROMA_COLLECTION_TEXT",
    "petra_text_chunks",
)

COLLECTION_TAG = os.environ.get(
    "CHROMA_COLLECTION_TAG",
    "petra_tag_chunks",
)

# ── FastAPI App ───────────────────────────────────────────────
app = FastAPI(
    title="PETRA-RAG Ingestion API",
    description="REST-API für PETRA-RAG PDF-Ingestion-Pipeline (Phase 1b)",
    version="1.1.0",
    docs_url="/docs",     # Swagger UI
    redoc_url="/redoc",   # ReDoc
)

# Serialisiert ChromaDB-Zugriffe (Schreiben und Statusabfragen).
CHROMA_DB_LOCK = threading.Lock()

# ── Include Routes ───────────────────────────────────────────────
for r in (query_router, retrieval_router, cache_router,
          agent_router, telemetry_router, ops_router):
    app.include_router(r)
status_cache.beim_start()   # erste Dokumentstatistik im Hintergrund

# ── Pydantic Request-Modelle ──────────────────────────────────

class ChunkRequest(BaseModel):
    """Eingabe für /chunk — Fliesstext-Blöcke von extract_text."""
    chunks:   list[dict[str, Any]] = Field(..., description="Chunk-Liste aus extract_text")
    filename: str                  = Field(..., description="PDF-Dateiname")


class SaveRequest(BaseModel):
    """Eingabe für /save-to-chromadb — fertige Chunks mit Hash."""
    chunks:   list[dict[str, Any]] = Field(..., description="Chunk-Liste aus chunking")
    filename: str                  = Field(..., description="PDF-Dateiname")
    hash:     str                  = Field(..., description="SHA-256 Hash der PDF")

class OCRRequest(BaseModel):
    page_numbers: list[int] = Field(
        default_factory=list,
        description="1-basierte PDF-Seitennummern für OCR",
    )

class SplitPdfRequest(BaseModel):
    """
    Eingabe für den Endpoint /split-pdf.

    filename:
        Name der großen PDF im Ordner /data/pdfs.
    """

    filename: str = Field(
        ...,
        description="Name der PDF, die bei Bedarf geteilt werden soll",
    )
    
    hash: str
# ── Hilfsfunktion: Skript ausführen ──────────────────────────

def run_script(script_name: str, args: str = "", timeout: int = 360,) -> dict[str, Any]:
    """
    Führt ein Pipeline-Skript mit dem venv-Interpreter aus und parst dessen JSON-Ausgabe.

    Die Skripte laufen als Subprozess, damit ihre Abhängigkeiten
    (PyMuPDF, pytesseract, chromadb, …) aus der venv genutzt werden.
    Ausgaben vor dem ersten "{" werden ignoriert.

    Args:
        script_name: Dateiname in PIPELINE_DIR.
        args: Zusätzliche Kommandozeilenargumente (Shell-Syntax).
        timeout: Maximale Laufzeit in Sekunden.

    Raises:
        HTTPException: 500 bei fehlendem Skript, Exit-Code != 0 oder
            ungültigem JSON.
    """
    script_path = PIPELINE_DIR / script_name
    if not script_path.exists():
        log.error("Skript nicht gefunden: %s", script_path)
        raise HTTPException(
            status_code=500,
            detail={"error": f"Skript nicht gefunden: {script_name}"},
        )

    cmd = f"{VENV_PYTHON} {script_path} {args}".strip()
    log.info("Ausführen: %s", cmd)

    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout,  # 6 Minuten; Spez NODE 09: 300 s + Puffer
    )

    if result.returncode != 0:
        log.error("Skript-Fehler (%s): %s", script_name, result.stderr[:500])
        raise HTTPException(
            status_code=500,
            detail={
                "error":  result.stderr[:1000],
                "script": script_name,
            },
        )

    try:
        stdout = result.stdout.strip()
        json_start = stdout.find("{")

        if json_start == -1:
            raise ValueError("Keine JSON-Ausgabe gefunden")

        return json.loads(stdout[json_start:])

    except (json.JSONDecodeError, ValueError) as exc:
        log.error("JSON-Parse-Fehler (%s): %s | stdout: %s",
                  script_name, exc, result.stdout[:200])
        raise HTTPException(
            status_code=500,
            detail={"error": "Ungültige JSON-Ausgabe", "raw": result.stdout[:500]},
        ) from exc

def run_script_stdin(script_name: str, data: dict[str, Any]) -> dict[str, Any]:
    """
    Führt ein Pipeline-Skript aus, übergibt `data` als JSON über stdin und parst die JSON-Ausgabe.
    """

    script_path = PIPELINE_DIR / script_name

    result = subprocess.run(
        [VENV_PYTHON, str(script_path)],
        input=json.dumps(data, ensure_ascii=False),
        capture_output=True,
        text=True,
        timeout=360,
    )

    if result.returncode != 0:
        log.error("Skript-Fehler (%s): %s", script_name, result.stderr[:500])
        raise HTTPException(
            status_code=500,
            detail={
                "script": script_name,
                "error": result.stderr[:1000],
            },
        )

    stdout = result.stdout.strip()

    json_start = stdout.find("{")
    if json_start == -1:
        raise HTTPException(
            status_code=500,
            detail=f"{script_name} hat kein gültiges JSON zurückgegeben."
        )

    return json.loads(stdout[json_start:])
# ── Endpoints ─────────────────────────────────────────────────

@app.get("/health", summary="Server-Status prüfen")
def health() -> dict[str, str]:
    """
    Liefert Service-Status und Version.
    n8n ruft dies beim Start auf um die API-Verfügbarkeit zu prüfen.
    """
    return {
        "status":  "ok",
        "service": "PETRA-RAG Ingestion API",
        "version": "1.1.0",
    }


@app.get("/list-pdfs", summary="PDFs auflisten und Duplikat-Check")
def list_pdfs() -> dict[str, Any]:
    """
    PDFs im Watch-Folder mit filename, path, hash, neu, valid (Duplikatprüfung per SHA-256).
    """
    return run_script("list_pdfs.py")

@app.post(
    "/split-pdf",
    summary="Große PDF automatisch aufteilen",
)
def split_pdf_endpoint(
    request: SplitPdfRequest,
) -> dict[str, Any]:
    """
    Teilt eine große PDF in Teildateien; der Hash des Originals wird durchgereicht.
    """

    try:
        result = split_pdf(request.filename)

        # Alle Teile behalten die Dokumentidentität des Originals.
        result["hash"] = request.hash

        return result

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        log.exception(
            "PDF-Teilung fehlgeschlagen: %s",
            request.filename,
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

@app.post("/extract/{filename}", summary="PyMuPDF Text + Tabellen extrahieren")
def extract(filename: str) -> dict[str, Any]:
    """
    Extrahiert Text und Tabellen aus der angegebenen PDF.
    Gibt has_text=false zurück wenn die Seite wahrscheinlich gescannt ist.
    n8n nutzt diesen Flag für die OCR-Entscheidung (NODE 09).
    """
    pdf_path = PDF_FOLDER / filename
    if not pdf_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"PDF nicht gefunden: {filename}",
        )
    # Pfad in Anführungszeichen (Leerzeichen im Dateinamen).
    # Timeout 30 min für große PDF-Teile.
    return run_script(
        "extract_text.py",
        f'"{pdf_path}"',
        timeout=1800,
    )

@app.post("/ocr/{filename}", summary="Tesseract OCR für Scan-PDFs")
def ocr(filename: str, request: OCRRequest) -> dict[str, Any]:
    """
    Führt OCR auf der angegebenen PDF durch.
    Nur aufrufen wenn /extract keinen Text gefunden hat (has_text=false).
    NODE 09 OCR-Fallback.
    """
    pdf_path = PDF_FOLDER / filename

    if not pdf_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"PDF nicht gefunden: {filename}",
        )

    pages_json = json.dumps(request.page_numbers)

    return run_script(
        "ocr_fallback.py",
        f'"{pdf_path}" \'{pages_json}\'',
    )

@app.post("/chunk", summary="LangChain Chunking (512 Token, 50 Overlap)")
def chunk(request: ChunkRequest) -> dict[str, Any]:
    """
    Teilt Fliesstext in 512-Zeichen-Chunks mit 50 Zeichen Overlap auf.
    Tabellen und OCR-Blöcke bleiben ungeteilt (TAG-Prinzip).
    NODE 10 (Heading-aware Chunking).
    """
    splitter = build_splitter()
    final = process_chunks(request.chunks, splitter)

    return {
        "filename": request.filename,
        "total_chunks": len(final),
        "chunks": final,
    }


@app.post("/save-to-chromadb", summary="Chunks in ChromaDB speichern")
def save_to_chromadb(request: SaveRequest) -> dict[str, Any]:
    """Speichert Chunks inkl. Embeddings in ChromaDB.

    Raises:
        HTTPException: 503, wenn mindestens ein Chunk nicht gespeichert wurde.
    """   
    with CHROMA_DB_LOCK:
        result = save_chunks_to_chromadb(
            chunks=request.chunks,
            filename=request.filename,
            file_hash=request.hash,
        )

    status_cache.als_veraltet_markieren()   # nächstes /status rechnet neu
    if result.get("chunks_failed", 0) > 0:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "ChromaDB-Speicherung unvollständig",
                "result": result,
            },
        )

    return result

@app.get("/status-summary", summary="Schneller ChromaDB-Status")
def status_summary() -> dict[str, Any]:
    """
    Liefert nur die Chunk-Anzahl beider Collections.
    Es werden keine vollständigen Metadaten geladen.
    """
    try:
        with CHROMA_DB_LOCK:
            client = chromadb.HttpClient(
                host=CHROMA_HOST,
                port=CHROMA_PORT,
            )

            client.heartbeat()

            text_count = client.get_collection(
                COLLECTION_TEXT
            ).count()

            tag_count = client.get_collection(
                COLLECTION_TAG
            ).count()

        return {
            "status": "✅ OK",
            "gesamt_chunks": text_count + tag_count,
            "text_chunks": text_count,
            "tag_chunks": tag_count,
        }

    except Exception as exc:
        log.exception("Schneller ChromaDB-Status fehlgeschlagen")

        raise HTTPException(
            status_code=503,
            detail={
                "message": "ChromaDB-Status nicht verfügbar",
                "error": str(exc),
            },
        ) from exc

@app.get("/status", summary="Dokumentstatistik je Datei (aus dem Status-Cache)")
def status(neu: bool = False) -> dict[str, Any]:
    """
    Chunk-Statistik je Quelldatei. Antwortet sofort aus dem
    Status-Cache (Feld "stand"); die Vollzählung aller Chunks läuft im
    Hintergrund, wenn die Daten veraltet sind, ein Import stattfand oder
    neu=true übergeben wird.
    """
    daten = status_cache.lesen()
    if neu or status_cache.ist_veraltet(daten):
        status_cache.aktualisieren_im_hintergrund()
    if daten is None:
        daten = status_cache.warten(25)
    if daten is None:
        return {
            "gesamt_dateien": None,
            "dateien": [],
            "aktualisierung_laeuft": True,
            "hinweis": ("Die Dokumentstatistik wird gerade zum ersten Mal berechnet (alle Chunks werden "
                        "einmal gelesen, einige Minuten). Bitte danach erneut aktualisieren."),
        }
    return {**daten, "aktualisierung_laeuft": status_cache.laeuft()}


@app.get("/import-log", summary="Protokoll der letzten Importe")
def import_log(limit: int = 50) -> dict[str, Any]:
    """
    Letzte Importe aus REPORT_DIR/import_status.jsonl (je Datei eine Zeile,
    geschrieben von save_to_chromadb.py), älteste zuerst.
    """
    from collections import deque
    from datetime import datetime

    pfad = Path(os.environ.get("REPORT_DIR", "/data/processed")) / "import_status.jsonl"
    if not pfad.exists():
        return {"records": [], "anzahl_gesamt": 0, "hinweis": "Noch kein Import protokolliert."}
    letzte: deque = deque(maxlen=max(1, min(limit, 1000)))
    gesamt = 0
    with pfad.open(encoding="utf-8") as fh:
        for zeile in fh:
            if not zeile.strip():
                continue
            try:
                rec = json.loads(zeile)
            except ValueError:
                continue
            gesamt += 1
            ts = rec.get("timestamp")
            if isinstance(ts, (int, float)):
                rec["zeit"] = datetime.fromtimestamp(ts).strftime("%d.%m.%Y %H:%M")
            letzte.append(rec)
    records = sorted(letzte, key=lambda r: r.get("timestamp") or 0)
    return {"records": records, "anzahl_gesamt": gesamt}


@app.get("/export-report", summary="Import-Protokoll als CSV erstellen")
def export_report() -> dict[str, Any]:
    """
    Erstellt aus dem ChromaDB-Status ein CSV-Importprotokoll (Audit-Trail).
    """
    daten = status_cache.lesen()
    if daten is None:
        status_cache.aktualisieren_im_hintergrund()
        raise HTTPException(
            status_code=503,
            detail="Die Dokumentstatistik wird gerade berechnet. Bitte in einigen Minuten erneut versuchen.",
        )
    if status_cache.ist_veraltet(daten):
        status_cache.aktualisieren_im_hintergrund()
    ergebnis = run_script_stdin("export_report.py", daten)
    ergebnis["stand_daten"] = daten.get("stand")
    return ergebnis

# ── Server-Einstiegspunkt ─────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",   # Alle Interfaces (Docker-intern)
        port=8001,         # NF-11: nur localhost via docker-compose
        reload=False,      # Kein Auto-Reload in Produktion
        log_level="info",
    )
