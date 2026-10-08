#!/usr/bin/env python3
# ============================================================
# save_to_chromadb.py — Embedding und Speicherung in ChromaDB
#
#   - übernimmt fertige Chunks vom API-Server
#   - berechnet Embeddings (MODEL_NAME, Standard BAAI/bge-m3)
#   - speichert Text- und Tabellen-Chunks in getrennten Collections
#   - registriert dieselben Chunks im BM25-Index (index_chunks_hybrid)
#   - setzt .done-Marker für die Duplikaterkennung (list_pdfs.py)
#
# Performance: BM25Index.add_or_replace() baut den gesamten Index neu auf
# und schreibt die Persistenzdatei vollständig. Der Aufruf erfolgt daher
# nur einmal pro Datei, aber synchron im Request; bei großem Korpus kann
# das Sekunden bis Minuten dauern.
#
# Import durch api_server.py; Direktaufruf mit JSON über stdin möglich.
# ChromaDB-Verbindung und Upserts mit Wiederholungsversuchen.
# ============================================================

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any
import time
import chromadb

from backend.petra_hybrid.ingestion_hook import index_chunks_hybrid

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] save_to_chromadb – %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── ENV-Konfiguration ────────────────────────────────────────
MODEL_NAME: str = os.environ.get("MODEL_NAME", "BAAI/bge-m3")

CHROMA_HOST: str = os.environ.get("CHROMA_HOST", "localhost")
CHROMA_PORT: int = int(os.environ.get("CHROMA_PORT", "8000"))

REPORT_DIR = Path(os.environ.get("REPORT_DIR", "/data/processed"))
IMPORT_STATUS_FILE = REPORT_DIR / "import_status.jsonl"

COLLECTION_TEXT: str = os.environ.get(
    "CHROMA_COLLECTION_TEXT",
    "petra_text_chunks",
)
COLLECTION_TAG: str = os.environ.get(
    "CHROMA_COLLECTION_TAG",
    "petra_tag_chunks",
)

HASH_FOLDER: Path = Path(
    os.environ.get(
        "HASH_FOLDER",
        os.path.expanduser("~/petra-rag/data/processed/hashes"),
    )
)

EMBEDDING_BATCH_SIZE: int = int(os.environ.get("EMBEDDING_BATCH_SIZE", "8"))
CHROMA_BATCH_SIZE: int = int(os.environ.get("CHROMA_BATCH_SIZE", "500"))
MAX_RETRIES: int = int(os.environ.get("CHROMA_MAX_RETRIES", "3"))

# Muss dem Lesepfad in query_router_v3.py (PETRA_BM25_INDEX_PATH) entsprechen.
BM25_INDEX_PATH: str = os.environ.get("PETRA_BM25_INDEX_PATH", "/data/bm25_index.jsonl")

# --Status-import-------------------------
def write_import_status(
    filename: str,
    chunks: list[dict[str, Any]],
    stored: int,
    failed: int,
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    pages = set()
    text_count = 0
    table_count = 0
    ocr_count = 0

    for chunk in chunks:
        pages.add(str(chunk.get("page", "0")))
        typ = chunk.get("type", "text")

        if typ == "table":
            table_count += 1
        elif typ == "ocr":
            ocr_count += 1
        else:
            text_count += 1

    record = {
        "datei": filename,
        "chunks_gesamt": len(chunks),
        "chunks_saved": stored,
        "chunks_failed": failed,
        "text_chunks": text_count,
        "tabellen": table_count,
        "ocr_chunks": ocr_count,
        "seiten": len(pages),
        "status": "✅ OK" if failed == 0 else "⚠️ Teilweise",
        "timestamp": int(time.time()),
    }

    with IMPORT_STATUS_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

# ── Embedding-Modell laden ───────────────────────────────────

def load_embedding_model():
    """
    Lädt BAAI/bge-m3 via sentence-transformers.

    Nutzt CUDA nur, wenn torch CUDA sieht.
    Falls kein NVIDIA-Docker/GPU vorhanden ist, läuft es automatisch auf CPU.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("Lade Embedding-Modell '%s' auf '%s' ...", MODEL_NAME, device)

        model = SentenceTransformer(MODEL_NAME, device=device)

        log.info("Embedding-Modell geladen.")
        return model, device

    except ImportError:
        log.error(
            "sentence-transformers nicht installiert. "
            "Bitte: pip install sentence-transformers"
        )
        raise
# ------------------------------------------------------------
# Embedding-Modell einmal laden und wiederverwenden
# ------------------------------------------------------------

# Prozessweiter Modell-Cache
_MODEL = None
_DEVICE = None


def get_embedding_model():
    global _MODEL, _DEVICE

    if _MODEL is None:
        _MODEL, _DEVICE = load_embedding_model()

    return _MODEL, _DEVICE

def compute_embeddings(model, texts: list[str], device: str) -> list[list[float]]:
    """
    Berechnet L2-normalisierte Embeddings in Batches.

    Normalisierung ist erforderlich, da die Collections die Kosinus-Metrik
    verwenden. Bei CUDA-OOM wird auf CPU gewechselt und die Batchgröße
    halbiert; der Batch wird wiederholt.
    """
    import torch

    if not texts:
        return []

    all_embeddings: list[list[float]] = []
    batch_size = EMBEDDING_BATCH_SIZE

    index = 0
    while index < len(texts):
        batch = texts[index:index + batch_size]

        try:
            embeddings = model.encode(
                batch,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,  # Für Kosinus-Ähnlichkeit
            )
            all_embeddings.extend(embeddings.tolist())
            index += batch_size

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and device == "cuda":
                log.warning("GPU OOM. Wechsle auf CPU und reduziere Batchgröße.")
                torch.cuda.empty_cache()
                model = model.to("cpu")
                device = "cpu"
                batch_size = max(1, batch_size // 2)
            else:
                raise

    log.info(
        "Embeddings erzeugt: %d Texte → %d Vektoren",
        len(texts),
        len(all_embeddings),
    )

    return all_embeddings


# ── ChromaDB ─────────────────────────────────────────────────

def get_chroma_client() -> chromadb.HttpClient:
    """
    Verbindet sich mit ChromaDB (Heartbeat, bis MAX_RETRIES Versuche, 2 s Pause).

    Raises:
        RuntimeError: wenn kein Versuch erfolgreich war.
    """
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            client = chromadb.HttpClient(
                host=CHROMA_HOST,
                port=CHROMA_PORT,
            )
            client.heartbeat()
            return client

        except Exception as exc:
            last_error = exc
            wait_seconds = 2

            log.warning(
                "ChromaDB-Verbindung fehlgeschlagen "
                "(Versuch %d/%d): %s. Neuer Versuch in %d Sekunden.",
                attempt,
                MAX_RETRIES,
                exc,
                wait_seconds,
            )

            if attempt < MAX_RETRIES:
                time.sleep(wait_seconds)

    raise RuntimeError(
        f"ChromaDB nach {MAX_RETRIES} Versuchen nicht erreichbar"
    ) from last_error


def get_or_create_collection(client: chromadb.HttpClient, name: str):
    """Holt oder erstellt eine Collection mit cosine-Metrik."""
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def build_chunk_id(filename: str, chunk: dict[str, Any], idx: int) -> str:
    """
    Stabile Chunk-ID "<datei>_p<seite>_<typ>_<chunk_index>_<idx>".

    Ein Re-Import überschreibt per Upsert denselben Eintrag.
    """
    page = chunk.get("page", "0")
    typ = chunk.get("type", "text")
    chunk_index = chunk.get("chunk_index", idx)

    return f"{filename}_p{page}_{typ}_{chunk_index}_{idx}"


def prepare_batch(
    chunks: list[dict[str, Any]],
    filename: str,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """
    IDs, Texte und Metadaten für ChromaDB; leere Texte werden übersprungen.

    Metadatenwerte werden als String gespeichert.
    """
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for idx, chunk in enumerate(chunks):
        text = str(chunk.get("text", "")).strip()
        if not text:
            continue

        ids.append(build_chunk_id(filename, chunk, idx))
        documents.append(text)

        metadatas.append({
            "source": str(chunk.get("source", filename)),
            "filename": str(filename),
            "page": str(chunk.get("page", "0")),
            "type": str(chunk.get("type", "text")),
            "chunk_index": str(chunk.get("chunk_index", idx)),
            "ocr_confidence": str(chunk.get("ocr_confidence", "")),
        })

    return ids, documents, metadatas


def batch_upsert(
    collection,
    chunks: list[dict[str, Any]],
    embeddings: list[list[float]],
    filename: str,
) -> tuple[int, int, list[str], list[str], list[dict[str, Any]]]:
    """
    Speichert Chunks mit Embeddings in Batches zu CHROMA_BATCH_SIZE.

    Fehlgeschlagene Batches werden bis MAX_RETRIES wiederholt (Wartezeit
    5 s je Versuch, max. 20 s).

    Returns:
        (stored, failed, ids, documents, metadatas); die Listen enthalten
        nur erfolgreich gespeicherte Chunks für den anschließenden
        BM25-Abgleich in save_chunks_to_chromadb().

    Raises:
        ValueError: wenn Anzahl Texte und Embeddings abweichen.
    """
    ids, documents, metadatas = prepare_batch(chunks, filename)

    if not documents:
        return 0, 0, [], [], []

    if len(documents) != len(embeddings):
        raise ValueError(
            f"Embedding-Anzahl passt nicht: documents={len(documents)}, "
            f"embeddings={len(embeddings)}"
        )

    stored = 0
    failed = 0
    stored_ids: list[str] = []
    stored_docs: list[str] = []
    stored_metas: list[dict[str, Any]] = []

    for start in range(0, len(documents), CHROMA_BATCH_SIZE):
        end = start + CHROMA_BATCH_SIZE

        batch_ids = ids[start:end]
        batch_docs = documents[start:end]
        batch_metas = metadatas[start:end]
        batch_embeddings = embeddings[start:end]

        batch_ok = False
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                collection.upsert(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    documents=batch_docs,
                    metadatas=batch_metas,
                )
                stored += len(batch_ids)
                batch_ok = True
                break

            except Exception as exc:
                log.warning(
                    "ChromaDB Batch-Fehler %d/%d (%d-%d): %s",
                    attempt,
                    MAX_RETRIES,
                    start,
                    end,
                    exc,
                )

                if attempt < MAX_RETRIES:
                        wait_seconds = min(5 * attempt, 20)
                        log.info(
                            "Warte %d Sekunden vor erneutem Upsert-Versuch.",
                            wait_seconds,
                        )
                        time.sleep(wait_seconds)
                else:
                        failed += len(batch_ids)

        if batch_ok:
            stored_ids.extend(batch_ids)
            stored_docs.extend(batch_docs)
            stored_metas.extend(batch_metas)

    return stored, failed, stored_ids, stored_docs, stored_metas


# ── Done-Marker ───────────────────────────────────────────────

def set_done_marker(file_hash: str) -> None:
    """
    Legt <HASH_FOLDER>/<hash>.done an; 
    list_pdfs.py erkennt daran, dass eine PDF bereits verarbeitet wurde.
    """
    if not file_hash:
        return

    HASH_FOLDER.mkdir(parents=True, exist_ok=True)
    marker = HASH_FOLDER / f"{file_hash}.done"
    marker.write_text("done", encoding="utf-8")
    log.info("Done-Marker gesetzt: %s", marker.name)


# ── Hauptfunktion für api_server.py ───────────────────────────

def save_chunks_to_chromadb(
    chunks: list[dict[str, Any]],
    filename: str,
    file_hash: str,
) -> dict[str, Any]:
    """
    Zentrale Funktion für api_server.py.

    Text-Chunks und Tabellen-Chunks werden getrennt gespeichert:
    - petra_text_chunks
    - petra_tag_chunks
    Bei vollständigem Erfolg werden Done-Marker und
    Importstatus geschrieben.

    Returns:
        Ergebnis-Dict mit status ("success"/"partial"/"warning"),
        chunks_saved, chunks_failed und Konfigurationsangaben.
    """
    if not chunks:
        return {
            "status": "warning",
            "file": filename,
            "chunks_saved": 0,
            "chunks_failed": 0,
            "message": "Keine Chunks übergeben",
        }

    model, device = get_embedding_model()



    text_chunks = [
        c for c in chunks
        if c.get("type") != "table" and str(c.get("text", "")).strip()
    ]

    table_chunks = [
        c for c in chunks
        if c.get("type") == "table" and str(c.get("text", "")).strip()
    ]

    text_embeddings = compute_embeddings(
        model,
        [str(c["text"]) for c in text_chunks],
        device,
    ) if text_chunks else []

    table_embeddings = compute_embeddings(
        model,
        [str(c["text"]) for c in table_chunks],
        device,
    ) if table_chunks else []

    client = get_chroma_client()
    text_collection = get_or_create_collection(client, COLLECTION_TEXT)
    tag_collection = get_or_create_collection(client, COLLECTION_TAG)

    total_stored = 0
    total_failed = 0

    bm25_ids: list[str] = []
    bm25_docs: list[str] = []
    bm25_metas: list[dict[str, Any]] = []

    if text_chunks:
        stored, failed, s_ids, s_docs, s_metas = batch_upsert(
            text_collection,
            text_chunks,
            text_embeddings,
            filename,
        )
        total_stored += stored
        total_failed += failed
        bm25_ids.extend(s_ids)
        bm25_docs.extend(s_docs)
        bm25_metas.extend(s_metas)

    if table_chunks:
        stored, failed, s_ids, s_docs, s_metas = batch_upsert(
            tag_collection,
            table_chunks,
            table_embeddings,
            filename,
        )
        total_stored += stored
        total_failed += failed
        bm25_ids.extend(s_ids)
        bm25_docs.extend(s_docs)
        bm25_metas.extend(s_metas)

    # BM25-Sync GENAU EINMAL pro Datei (nicht pro 500er-ChromaDB-Batch,
    # siehe Docstring von batch_upsert) — nur fuer tatsaechlich in
    # ChromaDB erfolgreich gespeicherte Chunks.
    # v3-Sync: dieselben Chunks mit Kontextkopf in die v3-Collections (die der
    # Workflow seit Schritt 1 liest) und ins Produkt-Lexikon; der BM25-Index erhält
    # die Texte MIT Kopf – wie nach reindex_v3_contextual.py.
    if total_stored > 0 and os.environ.get("V3_SYNC", "true").lower() not in ("0", "false", "no"):
        try:
            from v3_sync import sync_datei
            _v3 = sync_datei(filename, model, device)
            if _v3.get("ids"):
                bm25_ids, bm25_docs, bm25_metas = _v3["ids"], _v3["docs"], _v3["metas"]
        except Exception as exc:  # noqa: BLE001 – der Import selbst darf daran nicht scheitern
            log.error("v3-Sync fehlgeschlagen fuer '%s': %s – nachholen mit: python3 v3_sync.py '%s'",
                      filename, exc, filename)

    if bm25_ids:
        try:
            n_new = index_chunks_hybrid(
                bm25_ids, bm25_docs, bm25_metas,
                bm25_index_path=BM25_INDEX_PATH,
            )
            log.info(
                "BM25-Index aktualisiert: %d neue Chunks (gesamt uebergeben: %d) fuer '%s'.",
                n_new, len(bm25_ids), filename,
            )
        except Exception as exc:  # noqa: BLE001 — BM25-Sync darf ChromaDB-Erfolg nicht zunichtemachen
            log.error(
                "BM25-Sync fehlgeschlagen fuer '%s': %s. ChromaDB-Daten sind "
                "trotzdem gespeichert; BM25 muss ggf. per "
                "backfill_bm25_from_chroma.py nachgetragen werden.",
                filename, exc,
            )

    if total_failed == 0:
        set_done_marker(file_hash)
    
    
        write_import_status(
            filename=filename,
            chunks=chunks,
            stored=total_stored,
            failed=total_failed,
    )

    return {
        "status": "success" if total_failed == 0 else "partial",
        "file": filename,
        "chunks_saved": total_stored,
        "chunks_failed": total_failed,
        "text_chunks": len(text_chunks),
        "table_chunks": len(table_chunks),
        "text_collection": COLLECTION_TEXT,
        "tag_collection": COLLECTION_TAG,
        "embedding_model": MODEL_NAME,
        "device": device,
    }


# ── Optionaler CLI-Test über stdin ────────────────────────────

def main() -> None:
    """
    Optionaler Direktaufruf:

    echo '{"chunks":[...],"filename":"x.pdf","hash":"abc"}' | python save_to_chromadb.py
    """
    raw = sys.stdin.read()

    if not raw.strip():
        log.error("Keine JSON-Daten über stdin erhalten.")
        sys.exit(1)

    try:
        data = json.loads(raw)
        if isinstance(data, str):
            data = json.loads(data)
    except json.JSONDecodeError as exc:
        log.error("Ungültiger JSON-Input: %s", exc)
        sys.exit(1)

    chunks = data.get("chunks", [])
    filename = data.get("filename", "unknown.pdf")
    file_hash = data.get("hash") or hashlib.md5(filename.encode()).hexdigest()

    result = save_chunks_to_chromadb(chunks, filename, file_hash)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()