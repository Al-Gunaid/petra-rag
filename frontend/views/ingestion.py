"""views/ingestion.py — PDFs hochladen, Import steuern, Fortschritt sehen.

Produktivpfad: PDF-Upload in den bestehenden Watch-Folder + Start des
bestehenden n8n-Workflows "PDF-Pipline-Parsing" über dessen Webhook-Trigger.
Der Fortschritt wird über GET /status-summary als Delta der Chunk-Zahlen
(Text/Tabellen) sichtbar gemacht — die Dokumentenzahl wird bewusst nicht
gepollt, da der zuständige Endpunkt GET /status bei großen Beständen
langsam sein kann (siehe api_server.py, Timeout 1800 s).

Entwicklungs-Fallback: Direktmodus über die bestehenden FastAPI-Endpunkte
in exakt der Reihenfolge des n8n-Workflows (keine neue Businesslogik),
falls n8n/Webhook nicht erreichbar ist.
"""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from api.client import PetraApiClient
from config import CONFIG
from services.clients import get_petra_client
from services.ingestion import (
    poll_status_summary,
    run_direct,
    save_uploaded_files,
    try_start_via_n8n,
)
from services.system_status import invalidate_status_caches
from ui.components import pipeline_progress


def _render_pdf_table(payload) -> int:
    items = []
    if isinstance(payload, dict):
        items = (payload.get("pdfs") or payload.get("files")
                 or payload.get("dateien") or [])
    if not items:
        st.caption("Keine PDFs im Watch-Folder gefunden.")
        return 0
    rows = [{
        "Datei": it.get("filename", it.get("name", "?")),
        "Neu": "🆕 ja" if it.get("neu", it.get("new", False)) else "✔️ bereits importiert",
        "Gültig": "ja" if it.get("valid", True) else "nein",
        "SHA-256": str(it.get("hash", ""))[:12] + "…",
    } for it in items if isinstance(it, dict)]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    return sum(1 for it in items
               if isinstance(it, dict) and it.get("neu", it.get("new", False)))


def _render_upload(api: PetraApiClient) -> None:
    st.subheader("1 · PDFs hochladen")
    st.caption(
        f"Dateien werden in den bestehenden Watch-Folder (`{CONFIG.watch_folder}`, "
        "UC-05) geschrieben — denselben Pfad, den die Ingestion-Pipeline "
        "bereits überwacht. Es wird kein neuer Upload-Mechanismus erfunden. "
        "Nach erfolgreichem Upload wird der Import automatisch über den "
        "n8n-Webhook gestartet."
    )
    files = st.file_uploader(
        "PDF-Dateien auswählen", type=["pdf"], accept_multiple_files=True,
        key="pdf_uploader",
    )
    if st.button("⬆️ In Watch-Folder übernehmen & Import starten",
                 disabled=not files, type="primary"):
        results = save_uploaded_files(files)
        ok_count = sum(1 for r in results if r.ok)
        for r in results:
            (st.success if r.ok else st.error)(f"**{r.filename}** — {r.detail}")

        if ok_count:
            st.toast(f"{ok_count} Datei(en) im Watch-Folder abgelegt.", icon="✅")
            with st.spinner("Watch-Folder wird geprüft (Hash-Check aller "
                            "PDFs, kann etwas dauern) …"):
                st.session_state["pdf_listing"] = api.list_pdfs()

            # Automatischer Start des Imports nach erfolgreichem Upload
            # (auf ausdrücklichen Wunsch: kein zusätzlicher Klick nötig).
            _start_via_webhook(api, auto=True)
        else:
            st.warning("Kein Import gestartet — keine der ausgewählten "
                       "Dateien konnte im Watch-Folder abgelegt werden.")

    st.divider()
    st.subheader("2 · Watch-Folder prüfen")
    st.caption("Listet alle PDFs im Watch-Folder inkl. SHA-256-Duplikat-Check "
               "(GET /list-pdfs). Bei vielen/großen Dateien kann dies bis zu "
               "einer Minute dauern.")
    if st.button("🔍 Neue PDFs prüfen", type="secondary"):
        with st.spinner("Watch-Folder wird geprüft …"):
            st.session_state["pdf_listing"] = api.list_pdfs()
    listing = st.session_state.get("pdf_listing")
    if listing is not None:
        if listing.ok:
            new_count = _render_pdf_table(listing.data)
            st.caption(f"Neue Dokumente: **{new_count}**")
        elif listing.timeout:
            st.error(
                f"PDF-Erkennung fehlgeschlagen: {listing.error}. "
                f"Das Timeout wurde bereits auf {CONFIG.timeout_list_pdfs:.0f}s "
                "angehoben — bei sehr vielen/großen Dateien im Watch-Folder "
                "kann der Hash-Check trotzdem länger dauern. Erneut versuchen "
                "oder Watch-Folder-Größe prüfen."
            )
        else:
            st.error(f"PDF-Erkennung fehlgeschlagen: {listing.error}")


def _render_progress_poll(api: PetraApiClient) -> None:
    """Zeigt den Fortschritt nach Webhook-Start als Delta der Chunk-Zahlen.

    Nutzt bewusst NUR GET /status-summary (schnell, laut api_server.py nur
    ein ChromaDB-Count je Collection). GET /status (chromadb_status.py)
    liefert zwar zusätzlich die Dokumentenzahl, kann laut Server-Timeout
    (1800 s) bei großen Beständen aber sehr langsam sein und eignet sich
    daher nicht für wiederholtes Polling während eines laufenden Imports.

    Cache-Invalidierung (gezielt statt bei jedem Poll): Diese Anzeige selbst
    ruft IMMER live ab (kein st.cache_data — der Fortschritt muss aktuell
    sein). Die an ANDERER Stelle zwischengespeicherten Bestandsdaten
    (Sidebar, Dashboard, Chat-Filter, siehe services/system_status.py)
    werden nur dann invalidiert, wenn sich die Gesamt-Chunk-Zahl gegenüber
    dem zuletzt GESEHENEN Wert tatsächlich verändert hat — nicht bei jedem
    5-Sekunden-Poll erneut, auch wenn sich zwischenzeitlich nichts geändert
    hat (Vorgabe: keine unnötigen Cache-Invalidierungen).
    """
    baseline = st.session_state.get("ingest_baseline")
    started_at = st.session_state.get("ingest_started_at")
    if baseline is None:
        return

    r = poll_status_summary(api)
    with st.container(border=True):
        c1, c2, c3 = st.columns([2, 2, 1])
        if r.ok and isinstance(r.data, dict):
            text_chunks = r.data.get("text_chunks")
            tag_chunks = r.data.get("tag_chunks")
            gesamt = r.data.get("gesamt_chunks")
            d_text = (text_chunks - baseline.get("text_chunks", 0)
                      ) if isinstance(text_chunks, int) else None
            d_tag = (tag_chunks - baseline.get("tag_chunks", 0)
                     ) if isinstance(tag_chunks, int) else None
            c1.metric("Text-Chunks gesamt", text_chunks if text_chunks is not None else "—",
                       delta=d_text if d_text else None)
            c2.metric("Tabellen-Chunks (TAG) gesamt", tag_chunks if tag_chunks is not None else "—",
                       delta=d_tag if d_tag else None)
            if gesamt is not None:
                st.caption(f"Chunks gesamt: {gesamt:,}".replace(",", "."))

            # Gezielte Invalidierung nur bei tatsächlicher Änderung seit
            # dem letzten Poll (last-seen-Vergleich in st.session_state).
            last_seen = st.session_state.get("ingest_last_seen_total")
            if gesamt is not None and gesamt != last_seen:
                st.session_state["ingest_last_seen_total"] = gesamt
                if last_seen is not None:  # nicht beim allerersten Poll (== Baseline)
                    invalidate_status_caches()
        else:
            c1.warning(f"Status derzeit nicht abrufbar: {r.error}")
        if c3.button("🔄", help="Status jetzt aktualisieren"):
            st.rerun()
        st.caption(f"Workflow gestartet um {started_at}. Die Dokumentenzahl "
                   "wird hier bewusst nicht gepollt (GET /status kann bei "
                   "großen Beständen langsam sein) — final sichtbar im "
                   "Dashboard nach Abschluss. Details zum Ausführungsstatus "
                   "zusätzlich im n8n-UI (Port 5678).")

    auto = st.checkbox("Automatisch alle 5 s aktualisieren (max. 24 Versuche)",
                        key="ingest_autopoll")
    if auto:
        n = st.session_state.get("ingest_poll_count", 0)
        if n < 24:
            st.session_state["ingest_poll_count"] = n + 1
            time.sleep(5)
            st.rerun()
        else:
            st.info("Automatische Aktualisierung angehalten — bitte manuell "
                    "mit 🔄 weiter prüfen oder die Checkbox erneut aktivieren.")

    if st.button("Fortschrittsanzeige schließen"):
        # Abschließende Invalidierung als Sicherheitsnetz: falls der
        # Workflow z. B. nur bereits vorhandene (duplikate) Dateien
        # verarbeitet hat, gab es evtl. keine Chunk-Delta-Änderung, aber
        # ggf. dennoch aktualisierte Dokument-Metadaten (Re-Import).
        invalidate_status_caches()
        for k in ("ingest_baseline", "ingest_started_at", "ingest_poll_count",
                  "ingest_last_seen_total"):
            st.session_state.pop(k, None)
        st.rerun()


def _start_via_webhook(api: PetraApiClient, auto: bool = False) -> bool:
    """Startet den n8n-Workflow per Webhook und setzt den Polling-Zustand.

    Gemeinsam genutzt vom automatischen Start nach Upload UND vom manuellen
    Button — vermeidet doppelte Logik. Gibt True zurück, wenn der Start
    erfolgreich angestoßen wurde.
    """
    res = try_start_via_n8n()
    if res.ok:
        prefix = "Import automatisch gestartet" if auto else "Workflow gestartet"
        st.success(f"{prefix}. Fortschritt unten sichtbar; Details zusätzlich "
                   "im n8n-UI (Port 5678).")
        baseline_r = poll_status_summary(api)
        baseline = {"text_chunks": 0, "tag_chunks": 0}
        if baseline_r.ok and isinstance(baseline_r.data, dict):
            baseline = {
                "text_chunks": baseline_r.data.get("text_chunks", 0) or 0,
                "tag_chunks": baseline_r.data.get("tag_chunks", 0) or 0,
            }
        st.session_state["ingest_baseline"] = baseline
        st.session_state["ingest_started_at"] = time.strftime("%H:%M:%S")
        st.session_state["ingest_poll_count"] = 0
        # Startwert für den Last-Seen-Vergleich in _render_progress_poll —
        # verhindert eine Invalidierung beim allerersten Poll (== Baseline,
        # noch keine echte Änderung).
        st.session_state["ingest_last_seen_total"] = (
            baseline.get("text_chunks", 0) + baseline.get("tag_chunks", 0)
        )
        return True

    if res.not_found:
        st.warning(
            "Der Webhook des n8n-Workflows ist nicht erreichbar (404). "
            "Prüfen Sie, ob der Workflow **aktiv** ist und der "
            f"Webhook-Pfad `{CONFIG.endpoints.n8n_ingest_webhook_path}` "
            "mit dem Trigger-Node übereinstimmt."
            + (" Alternativ: Direktmodus nutzen (Abschnitt 3)." if auto else "")
        )
    else:
        st.error(f"{'Automatischer ' if auto else ''}Start fehlgeschlagen: {res.error}"
                 + (" — Alternativ: Direktmodus nutzen (Abschnitt 3)." if auto else ""))
    return False


def render() -> None:
    st.title("Ingestion")
    st.caption(
        "Steuert die bestehende Pipeline: PDF-Erkennung → Parsing → OCR → "
        "Chunking → Embedding → Speicherung → Fertig. Dokumente können "
        "direkt hochgeladen ODER als Datei in den Watch-Folder gelegt "
        f"werden ({CONFIG.watch_folder}, UC-05)."
    )
    api = get_petra_client()

    _render_upload(api)

    # ── Schritt 3: Import (erneut) starten / Fallback ─────────
    st.subheader("3 · Import erneut starten / Fallback")
    st.caption(
        "Normalerweise nicht nötig — der Import startet nach dem Upload "
        "automatisch über den n8n-Webhook (Schritt 1). Diese Buttons sind "
        "für einen manuellen Neustart (z. B. nach einem Fehlversuch) oder "
        "den Entwicklungs-Fallback gedacht."
    )
    c1, c2 = st.columns(2)
    start_n8n = c1.button("▶️ Import über n8n-Workflow starten (Webhook)",
                          use_container_width=True)
    start_direct = c2.button("🛠️ Direktmodus (Entwicklungs-Fallback)",
                             use_container_width=True,
                             help="Nur verwenden, wenn n8n/Webhook nicht "
                                  "erreichbar ist — ruft die bestehenden "
                                  "FastAPI-Endpunkte in der Reihenfolge des "
                                  "n8n-Workflows direkt nacheinander auf.")

    progress_area = st.container()

    if start_n8n:
        _start_via_webhook(api, auto=False)

    if st.session_state.get("ingest_baseline") is not None:
        _render_progress_poll(api)

    if start_direct:
        with progress_area:
            slot = st.empty()
            with st.status("Import läuft (Direktmodus) …",
                           expanded=True) as box:
                run = None
                for run in run_direct(api):
                    with slot.container():
                        pipeline_progress(run)
                if run is not None and not run.error:
                    box.update(label="Import abgeschlossen ✅",
                               state="complete")
                else:
                    box.update(label="Import mit Fehlern beendet",
                               state="error")
            if run is not None:
                st.session_state["ingestion_run"] = run
                if run.error:
                    st.error(run.error)
                with st.expander("Protokoll des Laufs"):
                    st.code("\n".join(run.log) or "—", language=None)

    # letzten Lauf erneut anzeigen (Rerun-Festigkeit)
    if (not start_direct and st.session_state.get("ingestion_run")
            is not None):
        st.markdown("**Letzter Lauf (Direktmodus):**")
        pipeline_progress(st.session_state["ingestion_run"])

    # ── Schritt 4: Protokoll / Report ─────────────────────────
    st.subheader("4 · Import-Protokoll")
    if st.button("📄 CSV-Protokoll erzeugen (GET /export-report)"):
        res = api.export_report()
        if res.ok:
            st.success(f"{(res.data or {}).get('status', 'CSV exportiert')} — "
                       f"Ablage im Report-Verzeichnis des Ingestion-Service "
                       f"(REPORT_DIR).")
            st.json(res.data)
        else:
            st.error(f"Export fehlgeschlagen: {res.error}")
