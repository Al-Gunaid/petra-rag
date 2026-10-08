"""
views/evaluation.py — Seite „Evaluation" des PETRA-RAG-Frontends.

ERSETZT die vorherige, regelbasierte Evaluationsseite vollständig. Führt
den Weidmüller-Testdatensatz (75 Fälle) gegen die echte Pipeline
(POST /query) aus und bewertet die Antworten mit den vier
RAGAs-Metriken (Faithfulness, Answer Relevancy, Context Precision,
Context Recall) statt per Stichwort-Abgleich.

Bedienung:
  * Auswahl einer Teilmenge (Schwierigkeit, Fragetyp, Negativfälle)
  * Lauf starten — Ergebnisse erscheinen fortlaufend, nicht erst am Ende
  * frühere Läufe laden und vergleichen
  * Einzelfälle inkl. Kontext, Quellen und Metrik-Hinweisen ansehen
  * Export als Excel (vier Sheets, wie im Skript evaluation/ragas_evaluation.py
    im Projekt-Root — dieselbe excel_report.build_excel-Funktion)

Ein voller Lauf ist wegen der zusätzlichen RAGAs-LLM-Aufrufe je Fall
spürbar langsamer als der alte, rein regelbasierte Lauf. Nach jedem Fall
wird gespeichert; ein Abbruch kostet höchstens den laufenden Fall.
"""

from __future__ import annotations

from datetime import datetime
from io import BytesIO

import pandas as pd
import streamlit as st

from config import CONFIG
from services import evaluation as ev
from services.excel_report import build_excel


def render() -> None:
    st.title("Evaluation")
    st.caption("Weidmüller-Testdatensatz · 75 Fälle · RAGAs-Metriken")

    try:
        faelle = ev.lade_testfaelle()
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"Testfälle nicht ladbar: {exc}")
        st.info("Erwartet wird `evaluation/testfaelle.json` im Frontend-Verzeichnis.")
        return

    tab_lauf, tab_ergebnis, tab_details = st.tabs(
        ["Lauf starten", "Auswertung", "Einzelfälle"]
    )

    # ── Lauf starten ──────────────────────────────────────────────────────
    with tab_lauf:
        st.subheader("Auswahl")

        spalte_a, spalte_b = st.columns(2)
        with spalte_a:
            schwierigkeiten = sorted({f.schwierigkeit for f in faelle})
            gewaehlt_schwer = st.multiselect(
                "Schwierigkeit", schwierigkeiten, default=schwierigkeiten
            )
        with spalte_b:
            typen = sorted({f.fragetyp for f in faelle})
            gewaehlt_typ = st.multiselect("Fragetyp", typen, default=typen)

        auswahl = [
            f for f in faelle
            if f.schwierigkeit in gewaehlt_schwer and f.fragetyp in gewaehlt_typ
        ]

        spalte_c, spalte_d = st.columns(2)
        with spalte_c:
            latenz_vergleich = st.checkbox(
                "Latenzvergleich: kalt vs. warm (Cache)",
                value=False,
                help="Misst pro Fall zwei Aufrufe: Cache wird vor dem ersten Aufruf "
                     "geleert (kalt), der zweite, identische Aufruf direkt danach "
                     "sollte den Cache treffen (warm). Verdoppelt die Pipeline-Zeit, "
                     "nicht die RAGAs-Kosten — der LLM-Judge bewertet nur die kalte Antwort.",
            )
        with spalte_d:
            cache_leeren = st.checkbox(
                "CAG-Cache vor dem Lauf leeren",
                value=True,
                disabled=latenz_vergleich,
                help="Bei aktivem Latenzvergleich wird der Cache ohnehin vor "
                     "JEDEM Fall geleert — diese Option ist dann wirkungslos."
                     if latenz_vergleich else
                     "Ohne Leeren messen Wiederholungsläufe den Cache statt der Pipeline.",
            )

        ragas_ueberspringen = st.checkbox(
            "Nur Pipeline-Rohdaten sammeln (RAGAs überspringen)",
            value=False,
            help="Deutlich schneller — nützlich, um zuerst zu prüfen, ob die "
                 "Pipeline erreichbar ist, bevor der LLM-Judge mitläuft.",
        )

        st.caption(
            f"Judge: {CONFIG.ragas.judge_provider} · "
            f"Modell: {CONFIG.ragas.judge_model if CONFIG.ragas.judge_provider == 'ollama' else CONFIG.ragas.openai_model} "
            "· konfigurierbar über Umgebungsvariablen (siehe evaluation/.env.example)."
        )

        st.info(
            f"{len(auswahl)} Fälle ausgewählt"
            f"{' · je 2 Pipeline-Aufrufe (kalt+warm)' if latenz_vergleich else ''}"
            f"{' sowie vier zusätzliche LLM-Judge-Aufrufe je Fall' if not ragas_ueberspringen else ''}"
            " — das Browserfenster muss geöffnet bleiben; Zwischenstände werden nach "
            "jedem Fall gespeichert."
        )

        if st.button("Lauf starten", type="primary", disabled=not auswahl):
            lauf_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            balken = st.progress(0.0, text="Start …")
            live = st.empty()
            zaehler = {"verifiziert": 0, "nicht_verifiziert": 0, "fehler": 0}

            for i, erg in enumerate(
                ev.lauf_ausfuehren(
                    CONFIG.endpoints.api, auswahl, lauf_id,
                    cache_leeren=cache_leeren,
                    ragas_ueberspringen=ragas_ueberspringen,
                    latenz_vergleich=latenz_vergleich,
                ),
                start=1,
            ):
                if erg.fehler:
                    zaehler["fehler"] += 1
                elif erg.verified:
                    zaehler["verifiziert"] += 1
                else:
                    zaehler["nicht_verifiziert"] += 1

                balken.progress(
                    i / len(auswahl),
                    text=f"Fall {i}/{len(auswahl)} · "
                         f"{zaehler['verifiziert']} verifiziert · "
                         f"{zaehler['nicht_verifiziert']} offen · "
                         f"{zaehler['fehler']} Fehler",
                )
                metrik_text = " · ".join(
                    f"{label}={getattr(erg, key):.2f}" if isinstance(getattr(erg, key), (int, float)) else f"{label}=N/A"
                    for key, label in ev.METRIKEN
                ) if not ragas_ueberspringen else "RAGAs übersprungen"
                latenz_text = f"{erg.latency_ms} ms"
                if latenz_vergleich and erg.latency_ms_warm is not None:
                    latenz_text += (
                        f" (kalt) → {erg.latency_ms_warm} ms (warm"
                        f"{', Cache-Hit' if erg.cache_hit_warm else ', kein Cache-Hit'})"
                    )
                live.markdown(
                    f"**Fall {erg.nr}** · {latenz_text} · {erg.query_class or '—'}  \n"
                    f"{(erg.antwort or erg.fehler or '')[:200]}  \n"
                    f"*{metrik_text}*"
                )

            st.session_state["eval_lauf_id"] = lauf_id
            st.success(f"Lauf {lauf_id} abgeschlossen.")

    # ── Auswertung ────────────────────────────────────────────────────────
    with tab_ergebnis:
        laeufe = ev.verfuegbare_laeufe()
        if not laeufe:
            st.info("Noch kein Lauf vorhanden.")
            return

        vorauswahl = st.session_state.get("eval_lauf_id")
        index = laeufe.index(vorauswahl) if vorauswahl in laeufe else 0
        lauf_id = st.selectbox("Lauf", laeufe, index=index)
        ergebnisse = ev.lade_lauf(lauf_id)

        if not ergebnisse:
            st.warning("Lauf enthält keine Ergebnisse.")
            return

        k = ev.kennzahlen(ergebnisse)

        st.subheader("RAGAs-Metriken (Durchschnitt)")
        m1, m2, m3, m4 = st.columns(4)
        for spalte, (key, _label) in zip((m1, m2, m3, m4), ev.METRIKEN):
            info = k[key]
            wert = f"{info['mean']:.2f}" if info["mean"] is not None else "N/A"
            spalte.metric(info["label"], wert, f"n={info['n']}/{k['anzahl']}")

        st.subheader("Systemkennzahlen")
        s1, s2, s3 = st.columns(3)
        s1.metric("Verifiziert (System)", f"{k['verified_quote']:.0f} %")
        s2.metric("Latenz Median (kalt/Standard)", f"{k['latenz_median']} ms")
        s3.metric("NF-03 erfüllt (≤ 5 s)", f"{k['nf03_quote']:.0f} %")

        if k.get("latenzvergleich"):
            lv = k["latenzvergleich"]
            st.subheader("Latenzvergleich: kalt vs. warm (Cache)")
            l1, l2, l3, l4 = st.columns(4)
            l1.metric("Kalt (Mittelwert)", f"{lv['kalt_mean']} ms" if lv["kalt_mean"] else "—")
            l2.metric("Warm (Mittelwert)", f"{lv['warm_mean']} ms")
            l3.metric("Speedup-Faktor", f"{lv['speedup_faktor']}×" if lv["speedup_faktor"] else "—")
            l4.metric("Warm-Cache-Trefferquote", f"{lv['warm_cache_hit_quote']:.0f} %")
            st.caption(
                f"Basis: {lv['n']} Fälle mit gültiger Warm-Messung · "
                f"Median kalt {lv['kalt_median']} ms · Median warm {lv['warm_median']} ms. "
                "Eine Trefferquote unter 100 % ist normal — CAG cacht laut Architektur "
                "nur eine begrenzte Anzahl Dokumente (Top-20), nicht jede Anfrage trifft "
                "also zwangsläufig den Cache."
            )
            chart_df = pd.DataFrame({
                "Latenz (ms)": [lv["kalt_mean"] or 0, lv["warm_mean"]],
            }, index=["Kalt", "Warm"])
            st.bar_chart(chart_df)

        if k["fehler"]:
            st.warning(f"{k['fehler']} Fälle mit technischem Fehler — siehe Einzelfälle.")

        for key, _label in ev.METRIKEN:
            if k[key]["n"] < k["anzahl"]:
                st.caption(
                    f"⚠ {k[key]['label']}: nur {k[key]['n']} von {k['anzahl']} Fällen bewertbar "
                    "(z. B. korrekt verweigerte Negativfälle ohne Kontext, oder RAGAs nicht "
                    "erreichbar — siehe Einzelfälle für den jeweiligen Grund)."
                )
        
        st.divider()
        st.subheader("Beantwortbarkeit (F-13 / Kap. 4.7.2)")
        seg = ev.kennzahlen_segmentiert(faelle, ergebnisse)

        b1, b2 = st.columns(2)
        with b1:
            st.caption(f"Beantwortbare Fälle (n={seg['beantwortbar'].get('anzahl', 0)}) — gegen Architektur-Zielwerte")
            ziel_tabelle = []
            ziele_je_metrik = {
                "ragas_faithfulness": {"Stufe 1": 0.78, "Stufe 2 (NF-04)": 0.85},
                "ragas_context_precision": {"Stufe 1": 0.74, "Stufe 2": 0.80},
                "ragas_answer_relevancy": {"Stufe 2": 0.80},
            }
            for key, label in ev.METRIKEN:
                info = seg["beantwortbar"].get(key, {})
                mean = info.get("mean")
                for stufe, ziel in ziele_je_metrik.get(key, {}).items():
                    erreicht = "✓" if isinstance(mean, (int, float)) and mean >= ziel else "✗"
                    ziel_tabelle.append({
                        "Metrik": label, "Ist": mean, "Ziel": ziel, "Stufe": stufe, "Erreicht": erreicht,
                    })
            st.dataframe(pd.DataFrame(ziel_tabelle), hide_index=True, use_container_width=True)

        with b2:
            ab = seg["nicht_beantwortbar"]
            if ab:
                st.caption(f"Nicht-beantwortbare Fälle (n={ab['anzahl']}) — korrekt abgelehnt statt halluziniert?")
                st.metric("Korrekt abgelehnt (F-13)", f"{ab['korrekt_abgelehnt_quote']:.0f} %",
                          f"{ab['korrekt_abgelehnt']}/{ab['anzahl']}")
                if ab["vermutlich_halluziniert_nr"]:
                    st.warning(
                        f"Vermutlich halluziniert statt abgelehnt: Testfall-Nr. "
                        f"{', '.join(str(n) for n in ab['vermutlich_halluziniert_nr'])} "
                        "(siehe Einzelfälle)."
                    )
            else:
                st.caption("Keine nicht-beantwortbaren Testfälle im Lauf.")
        st.divider()
        st.subheader("Nach Gruppe")

        gruppen_metrik = st.selectbox(
            "Metrik für Gruppenansicht",
            options=[k for k, _ in ev.METRIKEN],
            format_func=lambda k: dict(ev.METRIKEN)[k],
        )
        g1, g2 = st.columns(2)
        with g1:
            st.caption("Schwierigkeit")
            st.dataframe(
                pd.DataFrame(ev.nach_gruppe(faelle, ergebnisse, "schwierigkeit", gruppen_metrik)),
                hide_index=True, use_container_width=True,
            )
        with g2:
            st.caption("Fragetyp")
            st.dataframe(
                pd.DataFrame(ev.nach_gruppe(faelle, ergebnisse, "fragetyp", gruppen_metrik)),
                hide_index=True, use_container_width=True,
            )

        st.divider()
        st.subheader("Export")
        st.download_button(
            "Auswertung als Excel (.xlsx)",
            data=_als_excel_bytes(faelle, ergebnisse, lauf_id),
            file_name=f"ragas_evaluation_{lauf_id}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    # ── Einzelfälle ───────────────────────────────────────────────────────
    with tab_details:
        laeufe = ev.verfuegbare_laeufe()
        if not laeufe:
            st.info("Noch kein Lauf vorhanden.")
            return

        vorauswahl = st.session_state.get("eval_lauf_id")
        index = laeufe.index(vorauswahl) if vorauswahl in laeufe else 0
        lauf_id = st.selectbox("Lauf", laeufe, index=index, key="detail_lauf")
        ergebnisse = ev.lade_lauf(lauf_id)
        per_nr = {f.nr: f for f in faelle}

        nur_auffaellige = st.checkbox(
            "Nur Fälle mit Fehler, N/A-Metrik oder Negativfall zeigen", value=False
        )

        for erg in sorted(ergebnisse, key=lambda e: e.nr):
            fall = per_nr[erg.nr]
            auffaellig = bool(
                erg.fehler or fall.nicht_beantwortbar or fall.data_quality_flags
                or not erg.hat_metriken
            )
            if nur_auffaellige and not auffaellig:
                continue

            symbol = "⚠" if erg.fehler else ("•" if fall.nicht_beantwortbar else "✓")
            titel = f"{symbol}  Fall {erg.nr} · {fall.schwierigkeit} · {fall.fragetyp}"

            with st.expander(titel):
                st.markdown(f"**Frage:** {fall.frage}")
                if fall.data_quality_flags:
                    st.caption(f"Testfall-Hinweis: {', '.join(fall.data_quality_flags)}")

                links, rechts = st.columns(2)
                with links:
                    st.caption("Referenzantwort")
                    st.info(fall.referenzantwort)
                with rechts:
                    st.caption(f"Systemantwort · {erg.query_class or '—'} · {erg.latency_ms} ms")
                    if erg.latency_ms_warm is not None:
                        st.caption(
                            f"Warm-Messung: {erg.latency_ms_warm} ms "
                            f"({'Cache-Hit' if erg.cache_hit_warm else 'kein Cache-Hit'})"
                        )
                    if erg.fehler:
                        st.error(erg.fehler)
                    else:
                        st.write(erg.antwort or "—")

                st.caption("RAGAs-Metriken")
                mm = st.columns(4)
                for spalte, (key, label) in zip(mm, ev.METRIKEN):
                    wert = getattr(erg, key)
                    hinweis = getattr(erg, f"{key.replace('ragas_', '')}_hinweis")
                    spalte.metric(label, f"{wert:.2f}" if isinstance(wert, (int, float)) else "N/A")
                    if hinweis:
                        spalte.caption(hinweis)

                if erg.faithfulness_delta is not None:
                    st.caption(
                        f"Faithfulness-Delta (System − RAGAs): {erg.faithfulness_delta:+.2f} "
                        "— große Abweichung deutet auf Kontextverlust durch die "
                        "220-Zeichen-Kürzung der Quellen hin (siehe README)."
                    )

                if erg.kontext:
                    st.caption(f"Kontext ({len(erg.kontext)} Ausschnitt(e), je max. 220 Zeichen)")
                    for j, ausschnitt in enumerate(erg.kontext, start=1):
                        st.text(f"[{j}] {ausschnitt}")
                else:
                    st.caption("Kein Kontext zurückgegeben.")

                if erg.quellen:
                    st.caption("Quellen")
                    st.dataframe(
                        pd.DataFrame([
                            {
                                "Dokument": q.get("dateiname", "—"),
                                "Seite": q.get("seite"),
                                "Artikelnummer": q.get("artikelnummer"),
                                "Score": q.get("score"),
                            } for q in erg.quellen
                        ]),
                        hide_index=True, use_container_width=True,
                    )


def _als_excel_bytes(faelle: list, ergebnisse: list, lauf_id: str) -> bytes:
    """Baut die Excel-Datei im Speicher (kein Zwischenfile nötig)."""
    per_nr = {f.nr: f for f in faelle}
    rows = []
    for e in ergebnisse:
        fall = per_nr.get(e.nr)
        rows.append({
            "nr": e.nr,
            "produktbereich": fall.produktbereich if fall else "",
            "produkt": fall.produkt if fall else "",
            "artikelnummer": fall.artikelnummer if fall else "",
            "schwierigkeit": fall.schwierigkeit if fall else "",
            "fragetyp": fall.fragetyp if fall else "",
            "nicht_beantwortbar": fall.nicht_beantwortbar if fall else False,
            "frage": e.frage,
            "referenzantwort": e.referenzantwort,
            "generierte_antwort": e.antwort,
            "kontext_anzahl": len(e.kontext),
            "kontext": "\n---\n".join(e.kontext),
            "quellen": " | ".join(
                f"{q.get('dateiname','')} S.{q.get('seite','')}" for q in e.quellen
            ),
            "query_class": e.query_class,
            "latency_ms": e.latency_ms,
            "cache_hit": e.cache_hit,
            "latency_ms_warm": e.latency_ms_warm,
            "cache_hit_warm": e.cache_hit_warm,
            "system_faithfulness_score": e.system_faithfulness_score,
            "verified": e.verified,
            "fallback_reason": e.fallback_reason,
            "ragas_faithfulness": e.ragas_faithfulness,
            "faithfulness_delta": e.faithfulness_delta,
            "ragas_answer_relevancy": e.ragas_answer_relevancy,
            "ragas_context_precision": e.ragas_context_precision,
            "ragas_context_recall": e.ragas_context_recall,
            "faithfulness_hinweis": e.faithfulness_hinweis,
            "answer_relevancy_hinweis": e.answer_relevancy_hinweis,
            "context_precision_hinweis": e.context_precision_hinweis,
            "context_recall_hinweis": e.context_recall_hinweis,
            "data_quality_flags": ", ".join(fall.data_quality_flags) if fall else "",
            "fehler": e.fehler,
        })

    buffer = BytesIO()
    tmp_path = f"/tmp/ragas_evaluation_{lauf_id}.xlsx"
    build_excel(rows, tmp_path, run_meta={
        "Lauf-ID": lauf_id,
        "Zeitpunkt": datetime.now().isoformat(timespec="seconds"),
        "Judge-Provider": CONFIG.ragas.judge_provider,
        "Anzahl Testfälle": len(ergebnisse),
    })
    with open(tmp_path, "rb") as f:
        buffer.write(f.read())
    return buffer.getvalue()
