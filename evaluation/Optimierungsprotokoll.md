# Optimierungsprotokoll PETRA-RAG (ab 01.10.2026)

Ziel: Ausgehend von v3.0 schrittweise verbessern. Jede Änderung muss durch eine Quelle begründet sein, wird einzeln gemessen und wird verworfen, wenn sie nicht hilft.

## Feste Messbedingungen (für alle Schritte gleich)
| Größe | Festlegung |
|---|---|
| Testset | testset_v2.xlsx, 75 Fälle (59 beantwortbar, 16 Negativfälle) |
| Judge | qwen2.5:32b, temperature 0, num_ctx 8192, OLLAMA_NUM_PARALLEL=1 |
| Kontextsicht | Prompt-Blöcke mit Quellen-Kopf („bloecke“) |
| Skript | evaluate_petra.py über run_eval_schritt.sh <Schritt> |
| Generator | llama3.1:8b (unverändert) |
| Backend | aktueller Stand (BM25-Index, Reranker-Modell, Preisagent); die Reihe misst Änderungen am Workflow |

## Regeln
1. Ein Schritt = eine Änderung, begründet durch einen Befund aus der Fehleranalyse des vorherigen Schritts und eine Quelle mit gemessenem Effekt.
2. Behalten nur, wenn die Zielmetrik über der Judge-Streuung steigt (gemessen: etwa 0,015) und keine andere Kernmetrik deutlich fällt; Vergleich je Fall (gepaart) mit Bootstrap-Konfidenzintervall.
3. Sonst zurücknehmen und im Protokoll als „verworfen“ führen.

## Schritte
| Schritt | Änderung | Quelle | F | CP | AR | CR | doc_hit | Entscheidung |
|---|---|---|---|---|---|---|---|---|
| s0 | v3.0 unverändert (nur Messausgabe) | – | 0,676 | 0,500 | 0,471 | 0,373 | 0,526 | Baseline (01.10.2026) |
| s1 | Dense + TAG lesen v3-Collections mit Kontextkopf (BM25 ist bereits kontextuell) | Anthropic 2024, Contextual Retrieval | 0,658 | 0,574 | 0,446 | 0,408 | 0,561 | behalten (CP +0,074; F/AR gepaart neutral: je 20 besser/20 schlechter) |
| s2 | Vierter Retrieval-Pfad: Produkt-Lexikon → Artikelnummer → Suche nur im Datenblatt (/product/retrieve), RRF-Quelle + Mindestquote | Poliakov & Shvai 2024 (Multi-Meta-RAG); Architektur Kap. 4.7.2 Stufe 2 | 0,748 | 0,612 | 0,514 | 0,493 | 0,737 | **behalten** (F +0,089, KI95 [+0,008; +0,169]) |
| (s3 alt) | Produktfilter in der Kontextauswahl (exakt aufgelöste Artikelnummer, K_MIN 2; Reranker gibt alle 30 bewerteten Chunks weiter) | Amiraz et al. 2025; Cuconasu et al. 2024; Akarsu et al. 2026 | – | – | – | – | – | **nicht eingesetzt** – Datenlage nach s2 widerspricht (s. u.) |
| s3 | Prompt-Reihenfolge nach Cross-Encoder-Score statt „Tabellen zuerst“ | Liu et al. 2024 (Lost in the Middle); Kap. 4.6.2; Es et al. 2024 (RAGAS-CP rangbewertet) | 0,720 | 0,628 | 0,497 | 0,484 | 0,754 | **verworfen** – zurück auf s2 |
| s4 | Rerank-Tiefe 30 → 50 (RRF topK 50, Cross-Encoder input 50; Dense/BM25 je 30 Treffer) | Akarsu et al. 2026 (T²-RAGBench-Ablation, Fig. 6) | 0,728 | 0,618 | 0,484 | 0,478 | 0,702 | **verworfen** – zurück auf s2 |
| s5 | HyDE nur Q3 (Kap. 4.2 [1a]) und nicht bei Artikelnummer / Zahl+Einheit | Architektur Kap. 4.2 [1a]; Akarsu et al. 2026; Gao et al. 2023 | 0,834 | 0,611 | 0,520 | 0,473 | 0,737 | **behalten** (F +0,091; davon ≈ +0,03 durch die Maßnahme) |

## s0 – Baseline v3.0 (01.10.2026)
- RAGAS (qwen2.5:32b, Sicht „bloecke“): F 0,676 (n=57) · CP 0,500 (n=59) · AR 0,471 (n=59; ohne „nicht belegt“ 0,577, n=39) · CR 0,373 (n=58)
- Deterministisch: doc_hit 0,526 · wert_recall 0,298 · beleg_quote 0,972 · Ablehnung Negativfälle 0,44 (n=16) · falsche Ablehnung 0,0
- Latenz Ø 10,2 s, Anteil ≤ 5 s: 0 % (NF-03 verfehlt)
- Befund Index: BM25 (/data/bm25_index.jsonl, 616 055 Chunks) trägt bereits den Kontextkopf [Produkt | Art.-Nr. | Abschnitt];
  Dense (petra_text_chunks_v2) und TAG (petra_tag_chunks) nicht. Kontextuelles Retrieval ist in v3.0 also nur zur Hälfte umgesetzt.

## s1 – kontextueller Index vollständig (geplant)
- Befund: Datenblatt des gefragten Produkts in 27 von 57 Fällen nicht im Kontext (doc_hit 0,526).
- Änderung: Dense → petra_text_chunks_v3, TAG → petra_tag_chunks_v3 (gleiche chunk_ids, Kopf wie im BM25-Index).
- Quelle: Anthropic (2024), Contextual Retrieval – Top-20-Fehlerrate −35 % (kontextuelle Embeddings), −49 % zusammen mit kontextuellem BM25.
- Messung: voller Lauf (75 Fälle), weil die Änderung jede Anfrage betrifft und die schwachen Fälle aus s0 bereits 53 von 75 ausmachen.

### Kurztest s1 (01.10.2026) – Befund für s2
- „Welche Schutzart hat die WDU 2.5?“: 1020000000_de.pdf jetzt an Rang 1 (s0: fehlte), aber Antwort „IP66“ aus 8000133629_de.pdf (fremdes Produkt).
- Ursachen in v3.0: CRAG-Auswahl ohne Produktbezug; Prompt-Komposition sortiert Tabellen vor Produkt-Treffer; Reranker gibt nur 8 von 30 bewerteten Chunks weiter.
- Bewusst NICHT in s1 korrigiert (eine Änderung je Schritt), sondern als s2 vorbereitet (patch_s2_produktfilter.py).

## s1 – Ergebnis (Lauf 01.10.2026, 19:54–22:10)
| Kennzahl | s0 | s1 | Δ |
|---|---|---|---|
| Faithfulness (n=57) | 0,676 | 0,658 | −0,018 |
| Context Precision (n=59) | 0,500 | 0,574 | **+0,074** |
| Answer Relevancy (n=59) | 0,471 | 0,446 | −0,025 |
| Context Recall (n=58) | 0,373 | 0,408 | +0,035 |
| doc_hit (n=57) | 0,526 | 0,561 | +0,035 |
| wert_recall (n=48) | 0,298 | 0,340 | +0,042 |
| beleg_quote | 0,972 (n=36) | 0,923 (n=39) | −0,049 |
| Ablehnung Negativfälle (n=16) | 0,44 | 0,69 | +0,25 |
| Latenz Ø | 10,2 s | 10,1 s | – |

Bewertung: Die Zielmetrik der Maßnahme (Retrieval: CP, CR, doc_hit) steigt deutlich über die Judge-Streuung (≈0,015);
Negativfälle werden deutlich öfter korrekt abgelehnt. F (−0,018) und AR (−0,025) liegen knapp über der Streuung →
gepaarter Vergleich je Fall mit vollständigem eval_s0.xlsx steht aus. Entscheidung: vorläufig behalten, s2 baut darauf auf.

Fehlerbild s1 (Einzelproduktfragen, n=44): in 37 Fällen stehen Chunks fremder Produkte im Kontext, nur 21 Fälle haben ≥ 2 Chunks
des Zielprodukts unter den 6–8 Kontext-Chunks. Fälle mit ≥ 2 Ziel-Chunks erreichen F 0,73 / CP 0,79 – Begründung für s2.

## Neuplanung nach s1 (02.10.2026) – Fehlergruppen der beantwortbaren Fälle (n=59)
| Gruppe | n | F | CP | AR | CR |
|---|---|---|---|---|---|
| A Einzelproduktfrage, ≥ 2 Chunks des Zielprodukts | 21 | 0,702 | 0,773 | 0,500 | 0,563 |
| B Einzelproduktfrage, 1 Chunk des Zielprodukts | 6 | 0,778 | 0,770 | 0,517 | 0,597 |
| C Einzelproduktfrage, Datenblatt fehlt | 16 | 0,574 | 0,404 | 0,426 | 0,286 |
| D Vergleich, Suche, allgemeine Frage, Preis | 15 | 0,645 | 0,417 | 0,348 | 0,226 |

- Ursache C: Varianten-Verdrängung (für „WDU 2.5“ kommen WDU 2.5 GN/WS/BL/…/1.5/ZR, nicht 1020000000; Fälle 1, 2, 3, 6, 7, 27).
  Ein Filter nach dem Retrieval hilft dort nicht → Reihenfolge geändert: erst Lexikon-Pfad (s2), dann Produktfilter (s3).
- Projektion: Erreicht C das Niveau von A, steigt CP gesamt auf ≈ 0,68 und F auf ≈ 0,69 – Stufe 1 (F ≥ 0,78, CP ≥ 0,74) wird mit
  Retrieval-Maßnahmen allein voraussichtlich nicht erreicht. Auch Gruppe A hat nur F 0,70 → Generator-/Prompt-Schritt nötig (s4).
- Weitere Kandidaten: s4 Prompt-Reihenfolge Produkt vor Tabelle (Liu et al. 2024, Lost in the Middle); s5 HyDE nur ohne Produktcode
  (Akarsu et al. 2026; spart einen LLM-Aufruf, Latenz Median 7,5 s, nur 1 von 75 Antworten ≤ 5 s).
- 17 Antworten „nicht belegt“ haben AR 0,24 (übrige 0,53) – betrifft das Stufe-2-Ziel AR.

## s2 – Ergebnis (Lauf 01./02.10.2026)
Gepaarter Vergleich (beantwortbare Fälle, Bootstrap 5000×, KI95):
| Kennzahl | s1 | s2 | Δ | KI95 | besser/schlechter (>0,05) |
|---|---|---|---|---|---|
| Faithfulness | 0,659 | 0,748 | **+0,089** | [+0,008; +0,169] | 21 / 12 |
| Context Precision | 0,574 | 0,612 | +0,038 | [−0,022; +0,103] | 15 / 9 |
| Answer Relevancy | 0,446 | 0,514 | +0,068 | [−0,015; +0,155] | 15 / 7 |
| Context Recall | 0,415 | 0,493 | **+0,078** | [+0,002; +0,157] | 11 / 5 |
| doc_hit | 0,561 | 0,737 | **+0,175** | [+0,088; +0,281] | 10 / 0 |
| wert_recall | 0,339 | 0,384 | +0,044 | [−0,019; +0,118] | 11 / 6 |
| Latenz Ø / Median | 10,1 / 8,1 s | 11,5 / 9,5 s | +1,4 s | – | – |

s0 → s2 gesamt: F +0,072, CP **+0,112** [+0,022; +0,206], CR **+0,113**, doc_hit **+0,211**, wert_recall **+0,086** (je signifikant außer F, AR).
Entscheidung: **behalten**. Kosten: +1,4 s Latenz (zusätzlicher Retrieval-Pfad).
Negativfall 48 („IP-Schutzart WDU 2.5“, steht nicht im Datenblatt) wird nicht mehr als Ablehnung erkannt: Antwort „hat keine IP-Klasse“ – inhaltlich vertretbar, aber keine Ablehnungsformel.

### Warum der Produktfilter (geplant s3) nicht eingesetzt wird
| Gruppe s2 (Einzelproduktfragen) | n | F | CP |
|---|---|---|---|
| nur Chunks des Zielprodukts | 7 | 0,748 | 0,742 |
| Zielprodukt + fremde Chunks (Ø 2,5) | 30 | **0,828** | 0,703 |
| Datenblatt fehlt | 7 | 0,653 | 0,423 |
Mit dem Lexikon-Pfad schaden die verbleibenden fremden Chunks nicht mehr messbar; die Fälle mit fremden Chunks sind sogar besser
(häufig Zubehör-/Varianten-Datenblätter, die zur Frage gehören). Projektion „Filter angewendet“ (Gruppe 2 auf Niveau Gruppe 1):
F 0,705, CP 0,632 – also eher schlechter. Regel 1 (Befund muss die Maßnahme begründen) ist nicht erfüllt → nicht umgesetzt.

### Verbleibende Lücken nach s2 (Ziel S1: F ≥ 0,78, CP ≥ 0,74, E2E ≤ 5 s)
- CP 0,612: Vergleich 0,40 (n=7), Produktsuche 0,29 (n=3), „Datenblatt fehlt“ 0,42 (n=7: UR20-Koppler ohne Typangabe 33–36,
  Zubehörfragen 9/70, Fall 16 ohne Ziel-Artikelnummer).
- Prompt-Komposition sortiert v3.0 „Tabellen zuerst“ und überstimmt damit die Cross-Encoder-Reihenfolge; RAGAS-CP ist rangbewertet
  (Average Precision über die Prompt-Blöcke) → Kandidat s3: Reihenfolge nach Rerank-Score (Liu et al. 2024; Kap. 4.6.2 Cross-Encoder
  als Schiedsrichter). Vorab Kalibrierung mit eval_s2.answers.jsonl (Proxy-CP je Reihenfolge).
- Latenz 11,5 s: weit vom Ziel; eigener Schritt nötig (Zeitanteile je Node messen).

## s3 – Kalibrierung ohne Judge (eval_s2.answers.jsonl, 02.10.2026)
Proxy-CP = Average Precision über die Prompt-Blöcke, relevant = Block enthält einen Referenzwert (Normalisierung wie wert_recall).
41 auswertbare beantwortbare Fälle; in 30 weicht die Prompt-Reihenfolge von der Cross-Encoder-Reihenfolge ab, 13 enthalten Tabellen.
| Teilmenge | n | Ist (v3.0-Sortierung) | nach Rerank-Score | Δ | besser / schlechter |
|---|---|---|---|---|---|
| Dev (ungerade Nr.) | 19 | 0,867 | 0,899 | +0,032 | 4 / 1 |
| Test (gerade Nr.) | 22 | 0,809 | 0,887 | +0,078 | 7 / 4 |
Einschränkung: Proxy-CP korreliert mit RAGAS-CP nur schwach (r = 0,12) – maßgeblich ist der Judge-Lauf s3.

## s3 – Ergebnis (02.10.2026): verworfen
Gepaarter Vergleich s2 → s3 (beantwortbar, Bootstrap 5000×):
| Kennzahl | s2 | s3 | Δ | KI95 | besser / schlechter |
|---|---|---|---|---|---|
| Faithfulness (n=55) | 0,759 | 0,720 | −0,039 | [−0,102; +0,024] | 13 / 18 |
| Context Precision (n=58) | 0,608 | 0,628 | +0,020 | [−0,035; +0,077] | 12 / 13 |
| Answer Relevancy (n=58) | 0,512 | 0,497 | −0,015 | [−0,089; +0,061] | 8 / 10 |
| Context Recall (n=56) | 0,497 | 0,486 | −0,011 | [−0,039; +0,016] | 5 / 6 |
| Ablehnung Negativfälle (n=16) | 0,625 | 0,562 | −1 Fall | – | – |
Zielmetrik CP steigt nur im Rauschen (+0,020, KI schließt 0 ein, 12 besser / 13 schlechter); F fällt um 0,039.
Die Proxy-Kalibrierung (+0,056) hat den Effekt überschätzt – wie wegen r = 0,12 befürchtet.
Deutung: Die v3.0-Regel „Tabellen zuerst“ (Kap. 4.2 [2c]: Tabellen = präzise Evidenz) ist für die Antwortqualität
offenbar nicht schädlich; Tabellenzeilen mit Werten am Anfang stützen die Faithfulness. Regel 2 nicht erfüllt → verworfen.
Judge-Streuung je Fall (AR bei 9 identischen Antworten s2/s3): Ø |Δ| = 0,086 – Einzelfall-Unterschiede < 0,1 sind nicht deutbar.

## Stand nach s3: bester Stand = s2
| | F | CP | AR | CR | Latenz Ø |
|---|---|---|---|---|---|
| s2 | 0,748 | 0,612 | 0,514 | 0,493 | 11,5 s |
| Ziel Stufe 1 | ≥ 0,78 | ≥ 0,74 | – | – | ≤ 5 s |

Latenz s2 (answers.jsonl): ohne HyDE Ø 10,4 s (n=67), mit HyDE Ø 20,9 s (n=8), Query-Rewrite 38,3 s (n=1);
Faithfulness-Validator (zweiter LLM-Aufruf) läuft in 74 von 75 Anfragen. Nächster Schritt: Zeitanteile je Node messen
(messe_latenz.py auf den gespeicherten n8n-Ausführungen von s2), dann gezielter Latenz-Schritt.

## s4 – Rerank-Tiefe 50 (vorbereitet, 02.10.2026)
- Basis: s2 (s3 verworfen).
- Quelle geprüft (arXiv:2604.01733, Fig. 6): Reranker-Tiefe 20 → Recall@5 0,458; 50 → 0,826; 100 → 0,888 (Cohere Rerank v4.0,
  Finanzberichte mit Text + Tabellen). Übertragbarkeit auf bge-reranker-v2-m3 und Datenblätter offen → Messung entscheidet.
- Erwartete Wirkung: CR und CP (bessere Chunks in den Top-8); Kosten: Cross-Encoder-Zeit ×5/3.

## s4 – Ergebnis (02.10.2026): verworfen
| Kennzahl (gepaart s2 → s4) | s2 | s4 | Δ | KI95 | besser / schlechter |
|---|---|---|---|---|---|
| Faithfulness (n=56) | 0,748 | 0,728 | −0,020 | [−0,096; +0,057] | 14 / 16 |
| Context Precision (n=59) | 0,612 | 0,618 | +0,006 | [−0,019; +0,034] | 7 / 4 |
| Answer Relevancy (n=59) | 0,514 | 0,484 | −0,031 | [−0,108; +0,044] | 9 / 11 |
| Context Recall (n=57) | 0,493 | 0,478 | −0,014 | [−0,052; +0,022] | 7 / 7 |
| doc_hit (n=57) | 0,737 | 0,702 | −0,035 | [−0,088; 0,000] | 0 / 2 |
| Latenz Ø | 11,5 s | 11,7 s | +0,2 s | – | – |
Varianten-Zitate (Antwort nennt Artikelnummer einer anderen Variante, Einzelproduktfragen n=36): s2 → Fälle 33, 34, 36; s4 → 2, 3, 33.
In den Fällen 2 und 3 (WDU 2.5) verdrängen zusätzliche Varianten-Datenblätter (WDU 2.5 BL u. a., Rerank-Score 0,988–0,991, fast wortgleich)
das gefragte Datenblatt. Deutung: Der Recall-Gewinn aus der Quelle (Finanzberichte, Cohere-Reranker) tritt hier nicht auf, weil der
Lexikon-Pfad (s2) das Recall-Problem bereits löst; die größere Tiefe bringt vor allem fast identische Varianten in den Pool, die der
Cross-Encoder nicht unterscheiden kann. Regel 2 nicht erfüllt → verworfen.

## s5 – HyDE-Regel (vorbereitet)
- Befund s2: HyDE in 8/75 Anfragen (5 × Q5, 3 × Q3), Ø 20,9 s gegenüber 10,4 s; diese Fälle sind schwach (F Ø ≈ 0,31).
- v3.0 weicht von der Architektur ab: Kap. 4.2 [1a] sieht HyDE **nur für Q3** vor, der Workflow aktiviert es auch für Q5.
- Änderung: Q5 ohne HyDE; Q3 ohne HyDE bei Artikelnummer oder Zahl+Einheit. Betroffen: Fälle 14, 23, 39, 65, 71, 68.

## Offene Maßnahmen aus den Quellen (Stand 02.10.2026, während s5 läuft)
| Maßnahme | Quelle | Befund bei uns | Ziel | Aufwand | Einschätzung |
|---|---|---|---|---|---|
| RRF architekturkonform: k = 60, ungewichtet (v3.0: k = 50, Gewichte 0,6/0,4/0,7) | Kap. 4.2 [3a]; Cormack et al. 2009 | Abweichung von der Architektur; Gewichte waren Notlösung für Top-15 | Konformität, kleiner Effekt | gering (Workflow) | Kandidat s6 |
| Ausreichender Kontext / selektive Antwort | Joren et al. 2025 (ICLR, Sufficient Context) | Ablehnung Negativfälle 0,625; 17 Antworten „nicht belegt“ | F, Ablehnung | mittel (Regel oder LLM-Prüfung, Latenz!) | Kandidat s7 |
| Deutsche Kompositazerlegung im BM25-Index | Braschler & Ripplinger 2004 | Fragen mit „Leiterquerschnitt“, „Anzugsdrehmoment“ … | Recall BM25 | hoch (BM25-Index neu bauen) | optional |
| Kontextauswahl ρ / k_max (weniger Chunks) | RAGAS-AP; eigene Kalibrierung v3.7 | Proxy unzuverlässig (s3: r = 0,12) | CP | gering | nur mit Judge-Lauf |
| Konvexe Score-Fusion statt RRF | Bruch et al. 2023 | erfordert Normalisierung + Abstimmung von α | CP/CR | mittel | optional |
| Tabellenstruktur bei der Ingestion | Akarsu et al. 2026 (73 % der Fehler) | Neu-Ingestion nötig | F, CP | sehr hoch | außerhalb des Rahmens |
| Latenz: Faithfulness-Validator (2. LLM-Aufruf) | Architektur (Validator vorgesehen) | läuft in 74/75 Anfragen | E2E ≤ 5 s | – | erst nach messe_latenz.py |

## Recherche Gewichtung Dense/BM25 (02.10.2026)
Frage: Ist die verbreitete Gewichtung „70 % Dense / 30 % BM25“ die beste Einstellung?
| Quelle | Verfahren | Ergebnis |
|---|---|---|
| Wang et al. 2024, „Searching for Best Practices in RAG“, arXiv:2407.01219 | S = α·S_sparse + S_dense, α ∈ {0,1; 0,3; 0,5; 0,7; 0,9} | α = 0,3 am besten (≈ 77 % Dense / 23 % BM25): DL19 nDCG@10 72,50 (Spanne 70,67–72,50), DL20 69,80 (65,55–69,80); Hybrid mAP 47,1 vs. BM25 30,1 / LLM-Embedder 44,7 |
| Ma et al. 2021, „A Replication Study of Dense Passage Retriever“, arXiv:2104.05740 | Sim + α·BM25, α je Datensatz per Gittersuche | optimales α je Datensatz sehr verschieden (WQ 0,3; NQ/TriviaQA 0,55; CuratedTREC 0,7; SQuAD 28); Hybrid Ø +3 Punkte Top-20 |
| Bruch, Gai & Ingber 2023, TOIS, arXiv:2210.11934 | konvexe Kombination (CC) vs. RRF | CC schlägt RRF in- und out-of-domain; RRF ist **empfindlich** gegenüber seinen Parametern; α für CC lässt sich mit wenigen gelabelten Anfragen abstimmen |
| Hsu et al. 2025, „DAT: Dynamic Alpha Tuning“, arXiv:2503.23013 | α je Anfrage durch LLM | bestes festes α = 0,6 (SQuAD, DRCD); dynamisch +2,8 / +3,3 Punkte P@1, bei „hybrid-sensitiven“ Anfragen +7,5 / +6,4 |
Folgerung: 70/30 ist ein für Web-Passagen (MS MARCO) gefundener Wert, kein allgemeines Optimum; das beste α hängt vom Datensatz ab
und muss auf eigenen Daten abgestimmt werden (Ma 2021, Bruch 2023). Bei PETRA wirkt die Gewichtung nur auf die Auswahl der 30
Kandidaten für den Cross-Encoder; seit s2 sichert die Produkt-Quote die Ziel-Chunks bei benannten Produkten → Effekt vor allem
bei Q3/Q5 und Vergleichen zu erwarten.
Korrektur: In früheren Notizen (v3.7, R2) stand „Gewichte in RRF ohne signifikanten Effekt (Bruch 2023)“. Bruch et al. zeigen
vielmehr, dass RRF parameterempfindlich ist und CC mit abgestimmtem α besser abschneidet.

## Manuelle Durchsicht (03.10.2026) und Abschluss-Schritt s6 (vorbereitet)
Befunde aus eval_s2 (bester gemessener Stand) zu den Beobachtungen der Durchsicht:

**Produktvergleich findet nicht alle Infos** – Ursache gefunden: Node „Vergleichskontext aufbereiten“ kürzt mit
`chunks.slice(0, 6)`; Differenzmatrix + 5 Chunks von Produkt A füllen alle Plätze, Produkt B fehlt (Fälle 31, 38, 44:
alle Quellen von Produkt A, Spalte B „nicht belegt“). Zusätzlich erzwingt der Vergleichshinweis fachfremde Standard-Zeilen
(Fall 72 Drucker: Schutzart, Tragschienentyp …). Architektur Kap. 4.4.1 / UC-02 benennt genau diese Verdrängung.

**Text besser als Tabelle/BM25?** – In den s2-Kontexten: 280 Text-Chunks, davon 54 % mit Referenzwert; 25 Tabellen-Chunks,
davon 56 % mit Referenzwert. Tabellen sind also nicht schlechter, sondern selten (8 % des Kontexts) und werden selten zitiert (4×).
Eine feste Umgewichtung ist durch die Daten nicht gedeckt; nach Bruch et al. 2023 / Ma et al. 2021 müsste α auf eigenen Daten
abgestimmt werden → Ausblick, nicht Teil des Abschlusses.

**Prompt-Fehler** (Node „Prompt-Komposition“):
1. Q5-Hinweis „Falls der Kontext nicht zur Frage passt, ignoriere den Kontext“ → Antwort aus Modellwissen (Fall 9).
2. Regel 5 „(Ausnahme: "AU" = Gold)“ = Antwort auf Negativfall 39 → Testset-Kontamination, widerspricht Regel 5 selbst.
3. „Nie die ganze Antwort verweigern … Verboten: 'brauche mehr Informationen'“ steht gegen F-13 (Negativfälle ablehnen).
4. Starre Schablone „Der/Die <Produkt> … hat <Attribut> von <Wert>“ → nur ein Wert je Attribut (Fall 12: nur min. 0,14 mm²).
5. numPredict 400: 8 von 75 Antworten enden mitten im Satz/in der Tabelle.

**Klassifikationstest** (Soll-Klasse je Frage nach Kap. 4.2 [1] annotiert; Datei klassifikation_test_s2.xlsx):
64 richtig, 3 vertretbar, 8 falsch → 89 % (67/75). Fehlerursachen:
| Fälle | Soll → Ist | Ursache |
|---|---|---|
| 37 | Q1 → Q4 | Preis-Heuristik prüft Teilstrings: „teuer“ in „Steuerspannung“ → Preis-Agent statt Datenblatt |
| 11, 63 | Q1 → Q2 | Regel „≥ 2 Produkte ⇒ Vergleich“ greift auch bei Mehrprodukt-Fragen ohne Vergleich |
| 9, 13 | Q2 → Q1 | zweites Produkt nicht als Code erkannt (WQV/ZQV ohne Ziffer; A2C 2.5) |
| 14, 65, 71 | Q1 → Q5 | Produktnamen ohne Ziffern (POK-Gehäuse, PrintJet ADVANCED) bzw. LLM wählt Restklasse |
Einschränkung: Die Soll-Klassen sind eigene Annotation (das Testset hat eine andere Fragetyp-Systematik).

**s6 = Abschlussschritt (Korrektur nachgewiesener Fehler, keine Parameter-Abstimmung)**, patch_s6_korrektur.py:
K1 Q5 nur aus Kontext (Zhou et al. 2023, Context-faithful Prompting) · K2 „AU = Gold“ entfernt · K3 Ablehnung, wenn Produkt
nicht im Kontext (Joren et al. 2025, Sufficient Context) · K4 Bereiche/mehrere Werte vollständig statt Schablone · K5 numPredict 900 ·
K6 Vergleich: je Produkt 4 Chunks im Wechsel + Attribute aus der Frage (Kap. 4.4.1/UC-02) · K7 Preis-Heuristik nur ganze Wörter.
Abweichung von Regel 1 (eine Änderung je Schritt) bewusst: Abschlussschritt; die Korrekturen wirken auf getrennte Fallgruppen
(Q5 / Fall 39 / Negativfälle / Q1 / lange Antworten / Q2 / Fall 37) und werden je Gruppe ausgewertet.
Nicht behoben (Ausblick): Klassifikationsregel ≥ 2 Produkte, Produktnamen ohne Ziffern, Gewichtung α, Latenz/Validator.

## Gewichtung Dense/BM25/TAG abstimmen (Wunsch 03.10.2026) – Vorgehen
- Keine Übernahme fester Praxiswerte (70/30). Stattdessen Abstimmung auf eigenen Daten nach Bruch et al. 2023 / Ma et al. 2021:
  kalibriere_gewichtung.py holt je beantwortbarem Fall einmal alle Kandidaten (Dense v3 20, BM25 20, TAG v3 10, Produkt-Pfad 8),
  bewertet sie einmal mit dem Cross-Encoder und simuliert dann für ein Raster aus Gewichten (Dense 0,4–1,0; BM25 0,2–1,0;
  TAG 0,2–1,0; k 50/60) die Workflow-Logik (RRF Top-30, Quoten, Cross-Encoder Top-8, CRAG 6 Chunks).
- Kennzahlen: doc_hit, Wertabdeckung, Proxy-CP; Auswahl auf Dev (ungerade), Bericht auf Test (gerade).
- Übernahme in s6 (K8, Option --gewichte) nur bei Test-Gewinn ≥ 0,02 und doc_hit nicht schlechter; sonst bleibt die Gewichtung.
- Grenzen: BM25-Anfrage im Skript = Fragetext (Workflow hängt erkannte Codes an); Proxy-Kennzahlen korrelieren nur schwach mit
  RAGAS (s3) → maßgeblich bleibt der Judge-Lauf s6.

## s5 – Ergebnis (03.10.2026): behalten
| Kennzahl (gepaart s2 → s5) | s2 | s5 | Δ | KI95 | besser / schlechter |
|---|---|---|---|---|---|
| Faithfulness (n=55) | 0,743 | 0,834 | **+0,091** | [+0,030; +0,155] | 19 / 7 |
| Context Precision (n=59) | 0,612 | 0,611 | −0,001 | [−0,010; +0,006] | 1 / 1 |
| Answer Relevancy (n=59) | 0,514 | 0,520 | +0,006 | [−0,057; +0,071] | 6 / 7 |
| Context Recall (n=57) | 0,493 | 0,473 | −0,020 | [−0,058; 0,000] | 0 / 2 |
| Ablehnung Negativfälle (n=16) | 0,625 | 0,562 | −1 Fall | – | – |
| Latenz Ø (alle / betroffene 6 Fälle) | 11,5 / 24,5 s | 11,3 / 18,9 s | – | – | – |
HyDE nur noch in 3 Anfragen (vorher 8). Betroffene beantwortbare Fälle 14, 23, 65, 68, 71: F 0,44 → 0,77; Fall 68 Latenz 38,3 → 6,8 s.

**Wichtiger Befund zur Messmethodik (Lauf-zu-Lauf-Streuung des Generators):** In den 54 beantwortbaren Fällen, die die
HyDE-Regel nicht betrifft, sind nur 21 Antworten wortgleich zu s2 (Generator llama3.1:8b mit temperature 0,1). Bei den
wortgleichen ändert sich F praktisch nicht (Ø Δ −0,02 → Judge stabil); bei den 33 geänderten Antworten steigt F um +0,12 –
ohne jede Konfigurationsänderung. Von den +0,091 F gehen daher nur ≈ +0,03 auf die Maßnahme zurück (5 betroffene Fälle × ≈ +0,34),
der Rest ist Zufallsstreuung der Generierung. Konsequenzen: (1) s5 wird behalten (Wirkung in den betroffenen Fällen klar, keine
Kernmetrik deutlich schlechter); (2) der Endwert F 0,834 ist mit dieser Unsicherheit zu berichten; (3) s6 setzt die
Generator-Temperatur auf 0 (K9), damit Läufe reproduzierbar werden. Frühere Verwerfungen (s3: F −0,039; s4: F −0,020) liegen
innerhalb dieser Streuung – sie werden verworfen, weil kein Gewinn belegt ist, nicht weil ein Schaden belegt ist.

## Kalibrierung Gewichtung – Ergebnis (03.10.2026, Basis s5, 59 beantwortbare Fälle, je 38–62 Kandidaten)
| Einstellung | Dense | BM25 | TAG | k | Dev-Ziel | Dev doc_hit | Test-Ziel | Test doc_hit |
|---|---|---|---|---|---|---|---|---|
| heute (v3.0) | 0,6 | 0,4 | 0,7 | 50 | 0,788 | 0,690 | 0,693 | 0,821 |
| 70/30-Praxis | 0,7 | 0,3 | 0,7 | 50 | 0,780 | 0,655 | 0,690 | 0,821 |
| Architektur (ungewichtet) | 1,0 | 1,0 | 1,0 | 60 | 0,772 | 0,655 | 0,694 | 0,821 |
| beste auf Dev | 0,4 | 0,2 | 0,7 | 50 | 0,788 | 0,690 | 0,693 | 0,821 |
(Ziel = Mittel aus Wertabdeckung und Proxy-CP der 6 Prompt-Chunks; Raster 161 Einstellungen.)
Ergebnis: Kein Raster-Punkt ist auf den Testfällen besser als die heutige Gewichtung (Δ 0,000). Die „70/30-Praxis“ findet auf Dev
sogar ein Datenblatt weniger. Deutung: Seit Lexikon-Pfad (s2, Produkt-Quote) und Cross-Encoder über die Auswahl entscheiden, ist die
RRF-Gewichtung für die Antwortqualität praktisch unerheblich – sie verschiebt nur Kandidaten auf den hinteren Plätzen der Top-30.
Entscheidung: **keine Änderung der Gewichtung (K8 entfällt)**; Befund für den Bericht: Gewichtung auf eigenen Daten geprüft (Bruch 2023),
kein Effekt im Gesamtsystem.

## s6 – Ergebnis (04.10.2026)
| Kennzahl (gepaart s5 → s6) | s5 | s6 | Δ | KI95 | besser / schlechter |
|---|---|---|---|---|---|
| Faithfulness (n=54) | 0,837 | 0,760 | **−0,077** | [−0,154; −0,001] | 12 / 22 |
| Context Precision (n=59) | 0,611 | 0,627 | +0,016 | [−0,001; +0,041] | 3 / 1 |
| Answer Relevancy (n=59) | 0,520 | 0,611 | **+0,091** | [+0,027; +0,166] | 18 / 6 |
| Context Recall (n=57) | 0,473 | 0,499 | +0,026 | [−0,009; +0,079] | 3 / 1 |
| wert_recall (n=48) | 0,360 | 0,417 | +0,057 | [−0,033; +0,148] | 9 / 8 |
| Ablehnung Negativfälle (n=16) | 0,562 | 0,688 | +2 Fälle (48, 49) | – | – |
| falsche Ablehnung (n=59) | 0,000 | 0,017 | +1 Fall (47) | – | – |
| Latenz Ø / Median | 11,3 / 9,5 s | 11,6 / 9,3 s | – | – | – |
Klassen s5 → s6: Vergleich AR 0,36 → 0,59, CR 0,42 → 0,50, wert_recall 0,08 → 0,28 (K6 wirkt); Produktspezifikation AR 0,55 → 0,62.
Faithfulness-Verlust: (a) Ablehnungen/„nicht belegt“-Antworten werden von RAGAS-F mit 0 bewertet (Fälle 71, 19); (b) vollständigere,
längere Antworten (Ø 472 → 574 Zeichen) enthalten mehr prüfbare Aussagen (Fall 12: alle Querschnitte genannt, F 1,00 → 0,29);
(c) ein Teil des s5-Werts war Generator-Zufall (+0,07, s. s5). Fehler: Fall 47 lehnt die WDU 2.5 fälschlich ab (K3 überdehnt),
Fall 19 lehnt mit „sensible Daten“ ab.

## Cache-Test CAG (04.10.2026)
Aufbau: Workflow s6, ohne Judge. Lauf 1 mit geleertem Cache (füllt den Cache), Lauf 2 eval_20261004_0750 mit gefülltem Cache
(Streamlit, Häkchen „Cache leeren“ aus); Auswertung `evaluate_petra.py --skip-ragas --answers … --out cache_warm.xlsx`.
| Kennzahl | ohne Cache (s6) | mit gefülltem Cache |
|---|---|---|
| Cache-Treffer | – | 66 / 75 (0,88) |
| Latenz Ø alle | 11,6 s | 1,2 s |
| Latenz Ø mit Treffer | – | 0,305 s |
| Latenz Ø ohne Treffer | 11,6 s | 10,06 s |
| Anteil ≤ 5 s | – | 0,88 |
| doc_hit / Ablehnung / wert_recall | 0,737 / 0,688 / 0,417 | 0,737 / 0,688 / 0,408 |
9 Fälle ohne Treffer: Antwort nicht cachewürdig (Validator / unbelegte Werte, Kap. 3.2.1 Prinzip 4).
88 % ist eine Obergrenze (wortgleiche Wiederholung); TTL 24 h (PETRA_CACHE_TTL_SECONDS), Invalidierung bei Dokumentimport.
Hinweis: Ein Judge-Lauf aus Streamlit unter Normalbetrieb (Ingestion an, OLLAMA_NUM_PARALLEL=4) lief ≈ 2 min je Bewertung
(≈ 10 h für 300) und wurde abgebrochen. Entscheidung: OLLAMA_NUM_PARALLEL dauerhaft 1 (Einzelplatz-Prototyp, NF-11);
Latenzen ab hier unter PARALLEL=1 gemessen (s0–s6: Phase 1 unter PARALLEL=4).
Fixes Streamlit: Schalter „Cache vor dem Lauf leeren“, Cache-Anzeige + Button, Spalten cache_hit/latency_client_ms,
Laufname wird übernommen (fester Widget-Schlüssel), doppelte Laufnamen werden abgelehnt.

### Cache-Test – Vergleich je Fall mit dem kalten Lauf von heute
Kalt eval_20261004_0730 (0 Treffer) gegen warm eval_20261004_0750:
- 66/66 Treffer-Antworten wortgleich mit dem kalten Lauf; 8/9 Nicht-Treffer ebenfalls wortgleich (Temperatur 0 reproduzierbar).
- Latenz kalt Ø 12,6 s (Median 9,9 s); Treffer-Fälle kalt Ø 12,9 s → warm Ø 0,30 s (Median 0,29, max 0,67), Faktor ×35 (Median je Fall).
- Ohne Treffer: kalt Ø 10,6 s → warm Ø 10,1 s. Fälle 19, 33, 35, 42, 54, 55, 58, 75 verified=false; Fall 24 verified=None.
Folgerung: Cache ist reine Latenzschicht (gleiche Antworten → gleiche RAGAS-Werte, kein zweiter Judge-Lauf nötig).
Vergleichsbefehl: docker exec -i petra_frontend python3 - (Skript im Chat 04.10.2026; liest nur die beiden answers.jsonl).

## OLLAMA_NUM_PARALLEL 4 → 1 – Kontrollmessung (04.10.2026)
Kalt eval_20261004_0730 (PARALLEL=4) gegen kalt_p1 (PARALLEL=1), beide s6, ohne Judge, 0 Cache-Treffer:
| Kennzahl | PARALLEL=4 | PARALLEL=1 |
|---|---|---|
| Latenz Ø / Median (Client) | 12,63 / 9,95 s | 12,57 / 9,79 s |
| Latenz Ø (Workflow) | 12,09 s | 11,95 s |
| gepaarte Differenz je Fall | – | Ø −0,05 s, KI95 [−1,03; +0,72]; 18 schneller / 20 langsamer (>0,5 s) |
| Anteil ≤ 5 s | 0 % | 0 % |
| doc_hit / Ablehnung / wert_recall / beleg_quote | 0,737 / 0,688 / 0,408 / 0,955 | 0,737 / 0,688 / 0,406 / 0,954 |
| verified (cachewürdig) | – | 66 / 75 |
Befund: Latenz unverändert → Umstellung ohne Nachteil; Latenzen aus s0–s6 bleiben vergleichbar.
Aber: nur 49/75 Antworten wortgleich (gleiche Einstellung: 8/9) → Temperatur 0 reproduzierbar nur bei gleicher
Laufzeitumgebung. RAGAS-Werte s6 gelten für Antworten unter PARALLEL=4; deterministische Kennzahlen praktisch gleich.
Optional: RAGAS auf kalt_p1.answers.jsonl unter Judge-Bedingungen (ingestion gestoppt) für einen Endstand unter PARALLEL=1.

## s7 – Einstellungen wirksam machen (vorbereitet 04.10.2026)
Befund (Code-Prüfung Frontend → FastAPI → n8n s6): Von der Seite „Einstellungen“ wirkte nur die Antwortsprache.
- temperature/model: von FastAPI (QueryRequest) verworfen; Generator-Node fest 0 / llama3.1:8b.
- top_k: in „Input normalisieren“ gesetzt, danach nirgends gelesen (Kontext fest MAX_PASSED = 6).
- web_agent_enabled: nicht gesendet; Agent liest nur PETRA_WEB_AGENT_ENABLED.
- score_threshold: nur im Rückfallpfad ohne Reranker (bleibt so, Seite sagt es jetzt).
Änderung (Standardwerte = s6, Evaluation unverändert):
- patch_s7_einstellungen.py (n8n): gen_temperature/gen_model/top_k/web_agent_override/gen_profile in „Input normalisieren“;
  CRAG MAX_PASSED = top_k; Generator temperature+model als Ausdruck; Cache-Schlüssel extra = gen_profile;
  Preis-Agent erhält web_agent_enabled; Output Parser meldet model+temperature.
- patch_backend_s7.py: QueryRequest temperature/model/web_agent_enabled, top_k ohne Standard; QueryResponse temperature;
  agent.web_agent_enabled() mit ContextVar-Vorgabe je Anfrage.
- patch_frontend_s7.py: neue views/settings.py (nur wirksame Regler, Modellprüfung per Knopf), config/chat Standard T=0, Top-K 6.
- evaluate_petra.py --temperature/--model/--top-k; Evaluation-Seite: Auswahl Generator-Temperatur (nur mit Cache leeren).
- test_einstellungen.py: Ende-zu-Ende-Prüfung über /query (T=0 wortgleich, T=1 verschieden, Top-K, Cache-Trennung).
Lokal geprüft: Patch-Kette v3.0→s0→s1→s2→s5→s6→s7, Input-Node in Node.js, Backend-Patch mit Attrappe, Frontend-Patch mit Attrappe.

## Temperatur-Test Generator (04.10.2026)
Kopie von s6, nur Generator-Temperatur geändert; ohne Judge, Cache je Lauf geleert, PARALLEL=1. Referenz kalt_p1 (T=0).
| Kennzahl | T0 | T0,1 | T0,8a | T0,8b |
|---|---|---|---|---|
| wortgleich mit T0 | – | 28/75 | 0/75 | 2/75 |
| Textähnlichkeit zu T0 (difflib) | – | 0,71 | 0,38 | 0,34 |
| doc_hit | 0,737 | 0,737 | 0,737 | 0,737 |
| wert_recall | 0,406 | 0,357 | 0,428 | 0,342 |
| beleg_quote | 0,954 | 0,953 | 0,956 | 0,933 |
| ablehnung_ok (n=16) | 11 | 12 | 11 | 9 |
| falsche Ablehnung (n=59) | 1 | 2 | 1 | 5 |
| Validator-Faithfulness | 0,874 | 0,824 | 0,847 | 0,840 |
| Länge Ø Zeichen | 604 | 605 | 722 | 614 |
| Latenz Ø | 12,6 s | 12,4 s | 12,5 s | 12,2 s |
Gepaart (Bootstrap 5000): wert_recall 0,8a→0,8b −0,086 [−0,165; −0,015] (reiner Zufall); T0→0,8a +0,022 [−0,037; +0,087];
Validator T0→0,1 −0,050 [−0,098; −0,010]. 0,8a↔0,8b Textähnlichkeit 0,29.
Ergebnis: kein systematischer Qualitätseffekt (vgl. Renze & Guven 2024, arXiv:2402.05201), aber Streuung und Ausreißer
(0,8b: 5 falsche Ablehnungen) → Temperatur 0 bleibt.

## Dashboard/Ingestion: Zeitüberschreitungen (vorbereitet 04.10.2026)
Befund: /status liest bei jedem Aufruf alle ≈ 720 000 Chunk-Metadaten (500er-Blöcke) → Minuten (Frontend-Timeout 60 s);
/export-report dieselbe Vollzählung (Timeout 120 s); /import-log fehlt (404), obwohl save_to_chromadb.py
REPORT_DIR/import_status.jsonl schreibt; /list-pdfs hasht bei jedem Aufruf alle PDFs (Timeout 90 s).
Änderung: status_cache.py (Statistik einmal berechnen, JSON-Cache, Hintergrund-Neuberechnung nach Import oder 6 h,
5000er-Blöcke, ohne ChromaDB-Sperre); api_server.py /status + /export-report aus Cache, neu GET /import-log;
list_pdfs.py Hash-Cache (Pfad, Größe, mtime) → nur neue/geänderte PDFs; Frontend zeigt Hinweis während der Erstberechnung.
Dateien: patch_ingestion_status.py + status_cache.py, patch_frontend_status.py. Lokal mit Attrappen geprüft.
Offener Befund: Ingestion schreibt in die v2-Collections, der Workflow liest seit s1 die v3-Collections (einmaliger Reindex)
→ neu importierte PDFs fehlen im Dense-/TAG-Retrieval.

## Ingestion → v3-Index (vorbereitet 04.10.2026)
Befund: Workflow liest seit s1 petra_text_chunks_v3/petra_tag_chunks_v3, Produktpfad (s2) das Produkt-Lexikon – beides nur
einmalig durch reindex_v3_contextual.py erzeugt. save_to_chromadb.py schreibt neue PDFs nur in v2 und ohne Kontextkopf in BM25
→ neue Datenblätter fehlen im Dense-, TAG- und Produktpfad.
Änderung: v3_sync.py (gleiche Funktionen wie der Reindex: Knowledge/enrich/is_junk) – je gespeicherter Datei: Chunks aus v2 lesen,
Kopf voranstellen, alte v3-Einträge der Datei löschen, neu einbetten (bge-m3 aus save_to_chromadb), Lexikon ergänzen;
BM25 erhält die Texte mit Kopf. patch_v3_sync.py hängt den Aufruf in save_chunks_to_chromadb() ein (abschaltbar PETRA_V3_SYNC=false;
Fehler brechen den Import nicht ab). Nachholen: v3_sync.py --fehlende. Lokal mit Chroma-Attrappe geprüft.

## Letzte Version installiert und geprüft (04.10.2026, 11:40–11:55)
installieren.sh (petra_letzte_version.zip): alle Patches im Container und in der Host-Quelle angewendet, Workflow
„PETRA-RAG · Schritt 7: Einstellungen wirksam“ importiert (ID O1VD1JA4lXqrsmM1).
test_einstellungen.py („Welche technischen Daten hat die WDU 2.5?“) → **Einstellungen wirken**:
| Prüfung | Ergebnis |
|---|---|
| T=0 zweimal | wortgleich (11,3 s; erster Aufruf 96,5 s = Modell laden) |
| T=1 zweimal | verschieden |
| Top-K 2 / 8 | 5 / 8 Quellen |
| Cache trennt Einstellungen | T=0,8 nach T=0 kein Treffer; T=0 erneut Treffer (0,0 s) |
v3-Index: `v3_sync.py --fehlende` → In v2: 26 821 Dateien, in v3: 26 821, fehlend 0 → kein Nachholen nötig;
neue Importe gehen ab jetzt automatisch auch in v3, Lexikon und BM25 (mit Kontextkopf).
Hinweis: PETRA_WEB_AGENT_ENABLED=true im Container; die Oberfläche sendet je Anfrage web_agent_enabled (Standard aus).

## Abnahmetest Einstellungen + Produktberatung (04.10.2026, Stand s7)
Einstellungen (Oberfläche): E3 Cache ✅ (143/186 ms), E4 Temperatur ✅ (0,8 kommt im Workflow an), E5 Top-K ✅ (2 → 5 Quellen,
8 → 8 Quellen), E7 Web-Agent ⚠️ (aus: kein Preis ✅; an: HTTP-Knoten bricht nach 16 s ab, ddgs „auto“ wählte Yahoo → 8–9 s
Timeout). Nebenbefund: Generator stand nach E2 auf qwen2.5:32b → 78–81 s je Antwort (Prämisse P2 verletzt).
Anbietertest im Container (ddgs 9.16): duckduckgo 0,9 s ✅, html 1,2 s ✅, bing Werbelinks, google/brave/mojeek 0 Treffer,
yahoo/lite 8–9 s Timeout.
test_produktberatung.py (20 Fälle, Web-Agent aus): **12/20 automatisch bestanden**; F-11 Quellen vollständig 97 %;
NF-03 ohne Cache 4/19 ≤ 5 s (Median 9,6 s), Cache-Treffer 0,3 s.
| Fall | Befund | Ursache |
|---|---|---|
| P06 Vergleich WDU 2.5 / A2C 2.5 | Q1 statt Q2 | „A2C 2.5“ verworfen: Buchstaben „AC“ stehen auf der Normbezeichnungs-Liste |
| P07 „Vergleiche die WDU 2.5.“ | kein Hinweis (UC-02 A1) | 1 Produkt → stumm Q1 |
| P09 „Welche Reihenklemmen … gibt es …?“ | Q1 statt Q3 | keine Suchsignale ohne Produktcode |
| P14 „Wofür steht der Verschmutzungsgrad …?“ | Q1 statt Q5 | „daten“ in „Datenblatt“ zählt als Attributwort |
| P17/P18 englische Frage | deutsche Antwort, auch bei Einstellung Englisch | body.language vor Erkennung; „- Sprache: Englisch“ im deutschen Prompt übergangen |
| P02 Eignungsfrage UR20 | Wert 1,5 mm² richtig, kein klares „Nein“ | Antwort prüfen (Prompt) |
| P08 M12-Leitung mit LED | Datenblatt 1094190500 nicht gefunden | Retrieval Q3 / Korpus prüfen |
| (Anzeige) | Faithfulness nie sichtbar | Frontend liest „faithfulness“, Backend liefert „faithfulness_score“ |

## s8 – Abnahme-Korrekturen (vorbereitet 04.10.2026)
patch_s8_abnahme.py (n8n): K1 Sprache = Sprache der Frage außer ausdrücklicher Vorgabe, Englisch-Anweisung oben und am
Prompt-Ende (NF-08); K2 Normbezeichnungs-Filter nur bei vorn stehendem Buchstabenblock (A2C 2.5 bleibt Produkt);
K3 Begriffsfragen ohne Produktcode → Q5; K4 „Welche … gibt es“, „eignen sich“ … ohne Produktcode → Q3; K5 Vergleich mit nur
einem Produkt → Hinweis (UC-02 A1); K6 Preis-Agent-Knoten 20 s, ehrliche Meldung ohne Preis.
patch_backend_s8.py: Web-Suche mit festen Anbietern (PETRA_WEB_BACKENDS=duckduckgo,html), Zeitgrenze = Restzeit des
Agenten (max. 6 s je Suche), Werbelinks gefiltert; QueryRequest.language Standard „auto“.
patch_frontend_s8.py: Antwortsprache „Automatisch (wie Frage)“, Modellauswahl nur ≤ 7,2 GB (P2), Faithfulness-Anzeige.
Lokal geprüft: alle Code-Knoten syntaktisch, Klassifikation simuliert (Testset: nur Fälle 12, 13, 14 ändern sich – alle
A2C 2.5, jeweils zur richtigen Klasse), Spracheingang, Hinweis UC-02 A1, Preis-Meldungen, Web-Suche mit simuliertem
Anbieterausfall und Werbelink, Einstellungsseite mit Attrappe.
Wirkung auf die Evaluation: Testset-Fragen sind deutsch und senden keine Sprache → unverändert; Klassen ändern sich nur in
Fällen 12–14. Kontrolllauf „kalt_s8“ ohne RAGAS gegen kalt_p1 empfohlen.

## Entscheidung 04.10.2026: Schritt 8 zurückgenommen – nur funktionale Reparatur des Web-Agenten
Regel (Vorgabe Nutzer): Repariert wird nur, was **keine Antwort liefert oder abbricht** (nicht bewertbar). Alles, was eine
Antwort liefert – auch eine unvollständige –, bleibt im Endstand und wird als Befund dokumentiert; sonst wären die
Schritte 1–7 und die Evaluation nicht mehr vergleichbar.
- Zurückgenommen (Sicherungen .bak_s8): QueryRequest-Sprachstandard, Frontend (Sprache „Automatisch“, Modellfilter,
  Faithfulness-Anzeige); Workflow „Schritt 8/8b“ unveröffentlicht, aktiv ist wieder **Schritt 7**.
- Behalten: agent.py – Web-Suche mit festen Anbietern (duckduckgo, html) und harter Zeitgrenze (vorher: ddgs „auto“ →
  Yahoo-Timeout, Abbruch). Im n8n-Knoten „Agentic: Preis-Recherche (HTTP)“ Timeout 16 000 → 30 000 ms (Agent endet nach
  15 s + ≈ 1,4 s Übertragung = 16,4 s gemessen; Architektur Kap. 4.4.2: 15–45 s). Kontrolle: /query mit Web-Agent →
  Antwort nach 13,9 s („Ich weiss es nicht …“, kein erfundener Preis); Web-Suche liefert Treffer (8 über html), Preise
  auf Shopseiten oft nicht lesbar (Bot-Sperren).
- Gültiger Abnahmetest des Endstands: test_produktberatung_20261004_1337.xlsx (12/20). Nicht bestandene Fälle →
  Befunde/Ausblick: NF-08 Antwortsprache (englische Fragen deutsch beantwortet), Produkterkennung „A2C 2.5“ (Filter „AC“),
  UC-02 A1 Hinweis bei nur einem Produkt, Klassifikation Q3 („Welche … gibt es“) und Q5 („Wofür steht …“),
  Platzhalter „<Produkt>“ in Preisantworten (s6-Vorlage K3), Eignungsfrage ohne klares Ja/Nein (P02), Q3-Retrieval
  (P08), Faithfulness-Wert im Chat nicht angezeigt (Feldname), 32b-Modell in der Auswahl (P2). Lösungsentwürfe:
  patch_s8_abnahme.py, patch_s8b_platzhalter.py, patch_frontend_s8.py (nicht eingesetzt).

## Sichtprüfung im Chat (04.10.2026, Stand Schritt 7 + Web-Agent-Reparatur)
| Prüfung | Ergebnis | Befund |
|---|---|---|
| Q1 Nennstrom WDU 2.5 | ✅ 24 A mit Quelle, 8 Quellen mit Datei/Seite/Relevanz, 15,6 s | – |
| Q2 Vergleich TRS 1CO/2CO | ✅ Tabelle 1 CO/2 CO, 6 A/8 A (= Referenz Fall 38) | Validator 0,5 → „Nicht verifiziert“ (falscher Alarm, Judge bei Tabellen zu streng); Quelle „Seite None“ |
| Negativfall IP WDU 2.5 | ✅ „IP-Klasse nicht belegt“ | erster Satz „hat keine IP-Klasse“ missverständlich |
| Q5 PUSH IN | ✅ Erklärung aus Datenblatt (2712290100, S. 3), keine Empfehlung, 16,5 s | – |
| Q5 Verschmutzungsgrad | ⚠️ Antwort vereinfacht („Leitfähigkeit von Verschmutzung“), Relevanz nur 17–23 %, Katalogquellen | schwacher Kontext für Begriffsfragen |
| Q4 Preis BL 3.50/05/180F SN OR BX (mit „?“) | ❌ „nicht belegt“, obwohl UC-04-Preisextraktion + Preisliste S. 232 im Kontext (415,36 €; im Test ohne „?“ korrekt genannt) | **falsche Ablehnung** durch den Generator (Regel „nicht belegt“ übergreift), nicht robust gegen Umformulierung |
| Q4 ifm O1D302, Web aus / an | ✅ ehrliche Antwort ohne Preis (0,5 s / 14,4 s), kein Abbruch mehr | Text „keine relevanten Dokumentabschnitte“ ungenau für Preisfragen |
| Q4 PRO ECO3 240W 24V 10A II, Web an | ✅ Web-Preis 98,99 € (elektronetshop, Stand 2026-10-04) + Listenpreis 212,70 € (Preisliste S. 336, undatiert), Hinweis auf Aktualität, 8 Quellen mit URL, 14,2 s (UC-04/UC-11 erfüllt) | Einleitung „… ist nicht belegt“ widersprüchlich; Fußzeile zeigt „lokale Pipeline“ auch beim Web-Pfad (agent_path nicht im Backend-Schema); Web-Quellen „Seite None“. Mit Web aus (Test P11) wurde derselbe Listenpreis 212,70 € **nicht** genannt → falsche Ablehnung |
Gesamt: Alle Fragetypen liefern eine Rückmeldung; Web-Agent funktional. Hauptbefund für den Bericht: falsche Ablehnungen bei
Preisfragen trotz belegtem Listenpreis (Generator, Prompt-Regel „nicht belegt“), Validator zu streng bei Vergleichen.

## Nachtrag Abgabe (08.10.2026)
- Der im Repository abgelegte Workflow `n8n_workflows/PETRA-RAG_Schritt_7_Einstellungen_wirksam.json` ist logisch identisch mit
  der am 04.10.2026 veröffentlichten Version (Prüfung gegen die n8n-Datenbank; Unterschiede nur in Kommentaren, Notizen und
  Knotenpositionen nach der Kommentarüberarbeitung).
- Die oben beschriebene Timeout-Änderung 16 000 → 30 000 ms im Knoten „Agentic: Preis-Recherche (HTTP)“ war in der
  Datenbank nicht gespeichert; sie ist in der abgegebenen Workflow-Datei eingetragen.
- `evaluation/testset/testset_v2.xlsx` ist aus `frontend/evaluation/testfaelle.json` erzeugt (dieselben 75 Fälle, 16 Negativfälle).
