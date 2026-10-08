"""
services/evaluation.py — RAGAs-Evaluation gegen den Weidmüller-Testdatensatz.

Berechnet Faithfulness, Answer Relevancy, Context Precision und Context
Recall für Antworten der laufenden Pipeline (POST /query).

Testdatensatz: evaluation/testfaelle.json (75 Fälle mit Referenzantworten).

Einschränkung: `sources[].textauszug` enthält nur die ersten 220 Zeichen je
Chunk (n8n-Node "Output Parser (Antwort + Quellen)"). Die kontextbezogenen
Metriken werden daher auf Ausschnitten berechnet. Die systeminterne
Faithfulness-Prüfung arbeitet auf dem vollen Kontext; die Differenz steht
in `faithfulness_delta`.

Nach jedem Testfall wird das Zwischenergebnis gespeichert; ein Abbruch
verliert höchstens den laufenden Fall.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import requests

from config import CONFIG

log = logging.getLogger(__name__)

# ── Pfade ─────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
TESTFAELLE_PATH = BASE_DIR / "evaluation" / "testfaelle.json"
ERGEBNIS_DIR = BASE_DIR / "evaluation" / "laeufe"


# ── Datenmodell ───────────────────────────────────────────────────────────
@dataclass
class Testfall:
    nr: int
    produktbereich: str
    produkt: str
    artikelnummer: str
    schwierigkeit: str
    fragetyp: str
    frage: str
    referenzantwort: str
    quelle_dokument: str = ""
    quelle_url: str = ""
    quelle_hersteller: str = ""
    testziel: str = ""
    nicht_beantwortbar: bool = False
    multi_produkt: bool = False
    data_quality_flags: list[str] = field(default_factory=list)


@dataclass
class Ergebnis:
    nr: int
    frage: str
    referenzantwort: str = ""
    antwort: str = ""
    kontext: list[str] = field(default_factory=list)
    quellen: list[dict] = field(default_factory=list)
    query_class: str = ""
    latency_ms: int = 0
    cache_hit: bool = False
    latency_ms_warm: Optional[int] = None
    cache_hit_warm: Optional[bool] = None
    verified: Optional[bool] = None
    system_faithfulness_score: Optional[float] = None
    fallback_reason: Optional[str] = None
    context_sufficient: Optional[bool] = None
    fehler: str = ""

    ragas_faithfulness: Optional[float] = None
    ragas_answer_relevancy: Optional[float] = None
    ragas_context_precision: Optional[float] = None
    ragas_context_recall: Optional[float] = None
    faithfulness_delta: Optional[float] = None

    faithfulness_hinweis: Optional[str] = None
    answer_relevancy_hinweis: Optional[str] = None
    context_precision_hinweis: Optional[str] = None
    context_recall_hinweis: Optional[str] = None

    zeitpunkt: str = ""

    @property
    def hat_metriken(self) -> bool:
        return any(
            isinstance(getattr(self, k), (int, float))
            for k in ("ragas_faithfulness", "ragas_answer_relevancy",
                      "ragas_context_precision", "ragas_context_recall")
        )


# ── Testfälle laden ───────────────────────────────────────────────────────
def lade_testfaelle(pfad: Path | None = None) -> list[Testfall]:
    p = pfad or TESTFAELLE_PATH
    with open(p, encoding="utf-8") as f:
        daten = json.load(f)
    return [Testfall(**x) for x in daten["faelle"]]


# ── Aufruf der echten Pipeline (unverändert ggü. Vorgängerversion) ───────
def frage_stellen(api_base: str, frage: str, timeout: float = 200.0) -> dict:
    """Einzelne Anfrage an POST /query — dieselbe echte Pipeline wie zuvor."""
    resp = requests.post(
        f"{api_base.rstrip('/')}/query",
        json={"query": frage, "language": "Deutsch"},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def cache_invalidieren(api_base: str) -> bool:
    try:
        requests.post(
            f"{api_base.rstrip('/')}/cache/invalidate",
            json={"corpus_version": f"eval-{int(time.time())}"},
            timeout=10,
        )
        return True
    except requests.RequestException:
        return False


def build_contexts(sources: list[dict]) -> list[str]:
    """Extrahiert die Kontext-Textausschnitte aus den API-Quellen.

    LIMITATION: `textauszug` ist auf 220 Zeichen gekürzt (siehe Docstring
    oben) — kein voller Chunk.
    """
    return [s["textauszug"].strip() for s in (sources or []) if (s.get("textauszug") or "").strip()]



def linearize_markdown_table(text: str) -> str:
    """Wandelt eine Markdown-Tabelle in Aussagesätze um.

    Der lokale Judge konnte Tabellenantworten nicht in Einzelaussagen
    zerlegen. Wird nur für die Faithfulness-Berechnung verwendet; die
    gespeicherte Antwort bleibt unverändert.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip().startswith('|')]
    if len(lines) < 3:
        return text  # keine Tabelle erkannt

    def split_row(l):
        return [c.strip() for c in l.strip('|').split('|')]

    header = split_row(lines[0])
    rows = [split_row(l) for l in lines[2:]]  # lines[1] ist die "---"-Trennzeile

    saetze = []
    for row in rows:
        if not row:
            continue
        attribut = row[0]
        for spalte, wert in zip(header[1:], row[1:]):
            if not wert:
                continue
            saetze.append(f"{spalte}: {attribut} ist {wert}.")
    return ' '.join(saetze) if saetze else text

# ── RAGAs-Anbindung ───────────────────────────────────────────────────────
class RagasEngine:
    """
    LLM-Judge, Embeddings und die vier RAGAs-Metriken.

    Verwendet ragas >= 0.2 (SingleTurnSample, `single_turn_ascore`) mit
    Fallback auf die Legacy-API `evaluate()`. Schlägt die Initialisierung
    fehl, liefern alle Metriken None mit Begründung im Hinweisfeld.
    """

    def __init__(self, settings, ollama_url: str) -> None:
        self.settings = settings
        self.ollama_url = ollama_url.rstrip("/")
        self.available = False
        self.setup_error = ""
        self.legacy_mode = False
        self._llm = None
        self._embeddings = None
        self._metrics: dict[str, Any] = {}

    def setup(self) -> None:
        try:
            self._setup_modern()
            self.available = True
        except Exception as exc:  # noqa: BLE001
            try:
                self._setup_legacy()
                self.available = True
                self.legacy_mode = True
            except Exception as exc2:  # noqa: BLE001
                self.setup_error = (
                    f"RAGAs-Initialisierung fehlgeschlagen. Moderne API: {exc}. "
                    f"Legacy-API: {exc2}. Prüfen Sie: (1) ist ragas/langchain-community "
                    f"installiert, (2) ist Ollama unter {self.ollama_url} erreichbar und "
                    f"sind '{self.settings.judge_model}'/'{self.settings.embedding_model}' gepullt."
                )
                log.warning(self.setup_error)

    def _setup_modern(self) -> None:
        from ragas import SingleTurnSample  # noqa: F401 (Importprüfung)
        from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper

        llm, embeddings = self._build_langchain_components()
        wrapped_llm = LangchainLLMWrapper(llm)
        wrapped_emb = LangchainEmbeddingsWrapper(embeddings)
        self._llm, self._embeddings = wrapped_llm, wrapped_emb
        self._metrics = {
            "faithfulness": Faithfulness(llm=wrapped_llm),
            "answer_relevancy": AnswerRelevancy(llm=wrapped_llm, embeddings=wrapped_emb),
            "context_precision": ContextPrecision(llm=wrapped_llm),
            "context_recall": ContextRecall(llm=wrapped_llm),
        }

    def _setup_legacy(self) -> None:
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        llm, embeddings = self._build_langchain_components()
        self._llm, self._embeddings = llm, embeddings
        self._metrics = {
            "faithfulness": faithfulness, "answer_relevancy": answer_relevancy,
            "context_precision": context_precision, "context_recall": context_recall,
        }

    def _build_langchain_components(self):
        """
        Judge-LLM und Embeddings über lokales Ollama (kein Cloud-Provider, P1).

        num_ctx/num_predict sind explizit gesetzt: Mit dem Ollama-Default von
        2048 Token wurden längere Eingaben stillschweigend abgeschnitten
        (Answer Relevancy 0.0, Faithfulness None).
        """
        from langchain_community.chat_models import ChatOllama
        from langchain_community.embeddings import OllamaEmbeddings
        return (ChatOllama(model=self.settings.judge_model, base_url=self.ollama_url,
                            temperature=0, num_ctx=16384, num_predict=2000),
                OllamaEmbeddings(model=self.settings.embedding_model, base_url=self.ollama_url))

    async def score_row(self, question: str, answer: str, contexts: list[str], ground_truth: str) -> dict:
        result: dict[str, Any] = {}
        if not self.available:
            reason = f"RAGAs nicht verfügbar: {self.setup_error or 'unbekannter Grund'}"
            for key in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
                result[f"ragas_{key}"], result[f"{key}_hinweis"] = None, reason
            return result

        has_answer, has_context = bool((answer or "").strip()), bool(contexts)
        has_gt = bool((ground_truth or "").strip())
        checks = {
            "faithfulness": (has_answer and has_context, "Keine Antwort und/oder kein Kontext vorhanden."),
            "answer_relevancy": (has_answer, "Keine Antwort vorhanden."),
            "context_precision": (has_context and has_gt, "Kein Kontext und/oder keine Referenzantwort vorhanden."),
            "context_recall": (has_context and has_gt, "Kein Kontext und/oder keine Referenzantwort vorhanden."),
        }
        for key, (ok, reason) in checks.items():
            if not ok:
                result[f"ragas_{key}"], result[f"{key}_hinweis"] = None, reason
                continue
            try:
                scoring_answer = linearize_markdown_table(answer) if key == "faithfulness" else answer
                value = await self._score_single(key, question, scoring_answer, contexts, ground_truth)
                result[f"ragas_{key}"] = round(float(value), 4) if value is not None else None
                result[f"{key}_hinweis"] = None if value is not None else "Metrik lieferte keinen Wert (None)."
            except Exception as exc:  # noqa: BLE001
                result[f"ragas_{key}"] = None
                result[f"{key}_hinweis"] = f"Berechnung fehlgeschlagen: {type(exc).__name__}: {exc}"
        return result

    async def _score_single(self, key, question, answer, contexts, ground_truth) -> Optional[float]:
        metric = self._metrics[key]
        if not self.legacy_mode:
            from ragas import SingleTurnSample
            sample = SingleTurnSample(user_input=question, response=answer,
                                       retrieved_contexts=contexts, reference=ground_truth)
            return await metric.single_turn_ascore(sample)
        return await asyncio.to_thread(self._score_legacy, key, question, answer, contexts, ground_truth)

    def _score_legacy(self, key, question, answer, contexts, ground_truth) -> Optional[float]:
        from datasets import Dataset
        from ragas import evaluate
        ds = Dataset.from_dict({"question": [question], "answer": [answer],
                                 "contexts": [contexts], "ground_truth": [ground_truth]})
        result = evaluate(ds, metrics=[self._metrics[key]], llm=self._llm, embeddings=self._embeddings)
        df = result.to_pandas()
        cols = [c for c in df.columns if key in c]
        if not cols:
            return None
        val = df.iloc[0][cols[0]]
        return None if val is None or (isinstance(val, float) and val != val) else float(val)


# ── Ausführung ────────────────────────────────────────────────────────────
def lauf_ausfuehren(
    api_base: str,
    faelle: list[Testfall],
    lauf_id: str,
    fortschritt: Optional[Callable[[int, int, Ergebnis], None]] = None,
    cache_leeren: bool = True,
    ragas_ueberspringen: bool = False,
    latenz_vergleich: bool = False,
) -> Iterator[Ergebnis]:
    """Führt die Testfälle nacheinander aus (Pipeline-Aufruf und RAGAs-Scoring).

    Generator, damit die Oberfläche Zwischenstände anzeigen kann. Nach jedem
    Fall wird gespeichert.

    Args:
        latenz_vergleich: Je Fall Cache leeren, kalt aufrufen und denselben
            Aufruf sofort wiederholen (warm). Ob der Cache traf, steht in
            `cache_hit_warm`. RAGAs bewertet nur die kalte Antwort.
            Überschreibt `cache_leeren`.
    """
    ERGEBNIS_DIR.mkdir(parents=True, exist_ok=True)
    ziel = ERGEBNIS_DIR / f"{lauf_id}.json"
    gesammelt: list[Ergebnis] = []

    if cache_leeren and not latenz_vergleich:
        cache_invalidieren(api_base)

    engine: Optional[RagasEngine] = None
    if not ragas_ueberspringen:
        engine = RagasEngine(CONFIG.ragas, CONFIG.endpoints.ollama)
        engine.setup()

    for i, fall in enumerate(faelle, start=1):
        erg = Ergebnis(nr=fall.nr, frage=fall.frage, referenzantwort=fall.referenzantwort,
                       zeitpunkt=datetime.now().isoformat(timespec="seconds"))
        try:
            if latenz_vergleich:
                cache_invalidieren(api_base)  # garantiert Kaltstart für DIESEN Fall

            daten = frage_stellen(api_base, fall.frage)
            sources = daten.get("sources") or []
            kontext = [c for c in (daten.get("contexts") or []) if c] or build_contexts(sources)  # v3.1 volle Chunks

            erg.antwort = daten.get("answer") or ""
            erg.kontext = kontext
            erg.quellen = sources
            erg.query_class = daten.get("query_class") or ""
            erg.latency_ms = int(daten.get("latency_ms") or 0)
            erg.cache_hit = bool(daten.get("cache_hit"))
            erg.verified = daten.get("verified")
            erg.system_faithfulness_score = daten.get("faithfulness_score")
            erg.fallback_reason = daten.get("fallback_reason")
            erg.context_sufficient = daten.get("context_sufficient")

            if latenz_vergleich:
                # Fehler der Warm-Messung werden vermerkt; kalte Messung und
                # Metriken bleiben gültig.
                try:
                    daten_warm = frage_stellen(api_base, fall.frage)
                    erg.latency_ms_warm = int(daten_warm.get("latency_ms") or 0)
                    erg.cache_hit_warm = bool(daten_warm.get("cache_hit"))
                except requests.RequestException as exc:
                    erg.fehler = (erg.fehler + " | " if erg.fehler else "") + \
                        f"Warm-Messung fehlgeschlagen: {type(exc).__name__}: {exc}"

            if ragas_ueberspringen:
                reason = "RAGAs-Berechnung übersprungen (Nutzerauswahl)."
                for key in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
                    setattr(erg, f"ragas_{key}", None)
                    setattr(erg, f"{key}_hinweis", reason)
            else:
                scores = asyncio.run(engine.score_row(
                    question=fall.frage, answer=erg.antwort,
                    contexts=kontext, ground_truth=fall.referenzantwort,
                ))
                for k, v in scores.items():
                    setattr(erg, k, v)
                if isinstance(erg.system_faithfulness_score, (int, float)) and \
                        isinstance(erg.ragas_faithfulness, (int, float)):
                    erg.faithfulness_delta = round(
                        erg.system_faithfulness_score - erg.ragas_faithfulness, 4
                    )
        except requests.RequestException as exc:
            erg.fehler = f"{type(exc).__name__}: {exc}"
        except (ValueError, KeyError) as exc:
            erg.fehler = f"Antwort nicht lesbar: {exc}"

        if erg.fehler:
            reason = f"Kein RAGAs-Scoring: Pipeline-Aufruf schlug fehl ({erg.fehler[:120]})."
            for key in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
                setattr(erg, f"{key}_hinweis", reason)

        gesammelt.append(erg)
        _speichern(ziel, lauf_id, gesammelt)

        if fortschritt:
            fortschritt(i, len(faelle), erg)
        yield erg


def _speichern(ziel: Path, lauf_id: str, ergebnisse: list[Ergebnis]) -> None:
    tmp = ziel.with_suffix(".tmp")
    inhalt = {
        "lauf_id": lauf_id,
        "gespeichert": datetime.now().isoformat(timespec="seconds"),
        "anzahl": len(ergebnisse),
        "ergebnisse": [asdict(e) for e in ergebnisse],
    }
    tmp.write_text(json.dumps(inhalt, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(ziel)


def lade_lauf(lauf_id: str) -> list[Ergebnis]:
    p = ERGEBNIS_DIR / f"{lauf_id}.json"
    if not p.exists():
        return []
    daten = json.loads(p.read_text(encoding="utf-8"))
    return [Ergebnis(**e) for e in daten.get("ergebnisse", [])]


def verfuegbare_laeufe() -> list[str]:
    if not ERGEBNIS_DIR.exists():
        return []
    return sorted((p.stem for p in ERGEBNIS_DIR.glob("*.json")), reverse=True)

# Textmarker für korrekt abgelehnte Antworten (F-13, UC-03 A2).
_VERWEIGERUNGS_MARKER = (
    "nicht belegt", "nicht verfügbar", "keine information", "keine relevanten dokumente",
    "nicht im kontext", "ist nicht bekannt", "kann ich nicht beantworten",
    "liegen keine daten vor", "wurde nicht gefunden",
)


def antwort_verweigert(antwort: str) -> bool:
    """
    True, wenn die Antwort fehlende Belege einräumt statt zu antworten.

    Reiner Textvergleich, bewusst konservativ. Ergänzt `context_sufficient`
    und `fallback_reason`, da das System auch bei gefundenem Kontext
    korrekt ablehnen kann, wenn die konkrete Angabe darin fehlt.
    """
    text = (antwort or "").lower()
    return any(marker in text for marker in _VERWEIGERUNGS_MARKER)

# ── Auswertung ────────────────────────────────────────────────────────────
METRIKEN = [
    ("ragas_faithfulness", "Faithfulness"),
    ("ragas_answer_relevancy", "Answer Relevancy"),
    ("ragas_context_precision", "Context Precision"),
    ("ragas_context_recall", "Context Recall"),
]


def kennzahlen(ergebnisse: list[Ergebnis]) -> dict[str, Any]:
    """Aggregiert Fehler, Verifikationsquote, Latenzen und RAGAs-Werte je Metrik."""
    import statistics
    n = len(ergebnisse)
    if not n:
        return {}

    out: dict[str, Any] = {
        "anzahl": n,
        "fehler": sum(1 for e in ergebnisse if e.fehler),
        "verified_quote": 100 * sum(1 for e in ergebnisse if e.verified) / n,
    }
    latenzen = sorted(e.latency_ms for e in ergebnisse if e.latency_ms > 0)
    out["latenz_median"] = latenzen[len(latenzen) // 2] if latenzen else 0
    out["nf03_quote"] = 100 * sum(1 for e in ergebnisse if 0 < e.latency_ms <= 5000) / n

    # Latenzvergleich kalt/warm — nur befüllt, wenn latenz_vergleich=True lief
    warm_latenzen = sorted(e.latency_ms_warm for e in ergebnisse if e.latency_ms_warm)
    kalt_latenzen = sorted(e.latency_ms for e in ergebnisse if e.latency_ms_warm and e.latency_ms > 0)
    out["latenzvergleich"] = None
    if warm_latenzen:
        n_warm = len(warm_latenzen)
        n_hits = sum(1 for e in ergebnisse if e.cache_hit_warm)
        mean_kalt = statistics.fmean(kalt_latenzen) if kalt_latenzen else None
        mean_warm = statistics.fmean(warm_latenzen)
        out["latenzvergleich"] = {
            "n": n_warm,
            "kalt_mean": round(mean_kalt) if mean_kalt else None,
            "kalt_median": kalt_latenzen[len(kalt_latenzen) // 2] if kalt_latenzen else None,
            "warm_mean": round(mean_warm),
            "warm_median": warm_latenzen[n_warm // 2],
            "warm_cache_hit_quote": round(100 * n_hits / n_warm, 1),
            "speedup_faktor": round(mean_kalt / mean_warm, 2) if mean_kalt and mean_warm else None,
        }

    for key, label in METRIKEN:
        werte = [getattr(e, key) for e in ergebnisse if isinstance(getattr(e, key), (int, float))]
        out[key] = {
            "label": label, "n": len(werte),
            "mean": round(statistics.fmean(werte), 3) if werte else None,
            "min": round(min(werte), 3) if werte else None,
            "max": round(max(werte), 3) if werte else None,
        }
    return out

def kennzahlen_segmentiert(faelle: list[Testfall], ergebnisse: list[Ergebnis]) -> dict[str, Any]:
    """
    Kennzahlen getrennt nach beantwortbaren und nicht beantwortbaren Fällen.

    Eine korrekte Ablehnung (F-13) hat regulär niedrige Answer Relevancy.
    Für beantwortbare Fälle werden daher die Metriken (Zielwerte Kap. 4.7.2)
    berechnet, für nicht beantwortbare die Quote korrekter Ablehnungen.
    """
    per_nr = {f.nr: f for f in faelle}
    beantwortbar = [e for e in ergebnisse if not per_nr[e.nr].nicht_beantwortbar]
    nicht_beantwortbar = [e for e in ergebnisse if per_nr[e.nr].nicht_beantwortbar]

    ablehnungs_kennzahlen: dict[str, Any] = {}
    if nicht_beantwortbar:
        n = len(nicht_beantwortbar)
        korrekt = sum(1 for e in nicht_beantwortbar if antwort_verweigert(e.antwort))
        halluziniert = [e for e in nicht_beantwortbar if not antwort_verweigert(e.antwort)]
        ablehnungs_kennzahlen = {
            "anzahl": n,
            "korrekt_abgelehnt": korrekt,
            "korrekt_abgelehnt_quote": round(100 * korrekt / n, 1),
            "vermutlich_halluziniert_nr": [e.nr for e in halluziniert],
        }

    return {
        "beantwortbar": kennzahlen(beantwortbar),
        "nicht_beantwortbar": ablehnungs_kennzahlen,
    }

def nach_gruppe(faelle: list[Testfall], ergebnisse: list[Ergebnis], feld: str, metrik: str) -> list[dict]:
    """Mittelwert einer Metrik je Ausprägung von `feld` (z. B. schwierigkeit, fragetyp)."""
    per_nr = {f.nr: f for f in faelle}
    gruppen: dict[str, list[float]] = {}
    for e in ergebnisse:
        wert = getattr(e, metrik)
        if not isinstance(wert, (int, float)):
            continue
        key = getattr(per_nr[e.nr], feld, "—")
        gruppen.setdefault(key, []).append(wert)

    zeilen = []
    for key, werte in sorted(gruppen.items()):
        zeilen.append({feld: key, "n": len(werte), "Durchschnitt": round(sum(werte) / len(werte), 3)})
    return zeilen
