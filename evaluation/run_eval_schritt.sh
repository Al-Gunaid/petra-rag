#!/bin/bash
# ============================================================
# run_eval_schritt.sh — Evaluation eines Optimierungsschritts vollständig und unbeaufsichtigt
#
# Start (läuft auch nach Trennen der SSH-Verbindung weiter):
#   nohup bash evaluation/run_eval_schritt.sh s7 > /dev/null 2>&1 &
#   Teilmenge: ... run_eval_schritt.sh s1t evaluation/testset/testset_s1.xlsx
# Ordner mit docker-compose.yml (Standard: <Repository>/docker):
#   PETRA_DOCKER_DIR=/pfad/zu/petra-rag/docker nohup bash evaluation/run_eval_schritt.sh s7 ...
# Fortschritt:
#   tail -5 /mnt/petra-rag/eval_${TAG}_ablauf.log
#
# Ablauf
#   Vorbereitung  Dateien prüfen, alte Antworten archivieren, Judge-Modelle
#                 entladen, Pipeline aufwärmen, aktiven Workflow protokollieren
#   Phase 1  Antworten erzeugen. Fälle mit Fehler werden einmal neu angefragt.
#   Umbau    petra_ingestion stoppen (gibt Grafikspeicher frei; für den Judge
#            nicht nötig, die Antworten liegen schon vor) und Ollama auf
#            OLLAMA_NUM_PARALLEL=1 -> qwen2.5:32b passt (fast) ganz auf die GPU
#   Phase 2  RAGAS, Judge qwen2.5:32b, Sicht = Prompt-Blöcke mit Quellen-Kopf
#   Rückbau  petra_ingestion starten und aufwärmen
# ============================================================
set -u
# Optimierungsreihe: EIN Lauf je Schritt, immer gleich gemessen
# (75 Fälle, Judge qwen2.5:32b, OLLAMA_NUM_PARALLEL=1, Sicht "bloecke").
TAG=${1:?"Bitte Schrittnamen angeben, z. B.: bash run_eval_schritt.sh s7"}
# Optional 2. Argument: Testset-Datei auf dem Host (z. B. Problem- + Kontrollfälle)
TS_HOST=${2:-}
if [ -n "$TS_HOST" ]; then TS=$(basename "$TS_HOST"); docker cp "$TS_HOST" "petra_frontend:/tmp/$TS"; else TS=testset_v2.xlsx; fi
D=${PETRA_DOCKER_DIR:-$(cd "$(dirname "$0")/../docker" && pwd)}
LOG=/mnt/petra-rag/eval_${TAG}_ablauf.log
ANS=/data/eval_${TAG}.answers.jsonl          # Pfad im Container petra_frontend
COMMON="cd /tmp && python3 evaluate_petra.py --testset $TS --webhook http://n8n:5678/webhook/petra-query --api http://ingestion-api:8001 --ollama http://ollama:11434 --baseline ragas_evaluation_20260923_213553.xlsx"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

ollama_parallel() {   # $1 = 1 oder 4
  sed -i "s/OLLAMA_NUM_PARALLEL=[0-9]*/OLLAMA_NUM_PARALLEL=$1/" "$D/docker-compose.yml"
  (cd "$D" && docker compose --env-file .env -f docker-compose.yml up -d ollama) >> "$LOG" 2>&1
  sleep 20
  log "Ollama läuft mit OLLAMA_NUM_PARALLEL=$(docker exec petra_ollama printenv OLLAMA_NUM_PARALLEL)"
}

modelle() {   # geladene Modelle mit GPU/CPU-Anteil
  docker exec petra_ollama ollama ps 2>/dev/null | awk 'NR>1{printf "%s (%s %s, %s %s)  ", $1, $3, $4, $5, $6}'
}

zusammenfassung() {   # $1 = Ergebnis-Excel im Container
  docker exec petra_frontend python3 -c "
import pandas as pd
s = pd.read_excel('$1', sheet_name='Zusammenfassung')
for r in s[s['Lauf'].astype(str).str.startswith('Neu')].itertuples():
    print(f'    {r.Metrik}: {r.Wert} (n={r.n})')
" >> "$LOG" 2>&1
}

rueckbau() {
  log "Rückbau: Normalbetrieb herstellen"
  ollama_parallel 1          # Endstand: dauerhaft 1 (Einzelplatz, siehe Optimierungsprotokoll)
  docker start petra_ingestion >> "$LOG" 2>&1
  sleep 30
  curl -s -m 120 -X POST localhost:8001/agent/price -H 'Content-Type: application/json' \
       -d '{"query":"Was kostet der WDU 2.5 aktuell?"}' > /dev/null && log "petra_ingestion aufgewärmt"
}

log "=== Evaluation ${TAG} gestartet ==="

# ── Vorbereitung ─────────────────────────────────────────────
if ! docker exec petra_frontend sh -c "cd /tmp && ls evaluate_petra.py $TS ragas_evaluation_20260923_213553.xlsx" >> "$LOG" 2>&1; then
  log "FEHLER: Dateien in petra_frontend:/tmp fehlen (siehe Zeile darüber). Abbruch."
  exit 1
fi
# Alte Antworten würden sonst fortgesetzt (Workflow-Versionen gemischt)
docker exec petra_frontend sh -c "if [ -f $ANS ]; then mv $ANS $ANS.alt_\$(date +%Y%m%d_%H%M%S); echo 'Alte Antworten archiviert'; fi" >> "$LOG" 2>&1
# Judge-Modelle aus dem Grafikspeicher entfernen, damit llama3.1:8b ganz auf der GPU läuft
for m in qwen2.5:32b qwen2.5:14b; do
  curl -s -m 60 localhost:11434/api/generate -d "{\"model\":\"$m\",\"keep_alive\":0}" > /dev/null
done
curl -s -m 240 -X POST localhost:8001/query -H 'Content-Type: application/json' \
     -d '{"query":"Welche Schutzart hat die Reihenklemme WDU 4?"}' > /dev/null && log "Aufgewärmt: Anfrage beantwortet"
curl -s -m 120 -X POST localhost:8001/agent/price -H 'Content-Type: application/json' \
     -d '{"query":"Was kostet der WDU 2.5 aktuell?"}' > /dev/null && log "Aufgewärmt: Preisagent beantwortet"
log "Geladene Modelle: $(modelle)"
AKTIV=$(docker exec petra_n8n n8n list:workflow --active=true 2>/dev/null | grep -iv 'ingest' | tr '\n' ' ')
log "Aktive Workflows: $AKTIV"
# Schutz: Lauf "sN..." misst nur, wenn genau der Workflow "Schritt N:" aktiv ist
NR=$(echo "$TAG" | sed -n 's/^s\([0-9]\+\).*/\1/p')
if [ -n "$NR" ]; then
  if ! echo "$AKTIV" | grep -q "Schritt ${NR}:"; then
    log "FEHLER: Workflow 'Schritt ${NR}' ist nicht aktiv – Abbruch, damit nicht der falsche Stand gemessen wird."
    exit 1
  fi
  if [ "$(echo "$AKTIV" | grep -o 'Schritt [0-9]\+:' | sort -u | wc -l)" != "1" ]; then
    log "FEHLER: mehrere 'Schritt'-Workflows aktiv ($AKTIV) – bitte nur Schritt ${NR} aktiv lassen. Abbruch."
    exit 1
  fi
fi

# ── Phase 1 ──────────────────────────────────────────────────
log "Phase 1: Antworten erzeugen"
if ! docker exec petra_frontend sh -c "$COMMON --out /data/eval_${TAG}.xlsx --skip-ragas > /data/eval_${TAG}_run.log 2>&1"; then
  log "FEHLER in Phase 1 – siehe /mnt/petra-rag/eval_${TAG}_run.log. Abbruch."
  exit 1
fi
FEHLER=$(docker exec petra_frontend python3 -c "import json; print(sum(1 for z in open('$ANS', encoding='utf-8') if z.strip() and json.loads(z)['response'].get('_error')))")
if [ "${FEHLER:-0}" != "0" ]; then
  log "Phase 1: $FEHLER Antworten mit Fehler – werden einmal neu angefragt"
  docker exec petra_frontend python3 -c "import json; p='$ANS'; z=[l for l in open(p, encoding='utf-8') if l.strip()]; open(p, 'w', encoding='utf-8').writelines(l for l in z if not json.loads(l)['response'].get('_error'))"
  docker exec petra_frontend sh -c "$COMMON --out /data/eval_${TAG}.xlsx --skip-ragas >> /data/eval_${TAG}_run.log 2>&1"
fi
N=$(docker exec petra_frontend sh -c "grep -c . $ANS")
log "Phase 1 fertig: $N Antworten"
zusammenfassung /data/eval_${TAG}.xlsx

# ── Umbau für den Judge ──────────────────────────────────────
log "Umbau für den Judge: petra_ingestion stoppen, Ollama auf 1 parallele Anfrage"
docker stop petra_ingestion >> "$LOG" 2>&1
ollama_parallel 1

# ── Phase 2 ──────────────────────────────────────────────────
log "Phase 2: RAGAS (Sicht: Prompt-Blöcke mit Quellen-Kopf)"
if docker exec petra_frontend sh -c "$COMMON --out /data/eval_${TAG}.xlsx --answers $ANS --judge ollama:qwen2.5:32b > /data/eval_${TAG}_judge.log 2>&1"; then
  log "Phase 2 fertig: /mnt/petra-rag/eval_${TAG}.xlsx"
  zusammenfassung /data/eval_${TAG}.xlsx
else
  log "FEHLER in Phase 2 – siehe /mnt/petra-rag/eval_${TAG}_judge.log"
fi

rueckbau
log "=== Evaluation ${TAG} beendet ==="
