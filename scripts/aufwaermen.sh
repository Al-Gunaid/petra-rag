#!/usr/bin/env bash
# PETRA-RAG: Nach dem Start Dienste prüfen und Modelle in den Grafikspeicher laden.
#
# Aufruf:
#   bash scripts/aufwaermen.sh            # Gesundheitscheck + zwei Testfragen
#   bash scripts/aufwaermen.sh --preis    # zusätzlich Preis-Agent mit Web-Suche
#
# Die erste Anfrage nach dem Start dauert 1–2 Minuten (Modelle werden geladen),
# danach etwa 10–15 Sekunden.

set -uo pipefail
API="${PETRA_API:-http://localhost:8001}"

echo ">> Warte auf das Backend ($API) ..."
for i in $(seq 1 60); do
  curl -s -m 5 "$API/ops/health/deep" > /tmp/petra_health.json 2>/dev/null && break
  sleep 5
done
if ! python3 -c "import json,sys; d=json.load(open('/tmp/petra_health.json')); sys.exit(0 if d.get('ok') else 1)" 2>/dev/null; then
  echo "WARNUNG: Gesundheitscheck nicht ok:"
  python3 -m json.tool /tmp/petra_health.json 2>/dev/null || cat /tmp/petra_health.json
  echo "Fehlende Modelle? → bash scripts/modelle_laden.sh"
else
  echo "   Gesundheitscheck: ok"
fi

frage() {   # $1 = Frage, $2 = web_agent_enabled (true/false)
  local t0 t1
  t0=$(date +%s)
  curl -s -m 300 -X POST "$API/query" -H 'Content-Type: application/json' \
    -d "{\"query\":\"$1\",\"temperature\":0,\"top_k\":6,\"web_agent_enabled\":${2:-false}}" \
    | python3 -c "import json,sys; d=json.load(sys.stdin); print('   Antwort:', (d.get('answer') or d)[:160].replace(chr(10),' '))" 2>/dev/null \
    || echo "   (keine gültige Antwort – Workflow 'Schritt 7' in n8n aktiv?)"
  t1=$(date +%s)
  echo "   Dauer: $((t1 - t0)) s"
}

echo ">> Testfrage 1 (lädt Modelle, 1–2 min):"
frage "Wie hoch ist der Nennstrom der WDU 2.5?" false
echo ">> Testfrage 2:"
frage "Was bedeutet die Anschlusstechnik PUSH IN?" false
if [[ "${1:-}" == "--preis" ]]; then
  echo ">> Preis-Agent mit Web-Suche:"
  frage "Was kostet das PRO ECO3 240W 24V 10A II?" true
fi
echo
echo "Fertig. Oberfläche: http://localhost:8501"
