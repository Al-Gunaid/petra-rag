#!/usr/bin/env bash
# PETRA-RAG: Repository vor dem Push zu GitHub prüfen.
# Aufruf im Repository-Ordner, NACH "git add .":   bash tools/abgabe_pruefen.sh
# Prüft: Pflichtdateien vorhanden, keine Geheimnisse, keine großen Dateien, Platzhalter ersetzt.

set -uo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

FEHLER=0; WARN=0
ok()   { echo "  OK      $*"; }
warn() { echo "  WARNUNG $*"; WARN=$((WARN+1)); }
err()  { echo "  FEHLER  $*"; FEHLER=$((FEHLER+1)); }

git rev-parse --git-dir >/dev/null 2>&1 || { echo "Kein Git-Repository. Zuerst: git init -b main && git add ."; exit 1; }

# Dateien, die committet würden (bereits im Index)
mapfile -t DATEIEN < <(git ls-files --cached)
[[ ${#DATEIEN[@]} -gt 0 ]] || { echo "Index leer. Zuerst: git add ."; exit 1; }
hat() { printf '%s\n' "${DATEIEN[@]}" | grep -qE "$1"; }

echo "1) Pflichtdateien"
for f in README.md .gitignore docker/docker-compose.yml docker/Dockerfile docker/.env.example \
         scripts/modelle_laden.sh scripts/aufwaermen.sh scripts/daten_einspielen.sh scripts/daten_packen.sh \
         evaluation/evaluate_petra.py evaluation/run_eval_schritt.sh evaluation/test_produktberatung.py \
         evaluation/testset/testset_v2.xlsx evaluation/baseline/ragas_evaluation_20260923_213553.xlsx \
         "n8n_workflows/PETRA-RAG_Schritt_7_Einstellungen_wirksam.json" n8n_workflows/PDF-Pipline-Parsing.json; do
  if hat "^${f//./\\.}$"; then ok "$f"; else err "$f fehlt"; fi
done
hat '^data/src/pipeline/.+\.py$' && ok "Backend-Code (data/src/pipeline/)" || err "Backend-Code fehlt"

echo "2) Geheimnisse"
hat '(^|/)\.env$' && err "docker/.env ist im Index → git rm --cached docker/.env" || ok "keine .env"
hat 'n8n_data/' && err "n8n_data/ im Index (enthält Credentials)" || ok "kein n8n_data"
TREFFER=$(git grep -nIiE '(api[_-]?key|secret|password|passwort|token)[[:space:]]*[:=][[:space:]]*["'"'"']?[A-Za-z0-9/_+\-]{8,}' --cached \
          -- ':!*.example' ':!README.md' ':!tools/abgabe_pruefen.sh' 2>/dev/null | head -5)
[[ -n "$TREFFER" ]] && { warn "mögliche Zugangsdaten im Code – bitte ansehen:"; echo "$TREFFER" | sed 's/^/            /'; } || ok "keine offensichtlichen Zugangsdaten"
# n8n-Workflows: eingebettete Credential-IDs sind unkritisch, Werte nicht
if git grep -qIE '"(password|apiKey)"[[:space:]]*:[[:space:]]*"[^"]{4,}"' --cached -- 'n8n_workflows/*.json' 2>/dev/null; then
  warn "n8n-Workflow enthält Passwort-/apiKey-Werte"; fi

echo "3) Dateigrößen (GitHub: Warnung ab 50 MB, Abbruch ab 100 MB)"
GROSS=0
for f in "${DATEIEN[@]}"; do
  [[ -f "$f" ]] || continue
  g=$(stat -c %s "$f")
  if (( g > 100*1024*1024 )); then err "$f ist $((g/1024/1024)) MB"; GROSS=1
  elif (( g > 20*1024*1024 )); then warn "$f ist $((g/1024/1024)) MB (gehört evtl. nach Sciebo)"; GROSS=1; fi
done
[[ $GROSS -eq 0 ]] && ok "keine Datei > 20 MB"
N_PDF=$(printf '%s\n' "${DATEIEN[@]}" | grep -ciE '\.pdf$' || true)
(( N_PDF > 2 )) && warn "$N_PDF PDF-Dateien im Index – Datenblätter gehören nach Sciebo"
echo "  Gesamt: ${#DATEIEN[@]} Dateien, $(printf '%s\0' "${DATEIEN[@]}" | du -ch --files0-from=- 2>/dev/null | tail -1 | cut -f1)"

echo "4) Platzhalter in README.md"
for p in '<benutzer>' '<SCIEBO-LINK>' '<Frontend-Ordner>'; do
  grep -qF "$p" README.md && warn "README enthält noch $p" || ok "$p ersetzt"
done

echo
if (( FEHLER > 0 )); then echo "Ergebnis: $FEHLER Fehler, $WARN Warnungen → bitte beheben, dann erneut prüfen."; exit 1
else echo "Ergebnis: keine Fehler, $WARN Warnungen. Bereit für git commit / git push."; fi
