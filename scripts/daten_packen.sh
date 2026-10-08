#!/usr/bin/env bash
# PETRA-RAG: Datenordner für die Abgabe über Sciebo verpacken.
#
# Aufruf (auf dem Server, im Repository-Ordner):
#   bash scripts/daten_packen.sh                 # ohne Modellgewichte (empfohlen)
#   bash scripts/daten_packen.sh --mit-modellen  # inkl. ollama/ und hf_cache/ (sehr groß)
#
# Umgebungsvariablen:
#   QUELLE   Datenordner           (Standard: /mnt/petra-rag)
#   ZIEL     Ausgabeordner         (Standard: $HOME/petra-abgabe-daten)
#   TEIL     Größe pro Teildatei   (Standard: 4G; größere Archive werden geteilt)
#
# Die petra-Container werden während des Packens angehalten, damit ChromaDB
# und der BM25-Index in einem konsistenten Zustand gesichert werden.
# Danach werden sie automatisch wieder gestartet.

set -euo pipefail

QUELLE="${QUELLE:-/mnt/petra-rag}"
ZIEL="${ZIEL:-$HOME/petra-abgabe-daten}"
TEIL="${TEIL:-4G}"
MIT_MODELLEN=0
[[ "${1:-}" == "--mit-modellen" ]] && MIT_MODELLEN=1

[[ -d "$QUELLE" ]] || { echo "FEHLER: $QUELLE existiert nicht." >&2; exit 1; }
mkdir -p "$ZIEL"
if compgen -G "$ZIEL/*" > /dev/null; then
  echo "FEHLER: $ZIEL ist nicht leer. Bitte leeren oder anderes ZIEL setzen." >&2
  exit 1
fi

# Kompressor: pigz (parallel) wenn vorhanden, sonst gzip
if command -v pigz >/dev/null 2>&1; then KOMP="pigz"; else KOMP="gzip"; fi

# Container anhalten und beim Beenden (auch bei Fehler) wieder starten
LAUFEND=""
if command -v docker >/dev/null 2>&1; then
  LAUFEND="$(docker ps --format '{{.Names}}' | grep -E '^petra' || true)"
fi
neu_starten() {
  if [[ -n "$LAUFEND" ]]; then
    echo ">> Starte Container wieder: $(echo $LAUFEND)"
    # shellcheck disable=SC2086
    docker start $LAUFEND >/dev/null
  fi
}
trap neu_starten EXIT
if [[ -n "$LAUFEND" ]]; then
  echo ">> Halte Container an: $(echo $LAUFEND)"
  # shellcheck disable=SC2086
  docker stop $LAUFEND >/dev/null
fi

AUSLASSEN=()
if [[ $MIT_MODELLEN -eq 0 ]]; then
  AUSLASSEN=(ollama hf_cache)
  echo ">> Modellordner werden ausgelassen (ollama, hf_cache) – sie werden mit scripts/modelle_laden.sh neu geladen."
fi

ist_ausgelassen() {
  local n="$1" a
  for a in "${AUSLASSEN[@]}"; do [[ "$n" == "$a" ]] && return 0; done
  return 1
}

cd "$QUELLE"
LOSE_DATEIEN=()
for eintrag in * .[!.]*; do
  [[ -e "$eintrag" ]] || continue
  if [[ -d "$eintrag" ]]; then
    ist_ausgelassen "$eintrag" && continue
    archiv="$ZIEL/$eintrag.tar.gz"
    echo ">> Packe $eintrag/ ($(du -sh "$eintrag" | cut -f1)) ..."
    tar -cf - "$eintrag" | $KOMP > "$archiv"
  else
    LOSE_DATEIEN+=("$eintrag")
  fi
done

if [[ ${#LOSE_DATEIEN[@]} -gt 0 ]]; then
  echo ">> Packe einzelne Dateien im Hauptordner: ${LOSE_DATEIEN[*]}"
  tar -cf - "${LOSE_DATEIEN[@]}" | $KOMP > "$ZIEL/_dateien.tar.gz"
fi

# Große Archive teilen (Sciebo-Upload im Browser ist so zuverlässiger)
cd "$ZIEL"
for a in *.tar.gz; do
  groesse=$(stat -c %s "$a")
  grenze=$(numfmt --from=iec "$TEIL")
  if (( groesse > grenze )); then
    echo ">> Teile $a in Stücke zu $TEIL ..."
    split -b "$TEIL" -d -a 3 "$a" "$a.part-"
    rm "$a"
  fi
done

echo ">> Berechne Prüfsummen ..."
sha256sum ./*.tar.gz* > SHA256SUMS

{
  echo "PETRA-RAG – Datenpaket für die Abgabe"
  echo "Erstellt:  $(date '+%Y-%m-%d %H:%M')"
  echo "Quelle:    $QUELLE"
  echo "Modelle:   $([[ $MIT_MODELLEN -eq 1 ]] && echo enthalten || echo 'nicht enthalten (scripts/modelle_laden.sh)')"
  echo
  echo "Einspielen: bash scripts/daten_einspielen.sh <Ordner mit diesen Dateien>"
  echo
  echo "Ordnergrößen im Original:"
  (cd "$QUELLE" && du -sh -- */ 2>/dev/null | grep -v -E '^\S+\s+(ollama|hf_cache)/$' || true)
  echo
  echo "Dateien im Paket:"
  ls -lh --time-style=+ ./*.tar.gz* | awk '{print "  " $5 "\t" $NF}'
  echo
  echo "Gesamtgröße: $(du -ch ./*.tar.gz* | tail -1 | cut -f1)"
} > MANIFEST.txt

echo
cat MANIFEST.txt
echo
echo "Fertig. Den gesamten Inhalt von $ZIEL nach Sciebo hochladen."
