#!/usr/bin/env bash
# PETRA-RAG: Datenpaket aus Sciebo nach /mnt/petra-rag entpacken.
#
# Aufruf:
#   bash scripts/daten_einspielen.sh <Ordner mit den Sciebo-Dateien>
#
# Umgebungsvariable:
#   ZIEL   Datenordner (Standard: /mnt/petra-rag)
#
# Ablauf: Prüfsummen kontrollieren → geteilte Archive zusammensetzen → entpacken.

set -euo pipefail

PAKET="${1:-}"
ZIEL="${ZIEL:-/mnt/petra-rag}"

if [[ -z "$PAKET" || ! -d "$PAKET" ]]; then
  echo "Aufruf: bash scripts/daten_einspielen.sh <Ordner mit den Sciebo-Dateien>" >&2
  exit 1
fi
PAKET="$(cd "$PAKET" && pwd)"

[[ -f "$PAKET/SHA256SUMS" ]] || { echo "FEHLER: $PAKET/SHA256SUMS fehlt – Download unvollständig?" >&2; exit 1; }

echo ">> Prüfe Prüfsummen ..."
( cd "$PAKET" && sha256sum -c SHA256SUMS ) || {
  echo "FEHLER: Mindestens eine Datei ist beschädigt oder fehlt. Bitte erneut aus Sciebo laden." >&2
  exit 1
}

BENUTZER="${USER:-$(id -un)}"
if [[ ! -d "$ZIEL" ]]; then
  echo ">> Lege $ZIEL an ..."
  if ! mkdir -p "$ZIEL" 2>/dev/null; then
    sudo mkdir -p "$ZIEL"
    sudo chown "$BENUTZER" "$ZIEL"
  fi
fi
[[ -w "$ZIEL" ]] || { echo "FEHLER: Keine Schreibrechte in $ZIEL → sudo chown -R $BENUTZER $ZIEL" >&2; exit 1; }

if command -v pigz >/dev/null 2>&1; then ENTP="pigz -dc"; else ENTP="gzip -dc"; fi

# Archivnamen sammeln (geteilte Teile auf den Basisnamen zurückführen)
mapfile -t ARCHIVE < <(cd "$PAKET" && ls ./*.tar.gz ./*.tar.gz.part-* 2>/dev/null \
  | sed -E 's#^\./##; s#\.part-[0-9]+$##' | sort -u)

[[ ${#ARCHIVE[@]} -gt 0 ]] || { echo "FEHLER: Keine Archive in $PAKET gefunden." >&2; exit 1; }

for a in "${ARCHIVE[@]}"; do
  ordner="${a%.tar.gz}"
  if [[ "$ordner" != "_dateien" && -d "$ZIEL/$ordner" ]] && compgen -G "$ZIEL/$ordner/*" > /dev/null; then
    echo "WARNUNG: $ZIEL/$ordner ist nicht leer – Inhalt wird überschrieben/ergänzt."
  fi
  echo ">> Entpacke $a ..."
  if [[ -f "$PAKET/$a" ]]; then
    $ENTP "$PAKET/$a" | tar -xf - -C "$ZIEL"
  else
    cat "$PAKET/$a".part-* | $ENTP | tar -xf - -C "$ZIEL"
  fi
done

mkdir -p "$ZIEL"/{pdfs,processed,chroma_db,ollama,hf_cache,eval}

echo
echo "Fertig. Inhalt von $ZIEL:"
du -sh "$ZIEL"/* 2>/dev/null
echo
echo "Weiter mit README → Installation, Schritt 4 (Konfiguration anlegen)."
