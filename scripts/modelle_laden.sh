#!/usr/bin/env bash
# PETRA-RAG: Sprachmodelle und Reranker einmalig herunterladen.
#
# Aufruf (Container müssen laufen: cd docker && docker compose up -d):
#   bash scripts/modelle_laden.sh          # llama3.1:8b, bge-m3, Reranker
#   bash scripts/modelle_laden.sh --eval   # zusätzlich qwen2.5:32b (Judge der Evaluation, ≈ 20 GB)
#
# Die Ollama-Modelle landen in /mnt/petra-rag/ollama, der Reranker in
# /mnt/petra-rag/hf_cache (HF_HOME im Container petra_ingestion).

set -euo pipefail

laeuft() { docker ps --format '{{.Names}}' | grep -qx "$1"; }
for c in petra_ollama petra_ingestion; do
  laeuft "$c" || { echo "FEHLER: Container $c läuft nicht. Zuerst: cd docker && docker compose up -d" >&2; exit 1; }
done

MODELLE=(llama3.1:8b bge-m3:latest)
[[ "${1:-}" == "--eval" ]] && MODELLE+=(qwen2.5:32b)

for m in "${MODELLE[@]}"; do
  echo ">> Ollama: $m"
  docker exec petra_ollama ollama pull "$m"
done

echo ">> Reranker: BAAI/bge-reranker-v2-m3"
docker exec petra_ingestion python -c "
from sentence_transformers import CrossEncoder
CrossEncoder('BAAI/bge-reranker-v2-m3')
print('Reranker geladen')
"

echo
docker exec petra_ollama ollama list
echo
echo "Fertig. Für Betrieb ohne Internet kann jetzt in docker/.env TRANSFORMERS_OFFLINE=1 gesetzt werden."
