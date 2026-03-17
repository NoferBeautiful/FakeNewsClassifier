#!/usr/bin/env bash
set -euo pipefail

NEW_DATA_CSV="${1:-./data/processed/new_batch.csv}"
MODEL_DIR="${2:-}"
OUTPUT_DIR="${3:-./models/bert}"

if [ -z "$MODEL_DIR" ]; then
  echo "Usage: bash scripts/update_bert.sh <new_data_csv> <model_dir> [output_dir]"
  exit 1
fi

python update_bert.py \
  --new-data-csv "$NEW_DATA_CSV" \
  --model-dir "$MODEL_DIR" \
  --output-dir "$OUTPUT_DIR"
