#!/usr/bin/env bash
set -euo pipefail

RAW_DIR="${1:-./data/raw}"
TRAIN_CSV="${2:-./data/processed/train.csv}"
OUTPUT_DIR="${3:-./models/bert}"

bash scripts/download_data.sh "$RAW_DIR"
bash scripts/train_bert.sh "$TRAIN_CSV" "$OUTPUT_DIR"
