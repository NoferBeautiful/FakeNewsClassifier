#!/usr/bin/env bash
set -euo pipefail

TRAIN_CSV="${1:-./data/processed/train.csv}"
OUTPUT_DIR="${2:-./models/bert}"

python train_bert.py \
  --train-csv "$TRAIN_CSV" \
  --output-dir "$OUTPUT_DIR"
