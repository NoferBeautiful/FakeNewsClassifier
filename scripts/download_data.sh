#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-./data/raw}"

python download_data.py --output-dir "$OUTPUT_DIR"
