#!/usr/bin/env bash
# E080: generate box025 p1/p2 3cm contact masks without rebuilding SPIDER data.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

RESULT_ROOT="${RESULT_ROOT:-workspace/core4d/results/E080}"
LOGS="${LOGS:-logs/E080}"
mkdir -p "$RESULT_ROOT" "$LOGS"

RESULT_ROOT="$RESULT_ROOT" bash workspace/core4d/data_preprocess/pipeline.sh \
  --case-file workspace/core4d/data_preprocess/cases_E080_box025.tsv \
  --skip-retarget \
  --skip-spider \
  "$@" \
  > "$LOGS/preprocess_box025.log" 2>&1

.venv/bin/python workspace/core4d/scripts/E080/generate_e080_overrides.py \
  > "$LOGS/generate_overrides.log" 2>&1

echo "[$(date '+%H:%M:%S')] === E080 preprocess done ==="
