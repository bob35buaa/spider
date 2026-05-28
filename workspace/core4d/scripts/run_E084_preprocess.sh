#!/usr/bin/env bash
# E084: generate constraint-group overrides and copy contact masks.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

RESULT_ROOT="${RESULT_ROOT:-workspace/core4d/results/E084}"
LOGS="${LOGS:-logs/E084}"
mkdir -p "$RESULT_ROOT" "$LOGS"

.venv/bin/python workspace/core4d/scripts/E084/generate_e084_overrides.py \
  --result-root "$RESULT_ROOT" \
  > "$LOGS/generate_overrides.log" 2>&1

echo "[$(date '+%H:%M:%S')] === E084 preprocess done ==="
