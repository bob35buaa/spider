#!/usr/bin/env bash
# E082: create D003 Box021 leg-object derived cases and overrides.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

RESULT_ROOT="${RESULT_ROOT:-workspace/core4d/results/E082}"
LOGS="${LOGS:-logs/E082}"
mkdir -p "$RESULT_ROOT" "$LOGS"

.venv/bin/python workspace/core4d/scripts/E082/create_legobj_cases.py --force \
  > "$LOGS/create_legobj_cases.log" 2>&1

.venv/bin/python workspace/core4d/scripts/E082/generate_e082_overrides.py \
  --result-root "$RESULT_ROOT" \
  > "$LOGS/generate_overrides.log" 2>&1

echo "[$(date '+%H:%M:%S')] === E082 preprocess done ==="
