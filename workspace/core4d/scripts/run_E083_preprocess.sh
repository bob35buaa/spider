#!/usr/bin/env bash
# E083: create upper-body-object derived cases and overrides.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

RESULT_ROOT="${RESULT_ROOT:-workspace/core4d/results/E083}"
LOGS="${LOGS:-logs/E083}"
mkdir -p "$RESULT_ROOT" "$LOGS"

.venv/bin/python workspace/core4d/scripts/E083/create_upperobj_cases.py --force \
  > "$LOGS/create_upperobj_cases.log" 2>&1

.venv/bin/python workspace/core4d/scripts/E083/generate_e083_overrides.py \
  --result-root "$RESULT_ROOT" \
  > "$LOGS/generate_overrides.log" 2>&1

echo "[$(date '+%H:%M:%S')] === E083 preprocess done ==="
