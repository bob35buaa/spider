#!/usr/bin/env bash
# E081: create derived leg/foot-object collision cases and overrides.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

RESULT_ROOT="${RESULT_ROOT:-workspace/core4d/results/E081}"
LOGS="${LOGS:-logs/E081}"
mkdir -p "$RESULT_ROOT" "$LOGS"

.venv/bin/python workspace/core4d/scripts/E081/create_legobj_cases.py --force \
  > "$LOGS/create_legobj_cases.log" 2>&1

.venv/bin/python workspace/core4d/scripts/E081/generate_e081_overrides.py \
  --result-root "$RESULT_ROOT" \
  > "$LOGS/generate_overrides.log" 2>&1

echo "[$(date '+%H:%M:%S')] === E081 preprocess done ==="
