#!/usr/bin/env bash
# E026: create E081-style leg/object baseline cases for the 13case union.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

VARIANTS_FILE="${VARIANTS_FILE:-workspace/core4d_collab_retarget/scripts/E026/e081_full_variants.tsv}"
RESULT_ROOT="${RESULT_ROOT:-workspace/core4d_collab_retarget/results/E026_E081_full}"
LOGS="${LOGS:-logs/core4d_collab_retarget/E026_E081_full}"
mkdir -p "$RESULT_ROOT" "$LOGS"

.venv/bin/python workspace/core4d/scripts/E081/create_legobj_cases.py \
  --variants "$VARIANTS_FILE" --force \
  > "$LOGS/create_legobj_cases.log" 2>&1

.venv/bin/python workspace/core4d/scripts/E081/generate_e081_overrides.py \
  --variants "$VARIANTS_FILE" \
  --result-root "$RESULT_ROOT" \
  > "$LOGS/generate_overrides.log" 2>&1

echo "[$(date '+%H:%M:%S')] === E026 E081 full preprocess done ==="
