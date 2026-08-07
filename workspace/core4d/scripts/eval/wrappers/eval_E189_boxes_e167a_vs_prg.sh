#!/usr/bin/env bash
# Evaluate all 43 E189 rows with the frozen 12-gate adapter, paired against
# the E172/E173 PRG baseline, then write the per-object comparison report.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-full}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
"$PYTHON_BIN" \
  workspace/core4d/scripts/eval/runners/eval_E189_boxes_e167a_vs_prg.py \
  "$MODE" --require-all
if [ "$MODE" = "full" ]; then
  "$PYTHON_BIN" \
    workspace/core4d/scripts/eval/reports/gen_E189_boxes_comparison.py
fi
