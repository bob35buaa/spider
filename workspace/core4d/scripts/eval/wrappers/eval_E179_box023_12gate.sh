#!/usr/bin/env bash
# Evaluate all 16 E179 rows with the frozen 12-gate adapter and write report.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-full}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
"$PYTHON_BIN" \
  workspace/core4d/scripts/eval/runners/eval_E179_box023_12gate.py \
  "$MODE" --require-all
if [ "$MODE" = "full" ]; then
  "$PYTHON_BIN" \
    workspace/core4d/scripts/eval/reports/gen_E179_box023_paired_report.py
fi
