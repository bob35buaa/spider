#!/usr/bin/env bash
# Fixed entry for E167 z-only axis and Holosoma-style gate evaluation.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-preflight}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

"${PYTHON_BIN}" workspace/core4d/scripts/eval/runners/eval_E167_zonly_axis_audit.py "$MODE"
"${PYTHON_BIN}" workspace/core4d/scripts/eval/runners/eval_E167_holosoma_zgate.py "$MODE"
if [ "$MODE" = "full" ]; then
  "${PYTHON_BIN}" workspace/core4d/scripts/eval/runners/eval_E167_cem_metrics.py "$MODE"
fi
