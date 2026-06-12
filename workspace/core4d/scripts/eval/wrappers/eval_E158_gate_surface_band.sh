#!/usr/bin/env bash
# Fixed entry for E158 gateA + surfaceBand-A clean6 evaluation.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
"${PYTHON_BIN}" workspace/core4d/scripts/eval/eval_E158_gate_surface_band.py "$@"
