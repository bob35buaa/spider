#!/usr/bin/env bash
# Fixed entry for E159 gateA + surfaceBand-A2 clean6 evaluation.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
"${PYTHON_BIN}" workspace/core4d/scripts/eval/eval_E159_gate_surface_band_A2.py "$@"
