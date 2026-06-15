#!/usr/bin/env bash
# Fixed entry for E163 narrow symmetric surfaceBand evaluation.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
"${PYTHON_BIN}" workspace/core4d/scripts/eval/runners/eval_E163_narrow_surface_band.py "$@"
