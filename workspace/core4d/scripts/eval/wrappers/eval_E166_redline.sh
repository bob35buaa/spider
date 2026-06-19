#!/usr/bin/env bash
# Fixed entry for E166 Phase0 redline predictive audit.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
"${PYTHON_BIN}" workspace/core4d/scripts/eval/runners/eval_E166_redline_predictive.py "$@"
