#!/usr/bin/env bash
# Fixed entry for E165-D peak-margin rerank evaluation.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
"${PYTHON_BIN}" workspace/core4d/scripts/eval/runners/eval_E165D_peak_margin_rerank.py "$@"
