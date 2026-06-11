#!/usr/bin/env bash
# Fixed entry for E156 clean8 gate/decay evaluation.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
"${PYTHON_BIN}" workspace/core4d/scripts/eval/eval_E156_clean8_gate_decay.py "$@"
