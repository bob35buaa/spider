#!/usr/bin/env bash
# Fixed entry for E161 surfaceBand release ablation clean8 evaluation.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
"${PYTHON_BIN}" workspace/core4d/scripts/eval/runners/eval_E161_surface_release_ablation.py "$@"
