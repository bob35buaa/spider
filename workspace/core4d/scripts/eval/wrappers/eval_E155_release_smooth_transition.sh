#!/usr/bin/env bash
# Fixed entry for E155 release smooth-transition evaluation.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
"${PYTHON_BIN}" workspace/core4d/scripts/eval/eval_E155_release_smooth_transition.py "$@"
