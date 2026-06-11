#!/usr/bin/env bash
# Fixed entry for E152 hand-object physics gate evaluation.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
"${PYTHON_BIN}" workspace/core4d/scripts/eval/eval_E152_axis1_hand_object_physics_gate.py "$@"
