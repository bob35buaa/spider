#!/usr/bin/env bash
# Fixed entry for E151 route-B evaluation.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
"${PYTHON_BIN}" workspace/core4d/scripts/eval/eval_E151_route_b_hand_surface_contact.py "$@"
