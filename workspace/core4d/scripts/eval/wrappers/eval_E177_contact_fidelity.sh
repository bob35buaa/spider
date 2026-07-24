#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
exec "$PYTHON_BIN" \
  workspace/core4d/scripts/eval/runners/eval_E177_contact_fidelity.py \
  "$@"
