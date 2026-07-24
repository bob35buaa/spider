#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-full}"
shift || true
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
exec "$PYTHON_BIN" \
  workspace/core4d/scripts/eval/runners/eval_E175_nonbox_multigeom.py \
  "$MODE" "$@"
