#!/usr/bin/env bash
# Fixed entrypoint for E122 snap warmstart evaluation.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-smoke}"
shift || true

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="${PYTHON:-python3}"
fi

"$PYTHON_BIN" workspace/core4d/scripts/eval/eval_E122_snap_warmstart.py --stage "$STAGE" "$@"
