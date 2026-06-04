#!/usr/bin/env bash
# E126: export E125 fragment-only pair into Holosoma motion adapter artifacts.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [ ! -x "${PYTHON_BIN}" ]; then
  PYTHON_BIN="python3"
fi

"${PYTHON_BIN}" workspace/core4d/scripts/E126/export_holosoma_fragment_adapter.py "$@"
