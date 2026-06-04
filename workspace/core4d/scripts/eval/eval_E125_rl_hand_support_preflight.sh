#!/usr/bin/env bash
# E125: build RL hand-support preflight/export artifacts. No training is launched.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [ ! -x "${PYTHON_BIN}" ]; then
  PYTHON_BIN="python3"
fi

"${PYTHON_BIN}" workspace/core4d/scripts/E125/build_rl_hand_support_preflight.py "$@"
