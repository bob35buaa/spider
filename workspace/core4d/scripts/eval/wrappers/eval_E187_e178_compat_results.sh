#!/usr/bin/env bash
# E187 Gate S0 device-routed contract audit; all-array exactness is diagnostic only.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-uv run python}"
$PYTHON_BIN workspace/core4d/scripts/experiments/E187/test_e178_compat_eval.py
$PYTHON_BIN workspace/core4d/scripts/eval/runners/eval_E187_e178_compat.py "$@"
