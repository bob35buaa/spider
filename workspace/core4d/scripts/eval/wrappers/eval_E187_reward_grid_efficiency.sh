#!/usr/bin/env bash
# E187 A1 same-tape R/G efficiency on local RTX5090 GPU0.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-uv run python}"
$PYTHON_BIN workspace/core4d/scripts/experiments/E187/test_reward_grid_efficiency.py
$PYTHON_BIN workspace/core4d/scripts/eval/runners/eval_E187_reward_grid_efficiency.py "$@"
