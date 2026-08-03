#!/usr/bin/env bash
# E187 A1 continuation exact/grid fidelity; never launches CEM or touches GPUs.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-uv run python}"
$PYTHON_BIN workspace/core4d/scripts/experiments/E187/test_surface_distance_reward.py
$PYTHON_BIN workspace/core4d/scripts/experiments/E187/test_distance_fidelity.py
$PYTHON_BIN workspace/core4d/scripts/experiments/E187/test_canonical_distance_query.py
$PYTHON_BIN workspace/core4d/scripts/eval/runners/eval_E187_reward_grid_fidelity.py "$@"
