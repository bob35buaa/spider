#!/usr/bin/env bash
# E187 A1 immutable continuation reward/object-grid lock.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-uv run python}"
$PYTHON_BIN workspace/core4d/scripts/experiments/E187/test_surface_distance_reward.py
$PYTHON_BIN workspace/core4d/scripts/experiments/E187/test_distance_fidelity.py
$PYTHON_BIN workspace/core4d/scripts/experiments/E187/test_reward_grid_lock.py
$PYTHON_BIN workspace/core4d/scripts/experiments/E187/freeze_reward_grid_lock.py
