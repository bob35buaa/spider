#!/usr/bin/env bash
# E187 A2 generate and audit the 22 opt-in production overrides.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-uv run python}"
$PYTHON_BIN workspace/core4d/scripts/experiments/E187/test_surface_distance_reward.py
$PYTHON_BIN workspace/core4d/scripts/experiments/E187/test_production_overrides.py
$PYTHON_BIN workspace/core4d/scripts/experiments/E187/build_production_overrides.py
