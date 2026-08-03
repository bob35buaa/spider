#!/usr/bin/env bash
# Freeze E187 A2 production-integration evidence before A3 canary.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-uv run python}"
$PYTHON_BIN workspace/core4d/scripts/experiments/E187/test_surface_distance_reward.py
$PYTHON_BIN workspace/core4d/scripts/experiments/E187/test_production_overrides.py
$PYTHON_BIN workspace/core4d/scripts/experiments/E187/test_production_integration.py
$PYTHON_BIN workspace/core4d/scripts/experiments/E187/test_freeze_production_integration.py
$PYTHON_BIN workspace/core4d/scripts/experiments/E187/freeze_production_integration.py
