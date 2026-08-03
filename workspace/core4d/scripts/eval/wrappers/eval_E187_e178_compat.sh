#!/usr/bin/env bash
# E187 Gate S0 static authority and default-off compatibility checks.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-uv run python}"
SCRIPT_ROOT="workspace/core4d/scripts/experiments/E187"

$PYTHON_BIN "$SCRIPT_ROOT/freeze_authority.py"
$PYTHON_BIN "$SCRIPT_ROOT/test_freeze_authority.py"
$PYTHON_BIN "$SCRIPT_ROOT/test_surface_distance_reward.py"
$PYTHON_BIN "$SCRIPT_ROOT/build_e178_compat_replay.py"
$PYTHON_BIN "$SCRIPT_ROOT/test_e178_compat_replay.py"
$PYTHON_BIN workspace/core4d/scripts/experiments/E186/test_production_grid_backend.py
$PYTHON_BIN workspace/core4d/scripts/experiments/E182/test_query_tape.py

echo "E187_E178_COMPAT_STATIC=PASS"
echo "E187_GATE_S0_REPLAY=PENDING"
