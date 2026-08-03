#!/usr/bin/env bash
# Freeze the user-authorized E187 Gate S0 progression waiver without hiding FAIL.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-uv run python}"
SCRIPT_ROOT="workspace/core4d/scripts/experiments/E187"

$PYTHON_BIN "$SCRIPT_ROOT/freeze_authority.py"
$PYTHON_BIN "$SCRIPT_ROOT/test_freeze_authority.py"
$PYTHON_BIN "$SCRIPT_ROOT/freeze_gate0_user_waiver.py"
$PYTHON_BIN "$SCRIPT_ROOT/test_gate0_user_waiver.py"

echo "E187_GATE0_TECHNICAL_STATUS=FAIL"
echo "E187_GATE0_PROGRESSION_AUTHORITY=USER_WAIVED"
echo "E187_A1_REWARD_GRID=PENDING"
