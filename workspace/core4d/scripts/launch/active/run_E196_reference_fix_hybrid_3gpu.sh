#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
MODE="${MODE:-wave0}"
LOCAL_GPU_ID="${LOCAL_GPU_ID:-0}"
ALLOW_SUGAR_OVERLAP="${ALLOW_SUGAR_OVERLAP:-0}"
LOCAL="workspace/core4d/scripts/launch/active/run_E196_reference_fix_local.sh"
REMOTE="workspace/core4d/scripts/launch/active/run_E196_reference_fix_remote_Ada6000.sh"
AUDITOR="workspace/core4d/scripts/experiments/E196/audit_reference_fix.py"

"$PYTHON_BIN" "$AUDITOR" --scope prelaunch --require-all
MODE="$MODE" LOCAL_GPU_ID="$LOCAL_GPU_ID" CHECK_ONLY=1 PYTHON_BIN="$PYTHON_BIN" bash "$LOCAL"
MODE="$MODE" PREPARE_ONLY=1 ALLOW_SUGAR_OVERLAP="$ALLOW_SUGAR_OVERLAP" PYTHON_BIN="$PYTHON_BIN" bash "$REMOTE"
MODE="$MODE" USE_PREPARED=1 ALLOW_SUGAR_OVERLAP="$ALLOW_SUGAR_OVERLAP" PYTHON_BIN="$PYTHON_BIN" bash "$REMOTE"
MODE="$MODE" LOCAL_GPU_ID="$LOCAL_GPU_ID" PYTHON_BIN="$PYTHON_BIN" bash "$LOCAL"
echo "E196 hybrid launch complete: MODE=$MODE local=$LOCAL_GPU_ID Ada=0,1"
