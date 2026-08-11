#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
MODE="${MODE:-canary}"
SENTINEL_ONLY="${SENTINEL_ONLY:-0}"
"$PYTHON_BIN" workspace/core4d/scripts/experiments/E194/audit_g1_expansion.py --require-all
MODE="$MODE" SENTINEL_ONLY="$SENTINEL_ONLY" PYTHON_BIN="$PYTHON_BIN" \
  bash workspace/core4d/scripts/launch/active/run_E194_G1_expansion_local.sh
MODE="$MODE" SENTINEL_ONLY="$SENTINEL_ONLY" PYTHON_BIN="$PYTHON_BIN" \
  bash workspace/core4d/scripts/launch/active/run_E194_G1_expansion_remote_Ada6000.sh
echo "E194 G1 expansion hybrid launch complete MODE=$MODE SENTINEL_ONLY=$SENTINEL_ONLY"
