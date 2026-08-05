#!/usr/bin/env bash
# Canonical E188 vs frozen E187 15-case paired evaluation.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
BUILDER=workspace/core4d/scripts/experiments/E188/build_evaluation_manifest.py
RUNNER=workspace/core4d/scripts/eval/runners/eval_E188_vs_E187_full.py
MODE="${1:-run}"

case "$MODE" in
  preflight)
    "$PYTHON_BIN" "$BUILDER"
    "$PYTHON_BIN" "$RUNNER" preflight
    ;;
  run)
    "$PYTHON_BIN" "$BUILDER"
    "$PYTHON_BIN" "$RUNNER" preflight
    "$PYTHON_BIN" "$RUNNER" run
    ;;
  *)
    echo "usage: $0 {preflight|run}" >&2
    exit 2
    ;;
esac
