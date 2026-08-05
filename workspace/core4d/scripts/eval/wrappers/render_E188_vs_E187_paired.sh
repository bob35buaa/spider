#!/usr/bin/env bash
# Complete E188 deferred videos, then render E187-left/E188-right paired review videos.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
BUILDER=workspace/core4d/scripts/experiments/E188/build_evaluation_manifest.py
OFFLINE=workspace/core4d/scripts/experiments/E188/render_offline_videos.py
PAIRED=workspace/core4d/scripts/experiments/E188/render_paired_videos.py
TEST=workspace/core4d/scripts/experiments/E188/test_paired_videos.py
MODE="${1:-run}"
shift || true

case "$MODE" in
  preflight)
    "$PYTHON_BIN" "$BUILDER"
    "$PYTHON_BIN" "$OFFLINE" preflight
    ;;
  run)
    "$PYTHON_BIN" "$OFFLINE" run "$@"
    "$PYTHON_BIN" "$BUILDER"
    "$PYTHON_BIN" "$TEST"
    "$PYTHON_BIN" "$PAIRED" preflight
    "$PYTHON_BIN" "$PAIRED" run
    ;;
  audit)
    "$PYTHON_BIN" "$OFFLINE" audit
    "$PYTHON_BIN" "$TEST"
    "$PYTHON_BIN" "$PAIRED" audit
    ;;
  *)
    echo "usage: $0 {preflight|run|audit} [--overwrite]" >&2
    exit 2
    ;;
esac
