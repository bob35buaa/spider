#!/usr/bin/env bash
# E187 Evaluation paired video: E178 baseline left, E187 continuation right.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-uv run python}"
TEST=workspace/core4d/scripts/experiments/E187/test_paired_evaluation_videos.py
RENDERER=workspace/core4d/scripts/experiments/E187/render_paired_evaluation_videos.py

mode="${1:-run}"
shift || true
$PYTHON_BIN "$TEST"

case "$mode" in
  preflight|run|audit)
    $PYTHON_BIN "$RENDERER" "$mode" "$@"
    ;;
  *)
    echo "usage: $0 {preflight,run,audit} [--overwrite]" >&2
    exit 2
    ;;
esac
