#!/usr/bin/env bash
# E182 S2 CPU/Open3D dev3-only task-query screen. Never touches GPU processes.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
RUNNER="$REPO_ROOT/workspace/core4d/scripts/experiments/E182/run_task_query_screen.py"
MODE="${1:-all}"

cd "$REPO_ROOT"
case "$MODE" in
  freeze)
    uv run python "$RUNNER" freeze-protocol
    ;;
  bucket003)
    uv run python "$RUNNER" run-case --case-id bucket003_20231018_001_p1
    ;;
  bucket004)
    uv run python "$RUNNER" run-case --case-id bucket004_20231002_021_p1
    ;;
  bucket007)
    uv run python "$RUNNER" run-case --case-id bucket007_20231020_055_p1
    ;;
  screen)
    uv run python "$RUNNER" run-all
    ;;
  aggregate)
    uv run python "$RUNNER" aggregate
    ;;
  all)
    uv run python "$RUNNER" freeze-protocol
    uv run python "$RUNNER" run-all
    uv run python "$RUNNER" aggregate
    ;;
  *)
    echo "Usage: $0 {freeze|bucket003|bucket004|bucket007|screen|aggregate|all}" >&2
    exit 2
    ;;
esac
