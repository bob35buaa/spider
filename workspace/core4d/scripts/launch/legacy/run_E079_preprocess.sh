#!/usr/bin/env bash
# E079: run generic CORE4D preprocessing for existing p1 masks and new p2 cases.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-all}"
RESULT_ROOT="${RESULT_ROOT:-workspace/core4d/results/E079}"
LOGS="${LOGS:-logs/E079}"
mkdir -p "$RESULT_ROOT" "$LOGS"

run_p1() {
  echo "[$(date '+%H:%M:%S')] === E079 p1 trim/contact audit ==="
  RESULT_ROOT="$RESULT_ROOT" bash workspace/core4d/data_preprocess/pipeline.sh \
    --case-file workspace/core4d/data_preprocess/cases_E079_existing_p1.tsv \
    --skip-retarget \
    --skip-spider \
    "$@" \
    > "$LOGS/preprocess_p1.log" 2>&1
}

run_p2() {
  echo "[$(date '+%H:%M:%S')] === E079 existing p2 trim/contact audit ==="
  RESULT_ROOT="$RESULT_ROOT" bash workspace/core4d/data_preprocess/pipeline.sh \
    --case-file workspace/core4d/data_preprocess/cases_E079_existing_p2.tsv \
    --skip-retarget \
    --skip-spider \
    "$@" \
    > "$LOGS/preprocess_existing_p2.log" 2>&1

  echo "[$(date '+%H:%M:%S')] === E079 p2 build ==="
  RESULT_ROOT="$RESULT_ROOT" bash workspace/core4d/data_preprocess/pipeline.sh \
    --case-file workspace/core4d/data_preprocess/cases_E079_build_p2.tsv \
    "$@" \
    > "$LOGS/preprocess_p2.log" 2>&1
}

case "$MODE" in
  p1)
    shift || true
    run_p1 "$@"
    ;;
  p2)
    shift || true
    run_p2 "$@"
    ;;
  dry-run)
    RESULT_ROOT="$RESULT_ROOT" bash workspace/core4d/data_preprocess/pipeline.sh \
      --case-file workspace/core4d/data_preprocess/cases_E079_existing_p1.tsv \
      --skip-retarget \
      --skip-spider \
      --dry-run
    RESULT_ROOT="$RESULT_ROOT" bash workspace/core4d/data_preprocess/pipeline.sh \
      --case-file workspace/core4d/data_preprocess/cases_E079_existing_p2.tsv \
      --skip-retarget \
      --skip-spider \
      --dry-run
    RESULT_ROOT="$RESULT_ROOT" bash workspace/core4d/data_preprocess/pipeline.sh \
      --case-file workspace/core4d/data_preprocess/cases_E079_build_p2.tsv \
      --dry-run
    ;;
  all)
    shift || true
    run_p1 "$@"
    run_p2 "$@"
    ;;
  *)
    echo "Usage: $0 [dry-run|p1|p2|all] [pipeline args...]"
    exit 2
    ;;
esac

echo "[$(date '+%H:%M:%S')] === E079 preprocess ${MODE} done ==="
