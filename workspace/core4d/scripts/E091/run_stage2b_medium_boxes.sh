#!/usr/bin/env bash
# Run SPIDER CORE4D preprocessing for E091 medium-box rows after templates exist.
set -euo pipefail

HOLOSOMA_REPO="${HOLOSOMA_REPO:-/home/ubuntu/Workspace/holosoma}"
SPIDER_REPO="${SPIDER_REPO:-/home/ubuntu/Workspace/spider}"
V2_ROOT="${V2_ROOT:-$HOLOSOMA_REPO/workspace/v3/data_construction_v2}"
CASE_FILE_REL="${CASE_FILE_REL:-../holosoma/workspace/v3/data_construction_v2/inputs/cases_stage2b_medium_pipeline.tsv}"
RESULT_ROOT_REL="${RESULT_ROOT_REL:-../holosoma/workspace/v3/data_construction_v2/results/stage2b_medium/results}"
LOG_DIR="${LOG_DIR:-$V2_ROOT/logs}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
HSRETARGETING_BIN="${HSRETARGETING_BIN:-/home/ubuntu/.holosoma_deps/miniconda3/envs/hsretargeting/bin}"
REPLACE_WRIST_WITH_FINGERTIP="${REPLACE_WRIST_WITH_FINGERTIP:-0}"

mkdir -p "$LOG_DIR" "$V2_ROOT/results/stage2b_medium"

case_file_abs="$SPIDER_REPO/$CASE_FILE_REL"
if [ ! -f "$case_file_abs" ]; then
  echo "Missing case file: $case_file_abs" >&2
  exit 1
fi

enabled_count="$(awk -F '\t' 'NR > 1 && $1 == "1" {n++} END {print n+0}' "$case_file_abs")"
timestamp="$(date +%Y%m%d_%H%M%S)"
log_path="$LOG_DIR/stage2b_medium_${timestamp}.log"

echo "HOLOSOMA_REPO=$HOLOSOMA_REPO"
echo "SPIDER_REPO=$SPIDER_REPO"
echo "V2_ROOT=$V2_ROOT"
echo "CASE_FILE_REL=$CASE_FILE_REL"
echo "RESULT_ROOT_REL=$RESULT_ROOT_REL"
echo "REPLACE_WRIST_WITH_FINGERTIP=$REPLACE_WRIST_WITH_FINGERTIP"
echo "ENABLED_ROWS=$enabled_count"
echo "LOG=$log_path"

if [ "$enabled_count" -eq 0 ]; then
  echo "No enabled Stage2b rows. Create source scene templates, update cases_stage2b_medium_pipeline.tsv, then rerun."
  exit 0
fi

(
  cd "$SPIDER_REPO"
  export PATH="$HSRETARGETING_BIN:$PATH"
  REPO="$SPIDER_REPO" \
  HOLOSOMA_DIR="$HOLOSOMA_REPO" \
  RESULT_ROOT="$RESULT_ROOT_REL" \
  PYTHON_BIN="$PYTHON_BIN" \
  REPLACE_WRIST_WITH_FINGERTIP="$REPLACE_WRIST_WITH_FINGERTIP" \
  bash "$SPIDER_REPO/workspace/core4d/data_preprocess/pipeline.sh" \
    --case-file "$CASE_FILE_REL" \
    "$@"
) 2>&1 | tee "$log_path"
