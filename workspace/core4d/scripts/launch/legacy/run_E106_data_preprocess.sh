#!/usr/bin/env bash
# E106 Phase 0: build missing Holosoma/OmniRetarget/SPIDER inputs.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

LOG_DIR="${LOG_DIR:-/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2/logs}"
mkdir -p "$LOG_DIR"

export CASE_FILE_REL="${CASE_FILE_REL:-../holosoma/workspace/v3/data_construction_v2/inputs/cases_e106_box026_30candidate_pipeline.tsv}"
export RESULT_ROOT_REL="${RESULT_ROOT_REL:-../holosoma/workspace/v3/data_construction_v2/results/stage2b_medium/results}"
export REPLACE_WRIST_WITH_FINGERTIP="${REPLACE_WRIST_WITH_FINGERTIP:-0}"

echo "[$(date '+%H:%M:%S')] === E106 data preprocess start ==="
echo "CASE_FILE_REL=$CASE_FILE_REL"
echo "RESULT_ROOT_REL=$RESULT_ROOT_REL"
echo "REPLACE_WRIST_WITH_FINGERTIP=$REPLACE_WRIST_WITH_FINGERTIP"

bash workspace/core4d/scripts/E091/run_stage2b_medium_boxes.sh "$@"

echo "[$(date '+%H:%M:%S')] === E106 data preprocess complete ==="
