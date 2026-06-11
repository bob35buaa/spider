#!/usr/bin/env bash
# Pull E153 remote CEM sweep results/logs back to local workspace.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-full}"
REMOTE="${REMOTE:-spider-remote}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/xiayb/pHRI_workspace/spider}"
RESULTS="workspace/core4d/results/E153/gate_threshold_sweep/cem/${STAGE}"
LOGS="logs/E153/cem/${STAGE}"

mkdir -p "$RESULTS" "$LOGS"

rsync -az --ignore-missing-args "${REMOTE}:${REMOTE_ROOT}/${RESULTS}/" "$RESULTS/"
rsync -az --ignore-missing-args "${REMOTE}:${REMOTE_ROOT}/${LOGS}/" "$LOGS/"

echo "Pulled E153 ${STAGE} results to ${RESULTS}"
find "$RESULTS" -maxdepth 1 -name 'E153_*.npz' | sort | wc -l | awk '{print "root_npz_count="$1}'
find "$RESULTS" -maxdepth 1 -name 'E153_*_'"${STAGE}"'.mp4' | sort | wc -l | awk '{print "video_count="$1}'
