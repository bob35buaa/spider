#!/usr/bin/env bash
# Pull E147 remote CEM results/logs back to the local workspace.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-full}"
REMOTE="${REMOTE:-spider-remote}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/xiayb/pHRI_workspace/spider}"
RESULTS="workspace/core4d/results/E147/rubber_hand_collision/cem/${STAGE}"
LOGS="logs/E147/cem/${STAGE}"

mkdir -p "$RESULTS" "$LOGS"

rsync -az "${REMOTE}:${REMOTE_ROOT}/${RESULTS}/" "$RESULTS/"
rsync -az "${REMOTE}:${REMOTE_ROOT}/${LOGS}/" "$LOGS/"

echo "Pulled E147 ${STAGE} results to ${RESULTS}"
find "$RESULTS" -maxdepth 1 -name 'E147_*.npz' | sort | wc -l | awk '{print "root_npz_count="$1}'
find "$RESULTS" -maxdepth 1 -name 'E147_*_'"${STAGE}"'.mp4' | sort | wc -l | awk '{print "video_count="$1}'
