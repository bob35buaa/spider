#!/usr/bin/env bash
# Pull E152 remote CEM results/logs back to local workspace.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-full}"
REMOTE="${REMOTE:-spider-remote}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/xiayb/pHRI_workspace/spider}"
RESULTS="workspace/core4d/results/E152/axis1_hand_object_physics_gate/cem/${STAGE}"
LOGS="logs/E152/cem/${STAGE}"

mkdir -p "$RESULTS" "$LOGS"

rsync -az --ignore-missing-args "${REMOTE}:${REMOTE_ROOT}/${RESULTS}/" "$RESULTS/"
rsync -az --ignore-missing-args "${REMOTE}:${REMOTE_ROOT}/${LOGS}/" "$LOGS/"

echo "Pulled E152 ${STAGE} results to ${RESULTS}"
find "$RESULTS" -maxdepth 1 -name 'E152_*.npz' | sort | wc -l | awk '{print "root_npz_count="$1}'
find "$RESULTS" -maxdepth 1 -name 'E152_*_'"${STAGE}"'.mp4' | sort | wc -l | awk '{print "video_count="$1}'
