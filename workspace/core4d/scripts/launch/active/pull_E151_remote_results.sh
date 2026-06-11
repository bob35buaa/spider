#!/usr/bin/env bash
# Pull E151 remote CEM results/logs back to local workspace.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-full}"
REMOTE="${REMOTE:-spider-remote}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/xiayb/pHRI_workspace/spider}"
RESULTS="workspace/core4d/results/E151/route_b_hand_surface_contact/cem/${STAGE}"
LOGS="logs/E151/cem/${STAGE}"

mkdir -p "$RESULTS" "$LOGS"

rsync -az --ignore-missing-args "${REMOTE}:${REMOTE_ROOT}/${RESULTS}/" "$RESULTS/"
rsync -az --ignore-missing-args "${REMOTE}:${REMOTE_ROOT}/${LOGS}/" "$LOGS/"

echo "Pulled E151 ${STAGE} results to ${RESULTS}"
find "$RESULTS" -maxdepth 1 -name 'E151_*.npz' | sort | wc -l | awk '{print "root_npz_count="$1}'
find "$RESULTS" -maxdepth 1 -name 'E151_*_'"${STAGE}"'.mp4' | sort | wc -l | awk '{print "video_count="$1}'
