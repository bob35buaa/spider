#!/usr/bin/env bash
# Pull E167 remote CEM results/logs back to local workspace.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-full}"
REMOTE="${REMOTE:-spider-remote}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/xiayb/pHRI_workspace/spider}"
RESULTS="workspace/core4d/results/E167/holosoma_zonly/cem/${STAGE}"
LOGS="logs/E167/cem/${STAGE}"

mkdir -p "$RESULTS" "$LOGS"

pull_remote_dir() {
  local rel_path="$1"
  local local_path="$2"
  if ssh "$REMOTE" "test -d '${REMOTE_ROOT}/${rel_path}'"; then
    rsync -az "${REMOTE}:${REMOTE_ROOT}/${rel_path}/" "$local_path/"
  else
    echo "skip missing remote dir: ${REMOTE}:${REMOTE_ROOT}/${rel_path}"
  fi
}

pull_remote_dir "$RESULTS" "$RESULTS"
pull_remote_dir "$LOGS" "$LOGS"

echo "Pulled E167 ${STAGE} remote results to ${RESULTS}"
find "$RESULTS" -maxdepth 1 -name 'E167_*.npz' | sort | wc -l | awk '{print "root_npz_count="$1}'
find "$RESULTS" -maxdepth 1 -name 'E167_*_'"${STAGE}"'.mp4' | sort | wc -l | awk '{print "video_count="$1}'
