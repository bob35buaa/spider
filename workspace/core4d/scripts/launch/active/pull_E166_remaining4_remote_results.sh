#!/usr/bin/env bash
# Pull E166 remaining4 remote CEM/SUGAR results/logs back to local workspace.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-full}"
REMOTE="${REMOTE:-spider-remote}"
REMOTE_SPIDER_ROOT="${REMOTE_SPIDER_ROOT:-/home/xiayb/pHRI_workspace/spider}"
REMOTE_SUGAR_ROOT="${REMOTE_SUGAR_ROOT:-/home/xiayb/pHRI_workspace/Loco-Manipulation-projects/SUGAR-private}"
LOCAL_SUGAR_ROOT="${LOCAL_SUGAR_ROOT:-/home/ubuntu/Workspace/Loco-Manipulation/SUGAR}"
RESULTS="workspace/core4d/results/E166/foot_smooth_retarget/cem/${STAGE}"
POST="workspace/core4d/results/E166/foot_smooth_retarget/postprocess/${STAGE}"
LOGS="logs/E166/remaining4"
SUGAR_OUT="outputs/core4d/e166_remaining4_A_B2_refiner_rl"

mkdir -p "$RESULTS" "$POST" "$LOGS" "${LOCAL_SUGAR_ROOT}/${SUGAR_OUT}"

pull_remote_dir() {
  local remote_root="$1"
  local rel_path="$2"
  local local_path="$3"
  if ssh "$REMOTE" "test -d '${remote_root}/${rel_path}'"; then
    rsync -az "${REMOTE}:${remote_root}/${rel_path}/" "$local_path/"
  else
    echo "skip missing remote dir: ${REMOTE}:${remote_root}/${rel_path}"
  fi
}

pull_remote_dir "$REMOTE_SPIDER_ROOT" "$RESULTS" "$RESULTS"
pull_remote_dir "$REMOTE_SPIDER_ROOT" "$POST" "$POST"
pull_remote_dir "$REMOTE_SPIDER_ROOT" "$LOGS" "$LOGS"
pull_remote_dir "$REMOTE_SUGAR_ROOT" "$SUGAR_OUT" "${LOCAL_SUGAR_ROOT}/${SUGAR_OUT}"

echo "Pulled E166 remaining4 results"
find "$RESULTS" -maxdepth 1 -name 'E166_*.npz' | sort | wc -l | awk '{print "cem_root_npz_count="$1}'
find "$POST" -maxdepth 1 -name 'E166_*A_B2_postSmooth.npz' | sort | wc -l | awk '{print "postprocess_npz_count="$1}'
