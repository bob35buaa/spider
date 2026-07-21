#!/usr/bin/env bash
# Pull E168 A6000 canary CEM artifacts/logs/manifests back to local workspace.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-canary}"
REMOTE="${REMOTE:-spider-remote}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/xiayb/pHRI_workspace/spider}"

if [ "$MODE" != "canary" ]; then
  echo "E168 A6000 pull currently supports canary only." >&2
  exit 2
fi

retry() {
  local attempt=1
  local max_attempts="${E168_REMOTE_RETRY_MAX:-4}"
  while true; do
    if "$@"; then
      return 0
    fi
    if [ "$attempt" -ge "$max_attempts" ]; then
      echo "Command failed after ${attempt} attempts: $*" >&2
      return 1
    fi
    echo "Retry ${attempt}/${max_attempts}: $*" >&2
    sleep "$((attempt * 3))"
    attempt=$((attempt + 1))
  done
}

pull_remote_dir() {
  local rel_path="$1"
  local local_path="$2"
  mkdir -p "$local_path"
  if retry ssh "$REMOTE" "test -d '${REMOTE_ROOT}/${rel_path}'"; then
    retry rsync -az "${REMOTE}:${REMOTE_ROOT}/${rel_path}/" "$local_path/"
  else
    echo "skip missing remote dir: ${REMOTE}:${REMOTE_ROOT}/${rel_path}"
  fi
}

pull_remote_dir "workspace/core4d/results/E168/s6_downstream/cem/canary" \
  "workspace/core4d/results/E168/s6_downstream/cem/canary"
pull_remote_dir "workspace/core4d/results/E168/s6_downstream/cem/manifests" \
  "workspace/core4d/results/E168/s6_downstream/cem/manifests"
pull_remote_dir "workspace/core4d/results/E168/s0_environment" \
  "workspace/core4d/results/E168/s0_environment"
pull_remote_dir "logs/E168/cem/canary" "logs/E168/cem/canary"

echo "Pulled E168 A6000 ${MODE} artifacts."
find workspace/core4d/results/E168/s6_downstream/cem/canary -maxdepth 1 -name 'E168_*.npz' | sort | wc -l | awk '{print "root_npz_count="$1}'
find workspace/core4d/results/E168/s6_downstream/cem/canary -maxdepth 1 -name 'E168_*_smoke.mp4' | sort | wc -l | awk '{print "video_count="$1}'
find workspace/core4d/results/E168/s6_downstream/cem/canary -maxdepth 2 -name 'trajectory_mjwp_act.npz' | sort | wc -l | awk '{print "outdir_npz_count="$1}'
find workspace/core4d/results/E168/s6_downstream/cem/canary -maxdepth 2 -name 'config_act.yaml' | sort | wc -l | awk '{print "config_act_count="$1}'
