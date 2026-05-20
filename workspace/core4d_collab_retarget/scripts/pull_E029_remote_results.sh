#!/usr/bin/env bash
# Pull E029 result artifacts from spider-remote.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

REMOTE_HOST="${REMOTE_HOST:-spider-remote}"
REMOTE_REPO="${REMOTE_REPO:-/home/xiayb/pHRI_workspace/spider}"

mkdir -p workspace/core4d_collab_retarget/results/E029 logs/core4d_collab_retarget/E029
rsync -av \
  "$REMOTE_HOST:$REMOTE_REPO/workspace/core4d_collab_retarget/results/E029/" \
  workspace/core4d_collab_retarget/results/E029/
rsync -av \
  "$REMOTE_HOST:$REMOTE_REPO/logs/core4d_collab_retarget/E029/" \
  logs/core4d_collab_retarget/E029/
echo "Pulled E029 remote results."
