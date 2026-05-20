#!/usr/bin/env bash
# Pull E028 result artifacts from spider-remote.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

REMOTE_HOST="${REMOTE_HOST:-spider-remote}"
REMOTE_REPO="${REMOTE_REPO:-/home/xiayb/pHRI_workspace/spider}"

mkdir -p workspace/core4d_collab_retarget/results/E028 logs/core4d_collab_retarget/E028
rsync -av \
  "$REMOTE_HOST:$REMOTE_REPO/workspace/core4d_collab_retarget/results/E028/" \
  workspace/core4d_collab_retarget/results/E028/
rsync -av \
  "$REMOTE_HOST:$REMOTE_REPO/logs/core4d_collab_retarget/E028/" \
  logs/core4d_collab_retarget/E028/
echo "Pulled E028 remote results."
