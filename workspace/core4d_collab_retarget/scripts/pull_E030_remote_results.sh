#!/usr/bin/env bash
# Pull E030 result artifacts from spider-remote.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

REMOTE_HOST="${REMOTE_HOST:-spider-remote}"
REMOTE_REPO="${REMOTE_REPO:-/home/xiayb/pHRI_workspace/spider_e030_worktree}"

mkdir -p workspace/core4d_collab_retarget/results/E030 logs/core4d_collab_retarget/E030
rsync -av \
  "$REMOTE_HOST:$REMOTE_REPO/workspace/core4d_collab_retarget/results/E030/" \
  workspace/core4d_collab_retarget/results/E030/
rsync -av \
  "$REMOTE_HOST:$REMOTE_REPO/logs/core4d_collab_retarget/E030/" \
  logs/core4d_collab_retarget/E030/
echo "Pulled E030 remote results."
