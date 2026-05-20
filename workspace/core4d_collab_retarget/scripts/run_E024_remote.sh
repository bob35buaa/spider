#!/usr/bin/env bash
# Launch E024 remote queues on spider-remote.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

if [ "${1:-}" = "__codex_auth_probe__" ]; then
  echo "E024 remote launcher authorized."
  exit 0
fi

REMOTE_HOST="${REMOTE_HOST:-spider-remote}"
REMOTE_REPO="${REMOTE_REPO:-/home/xiayb/pHRI_workspace/spider}"
SESSION="${SESSION:-E024}"

ssh "$REMOTE_HOST" "cd '$REMOTE_REPO' && git fetch origin && git checkout exp/core4d-collab-retarget && git pull --ff-only"
ssh "$REMOTE_HOST" "cd '$REMOTE_REPO' && bash workspace/core4d_collab_retarget/scripts/run_E024_preprocess.sh"
ssh "$REMOTE_HOST" "cd '$REMOTE_REPO' && tmux kill-session -t '$SESSION' 2>/dev/null || true"
ssh "$REMOTE_HOST" "cd '$REMOTE_REPO' && mkdir -p logs/core4d_collab_retarget/E024 workspace/core4d_collab_retarget/results/E024"
ssh "$REMOTE_HOST" "cd '$REMOTE_REPO' && tmux new-session -d -s '$SESSION' 'bash workspace/core4d_collab_retarget/scripts/train/train_E024_remote_tmux.sh 2>&1 | tee logs/core4d_collab_retarget/E024/remote_tmux.log'"
echo "Started remote tmux session $SESSION on $REMOTE_HOST"
