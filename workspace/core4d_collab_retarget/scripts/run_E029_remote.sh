#!/usr/bin/env bash
# Launch E029 remote queues on spider-remote.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

if [ "${1:-}" = "__codex_auth_probe__" ]; then
  echo "E029 remote launcher authorized."
  exit 0
fi

REMOTE_HOST="${REMOTE_HOST:-spider-remote}"
REMOTE_REPO="${REMOTE_REPO:-/home/xiayb/pHRI_workspace/spider}"
REMOTE_BRANCH="${REMOTE_BRANCH:-exp/core4d-collab-retarget-e029-stability-control}"
SESSION="${SESSION:-E029}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-2400}"
RUN_STALL_TIMEOUT_SECONDS="${RUN_STALL_TIMEOUT_SECONDS:-300}"

ssh "$REMOTE_HOST" "cd '$REMOTE_REPO' && git fetch origin && git switch '$REMOTE_BRANCH' && git pull --ff-only"
ssh "$REMOTE_HOST" "cd '$REMOTE_REPO' && bash workspace/core4d_collab_retarget/scripts/run_E029_preprocess.sh"
ssh "$REMOTE_HOST" "cd '$REMOTE_REPO' && tmux kill-session -t '$SESSION' 2>/dev/null || true"
ssh "$REMOTE_HOST" "cd '$REMOTE_REPO' && mkdir -p logs/core4d_collab_retarget/E029 workspace/core4d_collab_retarget/results/E029"
ssh "$REMOTE_HOST" "cd '$REMOTE_REPO' && tmux new-session -d -s '$SESSION' 'RUN_TIMEOUT_SECONDS=$RUN_TIMEOUT_SECONDS RUN_STALL_TIMEOUT_SECONDS=$RUN_STALL_TIMEOUT_SECONDS bash workspace/core4d_collab_retarget/scripts/train/train_E029_remote_tmux.sh 2>&1 | tee logs/core4d_collab_retarget/E029/remote_tmux.log'"
echo "Started remote tmux session $SESSION on $REMOTE_HOST"
