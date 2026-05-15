#!/usr/bin/env bash
# Launch the E081 remote split on spider-remote GPU1 only.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

REMOTE_HOST="${REMOTE_HOST:-spider-remote}"
REMOTE_REPO="${REMOTE_REPO:-/home/xiayb/pHRI_workspace/spider}"
REMOTE_SESSION="${REMOTE_SESSION:-E081}"
REMOTE_GPU="${REMOTE_GPU:-1}"
SSH_OPTS=(
  -o BatchMode=yes
  -o ConnectTimeout=20
  -o ServerAliveInterval=10
  -o ServerAliveCountMax=3
)

DIRTY="$(git status --porcelain -- . ':(exclude).codex/config.toml' ':(exclude,glob)**/__pycache__/**')"
if [ -n "$DIRTY" ]; then
  echo "Working tree has uncommitted changes. Commit them before remote launch so git sync is exact." >&2
  echo "$DIRTY" >&2
  exit 1
fi

echo "[$(date '+%H:%M:%S')] pushing local branch to origin"
git push

echo "[$(date '+%H:%M:%S')] launching E081 remote split on ${REMOTE_HOST}:${REMOTE_REPO}, GPU=${REMOTE_GPU}"
ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_REPO' && git pull --ff-only && mkdir -p logs/E081 workspace/core4d/results/E081 && (tmux kill-session -t '$REMOTE_SESSION' 2>/dev/null || true)"
ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_REPO' && tmux new-session -d -s '$REMOTE_SESSION' \"bash workspace/core4d/scripts/train/train_E081.sh remote '$REMOTE_GPU' 2>&1 | tee logs/E081/remote_gpu${REMOTE_GPU}.log\""

echo "Remote launched on GPU${REMOTE_GPU} only."
echo "Monitor:"
echo "  ssh ${REMOTE_HOST} \"tmux capture-pane -t ${REMOTE_SESSION} -p | tail -40\""
echo "Pull results when done:"
echo "  REMOTE_HOST=${REMOTE_HOST} REMOTE_REPO=${REMOTE_REPO} bash workspace/core4d/scripts/pull_E081_remote_results.sh"
