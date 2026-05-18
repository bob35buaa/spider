#!/usr/bin/env bash
# Launch E012 dual-point partner pose closure sweep on spider-remote GPUs.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

if [ "${1:-}" = "__codex_auth_probe__" ]; then
  echo "E012 remote launcher authorized."
  exit 0
fi

REMOTE_HOST="${REMOTE_HOST:-spider-remote}"
REMOTE_REPO="${REMOTE_REPO:-/home/xiayb/pHRI_workspace/spider}"
REMOTE_SESSION="${REMOTE_SESSION:-E012}"
SSH_OPTS=(
  -o BatchMode=yes
  -o ConnectTimeout=20
  -o ServerAliveInterval=10
  -o ServerAliveCountMax=3
)

DIRTY="$(git status --porcelain -- . ':(exclude).codex/config.toml' ':(exclude)workspace/exp_task.md' ':(exclude)workspace/core4d_collab_retarget/progress.md' ':(exclude).claude/skills/experiment-brainstorm-zh/**' ':(exclude)workspace/core4d_collab_retarget/ideas/**' ':(exclude)workspace/core4d_collab_retarget/plan/04_E004_freejoint_virtual_partner_support_plan_v2.md' ':(exclude,glob)**/__pycache__/**')"
if [ -n "$DIRTY" ]; then
  echo "Working tree has uncommitted experiment changes. Commit them before remote launch so git sync is exact." >&2
  echo "$DIRTY" >&2
  exit 1
fi

echo "[$(date '+%H:%M:%S')] pushing local branch to origin"
git push

echo "[$(date '+%H:%M:%S')] syncing remote branch on ${REMOTE_HOST}:${REMOTE_REPO}"
ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_REPO' && git fetch origin && git switch exp/core4d-collab-retarget && git pull --ff-only"

echo "[$(date '+%H:%M:%S')] preparing E012 overrides on remote"
ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_REPO' && bash workspace/core4d_collab_retarget/scripts/run_E012_preprocess.sh"

echo "[$(date '+%H:%M:%S')] launching remote tmux session ${REMOTE_SESSION}"
ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_REPO' && mkdir -p logs/core4d_collab_retarget/E012 workspace/core4d_collab_retarget/results/E012 && (tmux kill-session -t '$REMOTE_SESSION' 2>/dev/null || true)"
ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_REPO' && tmux new-session -d -s '$REMOTE_SESSION' 'bash workspace/core4d_collab_retarget/scripts/train/train_E012_remote_tmux.sh'"

echo "Remote launched."
echo "Monitor:"
echo "  ssh ${REMOTE_HOST} \"tmux capture-pane -t ${REMOTE_SESSION} -p | tail -40\""
echo "Pull results when done:"
echo "  REMOTE_HOST=${REMOTE_HOST} REMOTE_REPO=${REMOTE_REPO} bash workspace/core4d_collab_retarget/scripts/pull_E012_remote_results.sh"
