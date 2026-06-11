#!/usr/bin/env bash
# Launch E092 remote split on spider-remote GPU0/GPU1.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

JOB="${1:-spider-dyn-smoke}"
REMOTE_HOST="${REMOTE_HOST:-spider-remote}"
REMOTE_REPO="${REMOTE_REPO:-/home/xiayb/pHRI_workspace/spider}"
REMOTE_SESSION="${REMOTE_SESSION:-E092_${JOB//-/_}}"
SSH_OPTS=(
  -o BatchMode=yes
  -o ConnectTimeout=20
  -o ServerAliveInterval=10
  -o ServerAliveCountMax=3
)

case "$JOB" in
  spider-dyn-smoke)
    TRAIN_SCRIPT="workspace/core4d/scripts/train/train_E092_spider_dyn.sh"
    STAGE="smoke"
    ROUTE_DIR="spider_dyn"
    ;;
  spider-dyn-full)
    TRAIN_SCRIPT="workspace/core4d/scripts/train/train_E092_spider_dyn.sh"
    STAGE="full"
    ROUTE_DIR="spider_dyn"
    ;;
  rl-omni-smoke)
    TRAIN_SCRIPT="workspace/core4d/scripts/train/train_E092_rl_from_omni.sh"
    STAGE="smoke"
    ROUTE_DIR="rl_from_omni"
    ;;
  rl-omni-main)
    TRAIN_SCRIPT="workspace/core4d/scripts/train/train_E092_rl_from_omni.sh"
    STAGE="main"
    ROUTE_DIR="rl_from_omni"
    ;;
  rl-spider-smoke)
    TRAIN_SCRIPT="workspace/core4d/scripts/train/train_E092_rl_from_spider.sh"
    STAGE="smoke"
    ROUTE_DIR="rl_from_spider"
    ;;
  rl-spider-main)
    TRAIN_SCRIPT="workspace/core4d/scripts/train/train_E092_rl_from_spider.sh"
    STAGE="main"
    ROUTE_DIR="rl_from_spider"
    ;;
  *)
    echo "Unknown E092 remote job: $JOB" >&2
    echo "Use: spider-dyn-smoke|spider-dyn-full|rl-omni-smoke|rl-omni-main|rl-spider-smoke|rl-spider-main" >&2
    exit 2
    ;;
esac

DIRTY="$(git status --porcelain -- . \
  ':(exclude).codex/config.toml' \
  ':(exclude)workspace/exp_diagnostic/my_thoughts.md' \
  ':(exclude,glob)**/__pycache__/**')"
if [ -n "$DIRTY" ]; then
  echo "Working tree has uncommitted changes. Commit them before remote launch so git sync is exact." >&2
  echo "$DIRTY" >&2
  exit 1
fi

echo "[$(date '+%H:%M:%S')] pushing local branch to origin"
git push

echo "[$(date '+%H:%M:%S')] launching E092 ${JOB} on ${REMOTE_HOST}:${REMOTE_REPO}"
ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_REPO' && git pull --ff-only && mkdir -p logs/E092/${ROUTE_DIR}/${STAGE} workspace/core4d/results/E092/${ROUTE_DIR}/${STAGE} && (tmux kill-session -t '$REMOTE_SESSION' 2>/dev/null || true)"
ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_REPO' && tmux new-session -d -s '$REMOTE_SESSION' \"set -euo pipefail; python workspace/core4d/scripts/E092/build_three_case_tasks.py >/tmp/e092_build_${JOB}.log 2>&1 || true; (bash '$TRAIN_SCRIPT' remote-gpu0 '$STAGE' 0 2>&1 | tee logs/E092/${ROUTE_DIR}/${STAGE}/remote_gpu0.log) & PID0=\\\$!; (bash '$TRAIN_SCRIPT' remote-gpu1 '$STAGE' 1 2>&1 | tee logs/E092/${ROUTE_DIR}/${STAGE}/remote_gpu1.log) & PID1=\\\$!; echo E092 remote ${JOB} launched PID0=\\\$PID0 PID1=\\\$PID1; wait \\\$PID0; echo GPU0_DONE; wait \\\$PID1; echo GPU1_DONE; echo E092 remote ${JOB} complete\""

echo "Remote launched: session=${REMOTE_SESSION}"
echo "Monitor:"
echo "  ssh ${REMOTE_HOST} \"tmux capture-pane -t ${REMOTE_SESSION} -p | tail -60\""
echo "Pull results:"
echo "  bash workspace/core4d/scripts/pull_E092_remote_results.sh ${JOB}"
