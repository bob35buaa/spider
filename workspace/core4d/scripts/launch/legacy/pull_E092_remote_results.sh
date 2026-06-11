#!/usr/bin/env bash
# Pull E092 remote results and logs back to the local workspace.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

JOB="${1:-spider-dyn-smoke}"
REMOTE_HOST="${REMOTE_HOST:-spider-remote}"
REMOTE_REPO="${REMOTE_REPO:-/home/xiayb/pHRI_workspace/spider}"
SSH_OPTS=(
  -o BatchMode=yes
  -o ConnectTimeout=20
  -o ServerAliveInterval=10
  -o ServerAliveCountMax=3
)

case "$JOB" in
  spider-dyn-smoke)
    STAGE="smoke"; ROUTE_DIR="spider_dyn";;
  spider-dyn-full)
    STAGE="full"; ROUTE_DIR="spider_dyn";;
  rl-omni-smoke)
    STAGE="smoke"; ROUTE_DIR="rl_from_omni";;
  rl-omni-main)
    STAGE="main"; ROUTE_DIR="rl_from_omni";;
  rl-spider-smoke)
    STAGE="smoke"; ROUTE_DIR="rl_from_spider";;
  rl-spider-main)
    STAGE="main"; ROUTE_DIR="rl_from_spider";;
  *)
    echo "Unknown E092 remote job: $JOB" >&2
    exit 2
    ;;
esac

RESULTS_DIR="workspace/core4d/results/E092/${ROUTE_DIR}/${STAGE}"
LOGS_DIR="logs/E092/${ROUTE_DIR}/${STAGE}"
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

echo "[$(date '+%H:%M:%S')] remote result files:"
ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_REPO' && find '$RESULTS_DIR' -maxdepth 3 -type f | sort | sed -n '1,120p'"
echo "[$(date '+%H:%M:%S')] pulling ${RESULTS_DIR}"
scp -r "${REMOTE_HOST}:${REMOTE_REPO}/${RESULTS_DIR}/"* "$RESULTS_DIR/" 2>/dev/null || true
echo "[$(date '+%H:%M:%S')] pulling ${LOGS_DIR}"
scp -r "${REMOTE_HOST}:${REMOTE_REPO}/${LOGS_DIR}/"* "$LOGS_DIR/" 2>/dev/null || true

echo "Pulled E092 ${JOB}:"
find "$RESULTS_DIR" -maxdepth 2 -type f | sort | sed -n '1,120p'
