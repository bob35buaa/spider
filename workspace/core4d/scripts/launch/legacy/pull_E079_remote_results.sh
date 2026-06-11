#!/usr/bin/env bash
# Pull E079 remote results/logs back to the local workspace.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

REMOTE_HOST="${REMOTE_HOST:-spider-remote}"
REMOTE_REPO="${REMOTE_REPO:-/home/xiayb/pHRI_workspace/spider}"
SCP_OPTS=(
  -o BatchMode=yes
  -o ConnectTimeout=20
  -o ServerAliveInterval=10
  -o ServerAliveCountMax=3
)

mkdir -p workspace/core4d/results/E079 logs/E079

echo "[$(date '+%H:%M:%S')] pulling E079 results from ${REMOTE_HOST}:${REMOTE_REPO}"
scp "${SCP_OPTS[@]}" -r "${REMOTE_HOST}:${REMOTE_REPO}/workspace/core4d/results/E079/"* workspace/core4d/results/E079/
scp "${SCP_OPTS[@]}" -r "${REMOTE_HOST}:${REMOTE_REPO}/logs/E079/"* logs/E079/

echo "[$(date '+%H:%M:%S')] running local E079 eval"
.venv/bin/python workspace/core4d/scripts/eval/eval_E079.py | tee logs/E079/eval_E079_local_after_pull.log
echo "Done."
