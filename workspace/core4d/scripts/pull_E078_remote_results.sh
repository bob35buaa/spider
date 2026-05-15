#!/usr/bin/env bash
# Pull E078 results/logs from the remote machine back to this workspace.
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

mkdir -p workspace/core4d/results/E078 logs/E078

echo "[$(date '+%H:%M:%S')] pulling E078 results from ${REMOTE_HOST}:${REMOTE_REPO}"
scp "${SCP_OPTS[@]}" -r "${REMOTE_HOST}:${REMOTE_REPO}/workspace/core4d/results/E078/"* workspace/core4d/results/E078/
scp "${SCP_OPTS[@]}" -r "${REMOTE_HOST}:${REMOTE_REPO}/logs/E078/"* logs/E078/

echo "[$(date '+%H:%M:%S')] running local E078 eval"
.venv/bin/python workspace/core4d/scripts/eval/eval_E078.py E078A E078B | tee logs/E078/eval_E078_local_after_pull.log
echo "Done."
