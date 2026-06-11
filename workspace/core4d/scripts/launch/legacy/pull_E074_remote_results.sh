#!/usr/bin/env bash
# Pull E074 results/logs from the remote machine back to this workspace.
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

mkdir -p workspace/core4d/results/E074 logs/E074

echo "[$(date '+%H:%M:%S')] pulling E074 results from ${REMOTE_HOST}:${REMOTE_REPO}"
scp "${SCP_OPTS[@]}" -r "${REMOTE_HOST}:${REMOTE_REPO}/workspace/core4d/results/E074/"* workspace/core4d/results/E074/
scp "${SCP_OPTS[@]}" -r "${REMOTE_HOST}:${REMOTE_REPO}/logs/E074/"* logs/E074/

echo "[$(date '+%H:%M:%S')] running local E074 eval"
.venv/bin/python workspace/core4d/scripts/eval/eval_E074.py E074A E074C | tee logs/E074/eval_E074_local_after_pull.log
echo "Done."
