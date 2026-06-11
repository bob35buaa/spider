#!/usr/bin/env bash
# Pull E081 remote results/logs back to the local workspace.
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

mkdir -p workspace/core4d/results/E081 logs/E081

echo "[$(date '+%H:%M:%S')] pulling E081 results from ${REMOTE_HOST}:${REMOTE_REPO}"
scp "${SCP_OPTS[@]}" -r "${REMOTE_HOST}:${REMOTE_REPO}/workspace/core4d/results/E081/"* workspace/core4d/results/E081/
scp "${SCP_OPTS[@]}" -r "${REMOTE_HOST}:${REMOTE_REPO}/logs/E081/"* logs/E081/

echo "[$(date '+%H:%M:%S')] running local E081 eval"
.venv/bin/python workspace/core4d/scripts/eval/eval_E081.py | tee logs/E081/eval_E081_local_after_pull.log
echo "Done."
