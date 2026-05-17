#!/usr/bin/env bash
# Pull E003 remote results/logs back to the local workspace and run local eval.
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

mkdir -p workspace/core4d_collab_retarget/results/E003 logs/core4d_collab_retarget/E003

echo "[$(date '+%H:%M:%S')] pulling E003 results from ${REMOTE_HOST}:${REMOTE_REPO}"
scp "${SCP_OPTS[@]}" -r "${REMOTE_HOST}:${REMOTE_REPO}/workspace/core4d_collab_retarget/results/E003/"* workspace/core4d_collab_retarget/results/E003/
scp "${SCP_OPTS[@]}" -r "${REMOTE_HOST}:${REMOTE_REPO}/logs/core4d_collab_retarget/E003/"* logs/core4d_collab_retarget/E003/

echo "[$(date '+%H:%M:%S')] running local E003 eval"
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E003.py | tee logs/core4d_collab_retarget/E003/eval_E003_local_after_pull.log
echo "Done."
