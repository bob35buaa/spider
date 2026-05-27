#!/usr/bin/env bash
# Pull E082 remote results/logs back to the local workspace and run merged eval.
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

mkdir -p workspace/core4d/results/E082 logs/E082

echo "[$(date '+%H:%M:%S')] pulling E082 results from ${REMOTE_HOST}:${REMOTE_REPO}"
rsync -av "${REMOTE_HOST}:${REMOTE_REPO}/workspace/core4d/results/E082/" workspace/core4d/results/E082/
rsync -av "${REMOTE_HOST}:${REMOTE_REPO}/logs/E082/" logs/E082/

echo "[$(date '+%H:%M:%S')] running local E082 eval"
RESULTS=workspace/core4d/results/E082 \
VARIANTS_FILE=workspace/core4d/scripts/E082/variants.tsv \
.venv/bin/python workspace/core4d/scripts/eval/eval_E082.py | tee logs/E082/eval_E082_local_after_pull.log
echo "Done."
