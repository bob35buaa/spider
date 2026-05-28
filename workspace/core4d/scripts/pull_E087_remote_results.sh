#!/usr/bin/env bash
# Pull E087 remote results/logs back to local workspace and run merged eval.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

REMOTE_HOST="${REMOTE_HOST:-spider-remote}"
REMOTE_REPO="${REMOTE_REPO:-/home/xiayb/pHRI_workspace/spider}"

mkdir -p workspace/core4d/results/E087 logs/E087

echo "[$(date '+%H:%M:%S')] pulling E087 results from ${REMOTE_HOST}:${REMOTE_REPO}"
rsync -av "${REMOTE_HOST}:${REMOTE_REPO}/workspace/core4d/results/E087/" workspace/core4d/results/E087/
rsync -av "${REMOTE_HOST}:${REMOTE_REPO}/logs/E087/" logs/E087/

echo "[$(date '+%H:%M:%S')] running local E087 merged eval"
RESULTS=workspace/core4d/results/E087 \
VARIANTS_FILE=workspace/core4d/scripts/E087/variants.tsv \
.venv/bin/python workspace/core4d/scripts/eval/eval_E087.py | tee logs/E087/eval_E087_local_after_pull.log
echo "Done."

