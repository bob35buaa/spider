#!/usr/bin/env bash
# Pull E085 remote results/logs back to local workspace and run merged eval.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

REMOTE_HOST="${REMOTE_HOST:-spider-remote}"
REMOTE_REPO="${REMOTE_REPO:-/home/xiayb/pHRI_workspace/spider}"

mkdir -p workspace/core4d/results/E085 logs/E085

echo "[$(date '+%H:%M:%S')] pulling E085 results from ${REMOTE_HOST}:${REMOTE_REPO}"
rsync -av "${REMOTE_HOST}:${REMOTE_REPO}/workspace/core4d/results/E085/" workspace/core4d/results/E085/
rsync -av "${REMOTE_HOST}:${REMOTE_REPO}/logs/E085/" logs/E085/

echo "[$(date '+%H:%M:%S')] running local E085 merged eval"
RESULTS=workspace/core4d/results/E085 \
VARIANTS_FILE=workspace/core4d/scripts/E085/variants.tsv \
.venv/bin/python workspace/core4d/scripts/eval/eval_E085.py | tee logs/E085/eval_E085_local_after_pull.log
echo "Done."
