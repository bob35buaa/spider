#!/usr/bin/env bash
# Pull E084 remote results/logs back to local workspace and run merged eval.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

REMOTE_HOST="${REMOTE_HOST:-spider-remote}"
REMOTE_REPO="${REMOTE_REPO:-/home/xiayb/pHRI_workspace/spider}"

mkdir -p workspace/core4d/results/E084 logs/E084

echo "[$(date '+%H:%M:%S')] pulling E084 results from ${REMOTE_HOST}:${REMOTE_REPO}"
rsync -av "${REMOTE_HOST}:${REMOTE_REPO}/workspace/core4d/results/E084/" workspace/core4d/results/E084/
rsync -av "${REMOTE_HOST}:${REMOTE_REPO}/logs/E084/" logs/E084/

echo "[$(date '+%H:%M:%S')] running local E084 merged eval"
RESULTS=workspace/core4d/results/E084 \
VARIANTS_FILE=workspace/core4d/scripts/E084/variants.tsv \
.venv/bin/python workspace/core4d/scripts/eval/eval_E084.py | tee logs/E084/eval_E084_local_after_pull.log
echo "Done."
