#!/usr/bin/env bash
# Pull E083 remote results/logs back to the local workspace and run merged eval.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

REMOTE_HOST="${REMOTE_HOST:-spider-remote}"
REMOTE_REPO="${REMOTE_REPO:-/home/xiayb/pHRI_workspace/spider}"

mkdir -p workspace/core4d/results/E083 logs/E083

echo "[$(date '+%H:%M:%S')] pulling E083 results from ${REMOTE_HOST}:${REMOTE_REPO}"
rsync -av "${REMOTE_HOST}:${REMOTE_REPO}/workspace/core4d/results/E083/" workspace/core4d/results/E083/
rsync -av "${REMOTE_HOST}:${REMOTE_REPO}/logs/E083/" logs/E083/

echo "[$(date '+%H:%M:%S')] running local E083 eval"
RESULTS=workspace/core4d/results/E083 \
VARIANTS_FILE=workspace/core4d/scripts/E083/variants.tsv \
.venv/bin/python workspace/core4d/scripts/eval/eval_E083.py | tee logs/E083/eval_E083_local_after_pull.log
echo "Done."
