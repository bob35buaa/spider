#!/usr/bin/env bash
# Pull E088 remote results/logs back to local workspace and run merged eval.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

REMOTE_HOST="${REMOTE_HOST:-spider-remote}"
REMOTE_REPO="${REMOTE_REPO:-/home/xiayb/pHRI_workspace/spider}"

mkdir -p workspace/core4d/results/E088 logs/E088

echo "[$(date '+%H:%M:%S')] pulling E088 results from ${REMOTE_HOST}:${REMOTE_REPO}"
rsync -av "${REMOTE_HOST}:${REMOTE_REPO}/workspace/core4d/results/E088/" workspace/core4d/results/E088/
rsync -av "${REMOTE_HOST}:${REMOTE_REPO}/logs/E088/" logs/E088/

echo "[$(date '+%H:%M:%S')] running local E088 merged eval"
RESULTS=workspace/core4d/results/E088 \
VARIANTS_FILE=workspace/core4d/scripts/E088/variants.tsv \
.venv/bin/python workspace/core4d/scripts/eval/eval_E088.py | tee logs/E088/eval_E088_local_after_pull.log
echo "Done."
