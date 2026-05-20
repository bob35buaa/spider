#!/usr/bin/env bash
# E026: pull E081 full rerun remote results/logs and rebuild local comparison.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

REMOTE_HOST="${REMOTE_HOST:-spider-remote}"
REMOTE_REPO="${REMOTE_REPO:-/home/xiayb/pHRI_workspace/spider}"
RESULTS="${RESULTS:-workspace/core4d_collab_retarget/results/E026_E081_full}"
LOGS="${LOGS:-logs/core4d_collab_retarget/E026_E081_full}"
VARIANTS_FILE="${VARIANTS_FILE:-workspace/core4d_collab_retarget/scripts/E026/e081_full_variants.tsv}"
SCP_OPTS=(-o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=10 -o ServerAliveCountMax=3)

mkdir -p "$RESULTS" "$LOGS"

echo "[$(date '+%H:%M:%S')] pulling E026 E081 remote results"
scp "${SCP_OPTS[@]}" -r "${REMOTE_HOST}:${REMOTE_REPO}/${RESULTS}/"* "$RESULTS/" 2>/dev/null || true
scp "${SCP_OPTS[@]}" -r "${REMOTE_HOST}:${REMOTE_REPO}/${LOGS}/"* "$LOGS/" 2>/dev/null || true

echo "[$(date '+%H:%M:%S')] rebuilding local E026 E081 comparison"
VARIANTS_FILE="$VARIANTS_FILE" RESULTS="$RESULTS" \
  .venv/bin/python workspace/core4d/scripts/eval/eval_E081.py \
  | tee "$LOGS/eval_after_pull.log"
