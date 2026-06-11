#!/usr/bin/env bash
# Pull E115 remote results/logs from spider-remote.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-full}"
REMOTE="${REMOTE:-spider-remote}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/xiayb/pHRI_workspace/spider}"

mkdir -p "workspace/core4d/results/E115/cem/${STAGE}" "logs/E115"
rsync -av "${REMOTE}:${REMOTE_ROOT}/workspace/core4d/results/E115/cem/${STAGE}/" \
  "workspace/core4d/results/E115/cem/${STAGE}/"
rsync -av "${REMOTE}:${REMOTE_ROOT}/logs/E115/" "logs/E115/"
echo "Pulled E115 ${STAGE} results from ${REMOTE}:${REMOTE_ROOT}"
