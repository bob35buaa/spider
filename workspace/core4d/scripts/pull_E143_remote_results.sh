#!/usr/bin/env bash
# Pull E143 remote results and logs back to local workspace.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-full}"
REMOTE="${REMOTE:-spider-remote}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/xiayb/pHRI_workspace/spider}"

mkdir -p "workspace/core4d/results/E143/cem/${STAGE}" "logs/E143/cem/${STAGE}"

rsync -az --ignore-missing-args \
  "${REMOTE}:${REMOTE_ROOT}/workspace/core4d/results/E143/cem/${STAGE}/" \
  "workspace/core4d/results/E143/cem/${STAGE}/"
rsync -az --ignore-missing-args \
  "${REMOTE}:${REMOTE_ROOT}/logs/E143/cem/${STAGE}/" \
  "logs/E143/cem/${STAGE}/"

echo "Pulled E143 ${STAGE} results and logs from ${REMOTE}"
