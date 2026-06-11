#!/usr/bin/env bash
# Pull E124 remote results/logs back to local workspace.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-smoke}"
REMOTE="${REMOTE:-spider-remote}"
REMOTE_REPO="${REMOTE_REPO:-/home/xiayb/pHRI_workspace/spider}"

mkdir -p "workspace/core4d/results/E124/cem/${STAGE}" "logs/E124"

rsync -av "${REMOTE}:${REMOTE_REPO}/workspace/core4d/results/E124/cem/${STAGE}/" \
  "workspace/core4d/results/E124/cem/${STAGE}/"
rsync -av "${REMOTE}:${REMOTE_REPO}/logs/E124/" "logs/E124/"

echo "Pulled E124 ${STAGE} results from ${REMOTE}:${REMOTE_REPO}"
