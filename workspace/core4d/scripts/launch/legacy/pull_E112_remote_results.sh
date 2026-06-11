#!/usr/bin/env bash
# Pull E112 remote CEM outputs back to local. Results remain out of git.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

REMOTE="${REMOTE:-spider-remote}"
REMOTE_REPO="${REMOTE_REPO:-/home/xiayb/pHRI_workspace/spider}"
STAGE="${1:-full}"

mkdir -p "workspace/core4d/results/E112/cem/${STAGE}" "logs/E112"

rsync -av \
  "${REMOTE}:${REMOTE_REPO}/workspace/core4d/results/E112/cem/${STAGE}/" \
  "workspace/core4d/results/E112/cem/${STAGE}/"

rsync -av \
  "${REMOTE}:${REMOTE_REPO}/logs/E112/" \
  "logs/E112/"

echo "Pulled E112 ${STAGE} results from ${REMOTE}:${REMOTE_REPO}"
