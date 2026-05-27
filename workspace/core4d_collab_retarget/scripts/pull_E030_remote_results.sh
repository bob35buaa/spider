#!/usr/bin/env bash
# Pull E030 remote npz/mp4/log outputs back to the local workspace.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

REMOTE="${REMOTE:-spider-remote}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/xiayb/pHRI_workspace/spider}"
RESULTS_REL="workspace/core4d_collab_retarget/results/E030/d6_locked_cem"
LOGS_REL="logs/core4d_collab_retarget/E030"

mkdir -p "$RESULTS_REL" "$LOGS_REL"
rsync -av --ignore-missing-args \
  "$REMOTE:$REMOTE_ROOT/$RESULTS_REL/" "$RESULTS_REL/"
rsync -av --ignore-missing-args \
  "$REMOTE:$REMOTE_ROOT/$LOGS_REL/" "$LOGS_REL/"

echo "Pulled E030 remote results into $RESULTS_REL and $LOGS_REL"
