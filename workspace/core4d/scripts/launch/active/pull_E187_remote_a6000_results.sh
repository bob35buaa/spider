#!/usr/bin/env bash
# Pull E187 Ada canary artifacts through isolated staging; never overwrite local rows.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-canary}"
if [[ "$MODE" != "canary" ]]; then
  echo "usage: $0 canary" >&2
  exit 2
fi

REMOTE_HOST="${ADA_HOST:-spider-remote}"
PYTHON_BIN="${E187_PYTHON_BIN:-.venv/bin/python}"
SCRIPT="workspace/core4d/scripts/experiments/E187/deploy_remote_canary.py"
DEPLOYMENT="workspace/core4d/results/E187/s4_canary/deployment/remote_deployment_manifest.json"
RESULT_ROOT="workspace/core4d/results/E187/s4_canary"
ROWS_ROOT="$RESULT_ROOT/rows"
SNAPSHOT_ROOT="$RESULT_ROOT/process_snapshots/remote"
CASES=(bucket004_20231002_021_p1 bucket007_20231020_055_p1)

if [[ ! -f "$DEPLOYMENT" ]]; then
  echo "E187 remote deployment manifest is missing: $DEPLOYMENT" >&2
  exit 2
fi
REMOTE_ROOT="$($PYTHON_BIN -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["remote_root"])' "$DEPLOYMENT")"
if [[ "$REMOTE_ROOT" != /home/xiayb/pHRI_workspace/e187_runs/e187_*/spider ]]; then
  echo "Refusing unsafe E187 remote root: $REMOTE_ROOT" >&2
  exit 2
fi
for case_id in "${CASES[@]}"; do
  if [[ -e "$ROWS_ROOT/$case_id" ]]; then
    echo "Refusing to overwrite existing local E187 row: $ROWS_ROOT/$case_id" >&2
    exit 2
  fi
done

mkdir -p "$RESULT_ROOT" "$ROWS_ROOT" "$SNAPSHOT_ROOT"
STAGING="$(mktemp -d "$RESULT_ROOT/.pull_staging.XXXXXX")"
STAGING_ROWS="$STAGING/workspace/core4d/results/E187/s4_canary/rows"
mkdir -p "$STAGING_ROWS"
for case_id in "${CASES[@]}"; do
  rsync --archive \
    "$REMOTE_HOST:$REMOTE_ROOT/$ROWS_ROOT/$case_id/" \
    "$STAGING_ROWS/$case_id/"
done

PYTHONPATH=. "$PYTHON_BIN" "$SCRIPT" verify-rows --root "$STAGING" \
  --case-id bucket004_20231002_021_p1 \
  --case-id bucket007_20231020_055_p1
for case_id in "${CASES[@]}"; do
  mv "$STAGING_ROWS/$case_id" "$ROWS_ROOT/$case_id"
done
PYTHONPATH=. "$PYTHON_BIN" "$SCRIPT" verify-rows --root . \
  --case-id bucket004_20231002_021_p1 \
  --case-id bucket007_20231020_055_p1

ssh "$REMOTE_HOST" \
  'nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.total,memory.free,utilization.gpu --format=csv,noheader,nounits' \
  > "$SNAPSHOT_ROOT/gpu_after.csv"
ssh "$REMOTE_HOST" \
  'nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits' \
  > "$SNAPSHOT_ROOT/compute_after.csv" || true
printf '%s\n' "$STAGING" > "$RESULT_ROOT/last_pull_staging.txt"
echo "E187_PULL_REMOTE_CANARY=PASS"
