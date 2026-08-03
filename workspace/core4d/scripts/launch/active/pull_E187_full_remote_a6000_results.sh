#!/usr/bin/env bash
# Pull completed E187 Full Ada rows through staging and verify every SHA twice.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

REMOTE_HOST="${ADA_HOST:-spider-remote}"
PYTHON_BIN="${E187_PYTHON_BIN:-.venv/bin/python}"
RUNNER="workspace/core4d/scripts/experiments/E187/run_full_queue.py"
DEPLOYMENT="workspace/core4d/results/E187/s5_full/deployment/remote_deployment_manifest.json"
RESULT_ROOT="workspace/core4d/results/E187/s5_full"
ROWS_ROOT="$RESULT_ROOT/rows"
SNAPSHOT_ROOT="$RESULT_ROOT/process_snapshots/remote"
CASES=(
  bucket007_20231003_2_021_p1
  bucket004_20231003_1_012_p1
  bucket007_20231023_075_p2
  bucket007_20231003_1_021_p2
  bucket003_20231020_064_p1
  bucket007_20231003_1_021_p1
  bucket007_20231003_2_023_p1
  bucket003_20231018_001_p2
  bucket007_20231023_073_p1
  bucket003_20231018_005_p2
  bucket007_20231023_075_p1
  bucket007_20231020_059_p1
  bucket007_20231018_019_p1
)

if [[ ! -f "$DEPLOYMENT" ]]; then
  echo "E187 Full remote deployment manifest is missing: $DEPLOYMENT" >&2
  exit 2
fi
REMOTE_ROOT="$($PYTHON_BIN -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["remote_root"])' "$DEPLOYMENT")"
if [[ "$REMOTE_ROOT" != /home/xiayb/pHRI_workspace/e187_runs/e187_full_*/spider ]]; then
  echo "Refusing unsafe E187 Full remote root: $REMOTE_ROOT" >&2
  exit 2
fi
for case_id in "${CASES[@]}"; do
  if [[ -e "$ROWS_ROOT/$case_id" ]]; then
    echo "Refusing to overwrite existing local E187 Full row: $ROWS_ROOT/$case_id" >&2
    exit 2
  fi
done

mkdir -p "$RESULT_ROOT" "$ROWS_ROOT" "$SNAPSHOT_ROOT"
STAGING="$(mktemp -d "$RESULT_ROOT/.pull_staging.XXXXXX")"
STAGING_ROWS="$STAGING/workspace/core4d/results/E187/s5_full/rows"
mkdir -p "$STAGING_ROWS"
for case_id in "${CASES[@]}"; do
  rsync --archive \
    "$REMOTE_HOST:$REMOTE_ROOT/$ROWS_ROOT/$case_id/" \
    "$STAGING_ROWS/$case_id/"
done

verify_args=()
for case_id in "${CASES[@]}"; do
  verify_args+=(--case-id "$case_id")
done
PYTHONPATH=. "$PYTHON_BIN" "$RUNNER" verify-rows --root "$STAGING" "${verify_args[@]}"
for case_id in "${CASES[@]}"; do
  mv "$STAGING_ROWS/$case_id" "$ROWS_ROOT/$case_id"
done
PYTHONPATH=. "$PYTHON_BIN" "$RUNNER" verify-rows --root . "${verify_args[@]}"

ssh "$REMOTE_HOST" \
  'nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.total,memory.free,utilization.gpu --format=csv,noheader,nounits' \
  > "$SNAPSHOT_ROOT/gpu_after.csv"
ssh "$REMOTE_HOST" \
  'nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits' \
  > "$SNAPSHOT_ROOT/compute_after.csv" || true
printf '%s\n' "$STAGING" > "$RESULT_ROOT/last_full_pull_staging.txt"
echo "E187_PULL_REMOTE_FULL=PASS rows=${#CASES[@]}"
