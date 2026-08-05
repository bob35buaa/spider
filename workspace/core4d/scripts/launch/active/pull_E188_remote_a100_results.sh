#!/usr/bin/env bash
# Pull E188 remote canary or remaining Full rows via staging and verify all SHA.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-canary}"
if [[ "$STAGE" != canary && "$STAGE" != full ]]; then
  echo "usage: $0 {canary|full}" >&2
  exit 2
fi
PYTHON_BIN="${E188_PYTHON_BIN:-.venv/bin/python}"
RUNNER=workspace/core4d/scripts/experiments/E188/run_full_queue.py
DEPLOYMENT="workspace/core4d/results/E188/s4_canary/deployment/remote_deployment_manifest.json"
RESULT_ROOT=workspace/core4d/results/E188/s4_canary
CASES=(bucket007_20231003_2_021_p1 bucket007_20231020_055_p1)
if [[ "$STAGE" == full ]]; then
  DEPLOYMENT="workspace/core4d/results/E188/s5_full/deployment/remote_deployment_manifest.json"
  RESULT_ROOT=workspace/core4d/results/E188/s5_full
  QUEUE_ROOT=workspace/core4d/results/E188/s5_full/queue_speed_rebalanced_v2
  mapfile -t CASES < <(for worker in a100-4 a100-5; do awk -F '\t' 'NR>1 && $1=="False" {print $2}' "$QUEUE_ROOT/$worker.tsv"; done)
  if [[ "${#CASES[@]}" != 6 ]]; then
    echo "E188 v2 remote queue must contain exactly 6 remaining rows" >&2
    exit 2
  fi
fi
HOST="${E188_A100_HOST:-61.172.170.106}"
PORT="${E188_A100_PORT:-30409}"
IDENTITY="${E188_A100_IDENTITY:-/home/ubuntu/.ssh/id_rsa_tianyiyun}"
REMOTE_ROOT="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["remote_root"])' "$DEPLOYMENT")"
for case_id in "${CASES[@]}"; do
  if [[ -e "$RESULT_ROOT/rows/$case_id" ]]; then
    echo "refusing to overwrite local E188 row: $case_id" >&2
    exit 2
  fi
done
STAGING="$(mktemp -d "$RESULT_ROOT/.pull_staging.XXXXXX")"
STAGING_ROWS="$STAGING/$RESULT_ROOT/rows"
mkdir -p "$STAGING_ROWS"
for case_id in "${CASES[@]}"; do
  rsync --archive -e "ssh -p $PORT -i $IDENTITY -o BatchMode=yes" "batchcom@$HOST:$REMOTE_ROOT/$RESULT_ROOT/rows/$case_id/" "$STAGING_ROWS/$case_id/"
done
verify_args=()
for case_id in "${CASES[@]}"; do verify_args+=(--case-id "$case_id"); done
PYTHONPATH=workspace/core4d/scripts/experiments/E188 "$PYTHON_BIN" "$RUNNER" verify-rows --root "$STAGING" --stage "$STAGE" "${verify_args[@]}"
mkdir -p "$RESULT_ROOT/rows"
for case_id in "${CASES[@]}"; do mv "$STAGING_ROWS/$case_id" "$RESULT_ROOT/rows/$case_id"; done
PYTHONPATH=workspace/core4d/scripts/experiments/E188 "$PYTHON_BIN" "$RUNNER" verify-rows --root . --stage "$STAGE" "${verify_args[@]}"
echo "E188_PULL_A100=PASS stage=$STAGE rows=${#CASES[@]} staging=$STAGING"
