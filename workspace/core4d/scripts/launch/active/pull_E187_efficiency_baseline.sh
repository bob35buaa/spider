#!/usr/bin/env bash
# Pull and verify two remote E178 timing baselines without overwriting local data.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

REMOTE_HOST="${ADA_HOST:-spider-remote}"
PYTHON_BIN="${E187_PYTHON_BIN:-.venv/bin/python}"
AUDIT="workspace/core4d/scripts/experiments/E187/audit_canary_efficiency.py"
DEPLOYMENT="workspace/core4d/results/E187/s4_canary/deployment/remote_deployment_manifest.json"
RESULT_ROOT="workspace/core4d/results/E187/s4_canary/efficiency_baseline"
SNAPSHOT_ROOT="workspace/core4d/results/E187/s4_canary/process_snapshots/efficiency/remote"
CASES=(bucket004_20231002_021_p1 bucket007_20231020_055_p1)

REMOTE_ROOT="$($PYTHON_BIN -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["remote_root"])' "$DEPLOYMENT")"
if [[ "$REMOTE_ROOT" != /home/xiayb/pHRI_workspace/e187_runs/e187_*/spider ]]; then
  echo "Refusing unsafe E187 remote root: $REMOTE_ROOT" >&2
  exit 2
fi
for case_id in "${CASES[@]}"; do
  if [[ -e "$RESULT_ROOT/rows/$case_id" ]]; then
    echo "Refusing to overwrite E187 efficiency row: $case_id" >&2
    exit 2
  fi
done

STAGING="$(mktemp -d "workspace/core4d/results/E187/s4_canary/.efficiency_pull.XXXXXX")"
STAGING_ROWS="$STAGING/workspace/core4d/results/E187/s4_canary/efficiency_baseline/rows"
mkdir -p "$STAGING_ROWS" "$RESULT_ROOT/rows" "$SNAPSHOT_ROOT"
for case_id in "${CASES[@]}"; do
  rsync --archive \
    "$REMOTE_HOST:$REMOTE_ROOT/$RESULT_ROOT/rows/$case_id/" \
    "$STAGING_ROWS/$case_id/"
  PYTHONPATH=. "$PYTHON_BIN" "$AUDIT" verify-row --root "$STAGING" --case-id "$case_id"
done
for case_id in "${CASES[@]}"; do
  mv "$STAGING_ROWS/$case_id" "$RESULT_ROOT/rows/$case_id"
done
ssh "$REMOTE_HOST" \
  'nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.total,memory.free,utilization.gpu --format=csv,noheader,nounits' \
  > "$SNAPSHOT_ROOT/gpu_after.csv"
ssh "$REMOTE_HOST" \
  'nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits' \
  > "$SNAPSHOT_ROOT/compute_after.csv" || true
PYTHONPATH=. "$PYTHON_BIN" "$AUDIT" audit --root .
printf '%s\n' "$STAGING" > "$RESULT_ROOT/last_pull_staging.txt"
echo "E187_EFFICIENCY_GATE=PASS"
