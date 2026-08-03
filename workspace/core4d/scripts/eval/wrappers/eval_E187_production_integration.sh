#!/usr/bin/env bash
# E187 A2 keep22 CPU/MJWarp compile plus 44-tape conservative-G audit.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-preflight}"
if [[ "$MODE" != "preflight" && "$MODE" != "run" ]]; then
  echo "usage: $0 [preflight|run]" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-uv run python}"
RESULT_ROOT="workspace/core4d/results/E187/s3_prg_audit"
SNAPSHOT_ROOT="$RESULT_ROOT/process_snapshots"
mkdir -p "$SNAPSHOT_ROOT"

capture_snapshot() {
  local label="$1"
  nvidia-smi \
    --query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw \
    --format=csv,noheader,nounits > "$SNAPSHOT_ROOT/local_gpu_${label}.csv"
  nvidia-smi \
    --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits > "$SNAPSHOT_ROOT/local_compute_${label}.csv"
}

capture_snapshot before
trap 'capture_snapshot after' EXIT

gpu_name="$(nvidia-smi --id=0 --query-gpu=name --format=csv,noheader | xargs)"
if [[ "$gpu_name" != "NVIDIA GeForce RTX 5090" ]]; then
  echo "E187 A2 requires local GPU0 RTX5090, got: $gpu_name" >&2
  exit 1
fi

$PYTHON_BIN workspace/core4d/scripts/experiments/E187/test_surface_distance_reward.py
$PYTHON_BIN workspace/core4d/scripts/experiments/E187/test_production_overrides.py
$PYTHON_BIN workspace/core4d/scripts/experiments/E187/test_production_integration.py
$PYTHON_BIN workspace/core4d/scripts/experiments/E187/audit_production_integration.py "$MODE"
$PYTHON_BIN workspace/core4d/scripts/experiments/E187/audit_reference_final_gate.py "$MODE"
