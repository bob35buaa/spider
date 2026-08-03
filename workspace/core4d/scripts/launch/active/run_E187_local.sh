#!/usr/bin/env bash
# E187 local RTX5090 canary/full worker entrypoint.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-preflight}"
if [[ "$MODE" == "full" ]]; then
  exec bash workspace/core4d/scripts/launch/active/run_E187_full_local.sh full
fi
if [[ "$MODE" != "preflight" && "$MODE" != "canary" ]]; then
  echo "usage: $0 {preflight|canary|full}" >&2
  exit 2
fi

PYTHON_BIN="${E187_PYTHON_BIN:-.venv/bin/python}"
CASE_ID="bucket003_20231018_003_p1"
RESULT_ROOT="workspace/core4d/results/E187/s4_canary"
SNAPSHOT_ROOT="$RESULT_ROOT/process_snapshots/local"
mkdir -p "$SNAPSHOT_ROOT"

bash workspace/core4d/scripts/convert/snapshot_scenes.sh \
  E187/s4_canary/local \
  dcv3_omnirt_v1_ref_fk_bucket003_20231018_003_p1

capture_snapshot() {
  local label="$1"
  nvidia-smi \
    --query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw \
    --format=csv,noheader,nounits > "$SNAPSHOT_ROOT/gpu_${label}.csv"
  nvidia-smi \
    --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits > "$SNAPSHOT_ROOT/compute_${label}.csv"
}

capture_snapshot before
trap 'capture_snapshot after' EXIT

gpu_name="$(nvidia-smi --id=0 --query-gpu=name --format=csv,noheader | xargs)"
if [[ "$gpu_name" != "NVIDIA GeForce RTX 5090" ]]; then
  echo "E187 local worker requires GPU0 RTX5090, got: $gpu_name" >&2
  exit 1
fi

PYTHONPATH=. MUJOCO_GL="${MUJOCO_GL:-egl}" "$PYTHON_BIN" \
  workspace/core4d/scripts/experiments/E187/test_production_canary.py

runner_mode="preflight"
if [[ "$MODE" == "canary" ]]; then
  runner_mode="run"
fi
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. MUJOCO_GL="${MUJOCO_GL:-egl}" "$PYTHON_BIN" \
  workspace/core4d/scripts/experiments/E187/run_production_canary.py \
  "$runner_mode" --case-id "$CASE_ID" --gpu-id 0 --python-bin "$PYTHON_BIN"
