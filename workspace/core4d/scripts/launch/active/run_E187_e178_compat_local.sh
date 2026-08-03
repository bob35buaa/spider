#!/usr/bin/env bash
# E187 Gate S0 E178 legacy replay on local RTX 5090; never touches E178 outputs.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

CASE_ID="${1:?usage: $0 <representative-case-id>}"
GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
RESULT_ROOT="workspace/core4d/results/E187/s0_environment/e178_compat"
SNAPSHOT_ROOT="$RESULT_ROOT/process_snapshots/$CASE_ID"
mkdir -p "$SNAPSHOT_ROOT"

bash workspace/core4d/scripts/convert/snapshot_scenes.sh E187 \
  dcv3_omnirt_v1_ref_fk_bucket003_20231018_003_p1 \
  dcv3_omnirt_v1_ref_fk_bucket004_20231002_021_p1 \
  dcv3_omnirt_v2_ref_fk_bucket007_20231020_055_p1

nvidia-smi \
  --query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw \
  --format=csv,noheader,nounits > "$SNAPSHOT_ROOT/local_gpu_before.csv"
nvidia-smi \
  --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
  --format=csv,noheader,nounits > "$SNAPSHOT_ROOT/local_compute_before.csv"

set +e
PYTHONPATH=. MUJOCO_GL="${MUJOCO_GL:-egl}" "$PYTHON_BIN" \
  workspace/core4d/scripts/experiments/E187/run_e178_compat_replay.py \
  "$CASE_ID" --python-bin "$PYTHON_BIN" --gpu-id "$GPU_ID"
status=$?
set -e

nvidia-smi \
  --query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw \
  --format=csv,noheader,nounits > "$SNAPSHOT_ROOT/local_gpu_after.csv"
nvidia-smi \
  --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
  --format=csv,noheader,nounits > "$SNAPSHOT_ROOT/local_compute_after.csv"

exit "$status"
