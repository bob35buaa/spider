#!/usr/bin/env bash
# E187 A1 bucket004 formal query tape on local RTX5090 GPU0.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-preflight}"
if [[ "$MODE" != "preflight" && "$MODE" != "run" ]]; then
  echo "usage: $0 [preflight|run]" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
RESULT_ROOT="workspace/core4d/results/E187/s2_canonical_grid_sdf/bucket004_formal_tape"
SNAPSHOT_ROOT="$RESULT_ROOT/process_snapshots"
mkdir -p "$SNAPSHOT_ROOT"

bash workspace/core4d/scripts/convert/snapshot_scenes.sh \
  E187/s2_canonical_grid_sdf/bucket004_formal_tape \
  dcv3_omnirt_v1_ref_fk_bucket004_20231002_021_p1

nvidia-smi \
  --query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw \
  --format=csv,noheader,nounits > "$SNAPSHOT_ROOT/local_gpu_before.csv"
nvidia-smi \
  --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
  --format=csv,noheader,nounits > "$SNAPSHOT_ROOT/local_compute_before.csv"

gpu_name="$(nvidia-smi --id=0 --query-gpu=name --format=csv,noheader | xargs)"
if [[ "$gpu_name" != "NVIDIA GeForce RTX 5090" ]]; then
  echo "E187 bucket004 requires local GPU0 RTX5090, got: $gpu_name" >&2
  exit 1
fi

runner_args=(
  workspace/core4d/scripts/experiments/E187/run_bucket004_formal_tape.py
  --gpu-id 0
  --python-bin "$PYTHON_BIN"
)
if [[ "$MODE" == "preflight" ]]; then
  runner_args+=(--preflight)
fi

set +e
PYTHONPATH=. MUJOCO_GL="${MUJOCO_GL:-egl}" "$PYTHON_BIN" \
  workspace/core4d/scripts/experiments/E187/test_bucket004_formal_tape.py
test_status=$?
if [[ $test_status -eq 0 ]]; then
  PYTHONPATH=. MUJOCO_GL="${MUJOCO_GL:-egl}" "$PYTHON_BIN" "${runner_args[@]}"
  status=$?
else
  status=$test_status
fi
set -e

nvidia-smi \
  --query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw \
  --format=csv,noheader,nounits > "$SNAPSHOT_ROOT/local_gpu_after.csv"
nvidia-smi \
  --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
  --format=csv,noheader,nounits > "$SNAPSHOT_ROOT/local_compute_after.csv"

exit "$status"
