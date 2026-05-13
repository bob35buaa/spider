#!/usr/bin/env bash
# E059: box023 baseline (E041c) vs warm (E055 snap warmstart) — Path D from E058 log.
#
# Default: parallel — baseline on GPU 0, warm on GPU 1.
# Override: `bash train_E059.sh serial 0` for both serially on GPU 0.
#
# Output (under workspace/core4d/results/E059/):
#   E059_baseline.{npz,mp4,log}
#   E059_warm.{npz,mp4,log}
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-parallel}"
GPU_BASELINE="${2:-0}"
GPU_WARM="${3:-1}"
TASK=box023_person1
RESULTS=workspace/core4d/results/E059
LOGS=logs/E059
WARMSTART=workspace/core4d/results/E055/box023_person1/warmstart_qpos.npz

mkdir -p "$RESULTS" "$LOGS"

run_one() {
  local name=$1 gpu=$2; shift 2
  local out_dir="$RESULTS/${name}_outdir"
  mkdir -p "$out_dir"
  echo "[$(date '+%H:%M:%S')] === $name (GPU $gpu, output_dir=$out_dir) ==="
  CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl PYTHONUNBUFFERED=1 .venv/bin/python -u examples/run_mjwp.py \
    +override=core4d_e041c \
    task=$TASK \
    +use_torch_compile=false \
    output_dir="$out_dir" \
    video_output_path="$RESULTS/${name}.mp4" \
    "$@" \
    > "$LOGS/${name}.log" 2>&1
  cp "$out_dir/trajectory_mjwp_act.npz" "$RESULTS/${name}.npz"
  echo "[$(date '+%H:%M:%S')] $name done. npz + mp4 at $RESULTS/"
}

if [[ "$MODE" == "parallel" ]]; then
  echo "Mode: parallel — baseline on GPU $GPU_BASELINE, warm on GPU $GPU_WARM"
  run_one E059_baseline "$GPU_BASELINE" &
  PID0=$!
  run_one E059_warm "$GPU_WARM" +warmstart_qpos_path=$WARMSTART &
  PID1=$!
  wait $PID0 || { echo "baseline failed"; exit 1; }
  wait $PID1 || { echo "warm failed"; exit 1; }
else
  echo "Mode: serial on GPU $GPU_BASELINE"
  run_one E059_baseline "$GPU_BASELINE"
  run_one E059_warm "$GPU_BASELINE" +warmstart_qpos_path=$WARMSTART
fi

echo
echo "=== E059 train done. Results: ==="
ls -lh "$RESULTS"
