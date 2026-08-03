#!/usr/bin/env bash
# E187 Gate S0 diagnostic: quantify same-source MJWarp run-to-run determinism.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
OVERRIDE="core4d_E178_bucket007_20231020_055_p1_contactAlignedTop"
TASK="dcv3_omnirt_v2_ref_fk_bucket007_20231020_055_p1"
ROOT="workspace/core4d/results/E187/s0_environment/e178_compat/repeat_v1"
LOG_ROOT="logs/E187/s0/e178_compat/repeat_v1"
A_OUT="$ROOT/current_a_outdir"
B_OUT="$ROOT/current_b_outdir"
A_NPZ="$A_OUT/trajectory_mjwp_act.npz"
B_NPZ="$B_OUT/trajectory_mjwp_act.npz"
REPORT="$ROOT/current_source_repeat_comparison.json"

if [[ -e "$A_NPZ" || -e "$B_NPZ" || -e "$REPORT" ]]; then
  echo "E187 repeat_v1 output already exists; refusing to overwrite." >&2
  exit 2
fi
mkdir -p "$A_OUT" "$B_OUT" "$LOG_ROOT"

COMMON_ARGS=(
  "+override=$OVERRIDE"
  "task=$TASK"
  "+use_torch_compile=false"
  "save_video=false"
  "video_camera=auto"
  "num_samples=1024"
  "max_num_iterations=32"
  "seed=0"
  "max_sim_steps=16"
)

nvidia-smi \
  --query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw \
  --format=csv,noheader,nounits > "$ROOT/local_gpu_before.csv"
nvidia-smi \
  --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
  --format=csv,noheader,nounits > "$ROOT/local_compute_before.csv"

for run in a b; do
  if [[ "$run" == "a" ]]; then
    outdir="$A_OUT"
  else
    outdir="$B_OUT"
  fi
  CUDA_VISIBLE_DEVICES="$GPU_ID" MUJOCO_GL=egl PYTHONPATH=. \
    "$PYTHON_BIN" -u examples/run_mjwp.py \
    "${COMMON_ARGS[@]}" \
    "output_dir=$outdir" \
    "video_output_path=$ROOT/current_${run}.mp4" \
    > "$LOG_ROOT/current_${run}.log" 2>&1
done

PYTHONPATH=. "$PYTHON_BIN" \
  workspace/core4d/scripts/experiments/E187/compare_e178_short_replay.py \
  --left "$A_NPZ" \
  --right "$B_NPZ" \
  --label current_source_repeat_v1 \
  --output "$REPORT"

nvidia-smi \
  --query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw \
  --format=csv,noheader,nounits > "$ROOT/local_gpu_after.csv"
nvidia-smi \
  --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
  --format=csv,noheader,nounits > "$ROOT/local_compute_after.csv"
