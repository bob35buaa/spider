#!/usr/bin/env bash
# E065 — task_obj_rew form ablation on box023 (per plan: box025 deprioritized, run later in R4).
#   GPU0: E065-A box023  (drop task_obj, scale=0)
#   GPU1: E065-D box023  (HDMI exp form, sigma=0.5, scale=1.0)
#
# Configs:
#   examples/config/override/core4d_e065a_box023.yaml
#   examples/config/override/core4d_e065d_box023.yaml
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-parallel}"
GPU_A="${2:-0}"
GPU_D="${3:-1}"
RESULTS=workspace/core4d/results/E065
LOGS=logs/E065
mkdir -p "$RESULTS" "$LOGS"

echo "[$(date '+%H:%M:%S')] === scene snapshot (idempotent) ==="
bash workspace/core4d/scripts/convert/snapshot_scenes.sh E065 box023_person1

run_one() {
  local name=$1 task=$2 override=$3 gpu=$4
  local out_dir="$RESULTS/${name}_outdir"
  mkdir -p "$out_dir"
  echo "[$(date '+%H:%M:%S')] === $name (override=$override, GPU $gpu) ==="
  CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl PYTHONUNBUFFERED=1 .venv/bin/python -u examples/run_mjwp.py \
    +override=$override \
    task=$task \
    +use_torch_compile=false \
    output_dir="$out_dir" \
    video_output_path="$RESULTS/${name}.mp4" \
    > "$LOGS/${name}.log" 2>&1
  cp "$out_dir/trajectory_mjwp_act.npz" "$RESULTS/${name}.npz"
  echo "[$(date '+%H:%M:%S')] $name done."
}

if [[ "$MODE" == "parallel" ]]; then
  echo "Mode: parallel — E065-A box023 GPU $GPU_A, E065-D box023 GPU $GPU_D"
  run_one E065A_box023 box023_person1 core4d_e065a_box023 "$GPU_A" &
  PID0=$!
  run_one E065D_box023 box023_person1 core4d_e065d_box023 "$GPU_D" &
  PID1=$!
  wait $PID0 || { echo "E065-A failed"; exit 1; }
  wait $PID1 || { echo "E065-D failed"; exit 1; }
else
  run_one E065A_box023 box023_person1 core4d_e065a_box023 "$GPU_A"
  run_one E065D_box023 box023_person1 core4d_e065d_box023 "$GPU_D"
fi

echo
echo "=== E065 train done ==="
ls -lh "$RESULTS"/E065*.{npz,mp4} 2>/dev/null
