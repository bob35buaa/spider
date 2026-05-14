#!/usr/bin/env bash
# E067 — port HDMI body partition (single-var: N) vs full HDMI clone (NS).
#   GPU0: E067-N  box023  (narrow body partition only, keep E062 baseline)
#   GPU1: E067-NS box023  (narrow + soft actuator + exp task_obj — full HDMI)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-parallel}"
GPU_N="${2:-0}"
GPU_NS="${3:-1}"
RESULTS=workspace/core4d/results/E067
LOGS=logs/E067
mkdir -p "$RESULTS" "$LOGS"

echo "[$(date '+%H:%M:%S')] === scene snapshot (idempotent) ==="
bash workspace/core4d/scripts/convert/snapshot_scenes.sh E067 box023_person1

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
  echo "Mode: parallel — E067-N box023 GPU $GPU_N, E067-NS box023 GPU $GPU_NS"
  run_one E067N_box023  box023_person1 core4d_e067n_box023  "$GPU_N" &
  PID0=$!
  run_one E067NS_box023 box023_person1 core4d_e067ns_box023 "$GPU_NS" &
  PID1=$!
  wait $PID0 || { echo "E067-N failed"; exit 1; }
  wait $PID1 || { echo "E067-NS failed"; exit 1; }
else
  run_one E067N_box023  box023_person1 core4d_e067n_box023  "$GPU_N"
  run_one E067NS_box023 box023_person1 core4d_e067ns_box023 "$GPU_NS"
fi

echo
echo "=== E067 train done ==="
ls -lh "$RESULTS"/E067*.{npz,mp4} 2>/dev/null
