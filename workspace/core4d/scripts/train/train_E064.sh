#!/usr/bin/env bash
# E064 Tier 2 + threshold raise — extends E063 (T1) with:
#   local_frame_root_sigma 0.5→0.3, contact_hdmi_gain 5.0→3.0,
#   stability_penalty_threshold 0.55→0.65
#
# Configs:
#   examples/config/override/core4d_e064_box023.yaml  ← target case
#   examples/config/override/core4d_e064_box025.yaml  ← regression guard
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-parallel}"
GPU_BOX023="${2:-0}"
GPU_BOX025="${3:-1}"
RESULTS=workspace/core4d/results/E064
LOGS=logs/E064
mkdir -p "$RESULTS" "$LOGS"

echo "[$(date '+%H:%M:%S')] === scene snapshot (idempotent) ==="
bash workspace/core4d/scripts/convert/snapshot_scenes.sh E064 box023_person1 box025_person1

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
  echo "Mode: parallel — box023 GPU $GPU_BOX023, box025 GPU $GPU_BOX025"
  run_one E064_box023 box023_person1 core4d_e064_box023 "$GPU_BOX023" &
  PID0=$!
  run_one E064_box025 box025_person1 core4d_e064_box025 "$GPU_BOX025" &
  PID1=$!
  wait $PID0 || { echo "box023 failed"; exit 1; }
  wait $PID1 || { echo "box025 failed"; exit 1; }
else
  run_one E064_box023 box023_person1 core4d_e064_box023 "$GPU_BOX023"
  run_one E064_box025 box025_person1 core4d_e064_box025 "$GPU_BOX025"
fi

echo
echo "=== E064 train done ==="
ls -lh "$RESULTS"/E064_*.{npz,mp4} 2>/dev/null
