#!/usr/bin/env bash
# E071: scene_act ctrl mapping fix validation on box023.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

GPU="${1:-0}"
RESULTS=workspace/core4d/results/E071
LOGS=logs/E071
mkdir -p "$RESULTS" "$LOGS"

run_one() {
  local name=$1
  local override=$2
  local out_dir="$RESULTS/${name}_outdir"
  mkdir -p "$out_dir"
  echo "[$(date '+%H:%M:%S')] === $name (override=$override, GPU $GPU) ==="
  CUDA_VISIBLE_DEVICES=$GPU MUJOCO_GL=egl PYTHONUNBUFFERED=1 .venv/bin/python -u examples/run_mjwp.py \
    +override="$override" \
    task=box023_person1 \
    +use_torch_compile=false \
    output_dir="$out_dir" \
    video_output_path="$RESULTS/${name}.mp4" \
    > "$LOGS/${name}.log" 2>&1
  cp "$out_dir/trajectory_mjwp_act.npz" "$RESULTS/${name}.npz"
  echo "[$(date '+%H:%M:%S')] $name done."
}

echo "[$(date '+%H:%M:%S')] === scene snapshot (idempotent) ==="
bash workspace/core4d/scripts/convert/snapshot_scenes.sh E071 box023_person1
run_one E071W02_box023 core4d_e071w02_box023
.venv/bin/python workspace/core4d/scripts/eval/eval_E071.py
echo "=== E071 done ==="
