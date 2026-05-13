#!/usr/bin/env bash
# E062: X1 auto palm_normal on sphere — box025 (baseline) + box023 (target).
#
# REQUIRES Phase 1 (sphere revert, commit 61abf4c) + Phase 2 (yamls generated).
# Auto palm_normals (per compute_palm_normal.py):
#   box025: L=[0,-1,0]  R=[0,+1,0]  (== E041c default; self-consistency check)
#   box023: L=[+1,0,0]  R=[+1,0,0]  (different from default; X1 contribution)
#
# Output:
#   workspace/core4d/results/E062/E062_box025_sphere_autopalm.{npz,mp4}
#   workspace/core4d/results/E062/E062_box023_sphere_autopalm.{npz,mp4}
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-parallel}"
GPU_BOX025="${2:-0}"
GPU_BOX023="${3:-1}"
RESULTS=workspace/core4d/results/E062
LOGS=logs/E062
mkdir -p "$RESULTS" "$LOGS"

echo "[$(date '+%H:%M:%S')] === scene snapshot (idempotent) ==="
bash workspace/core4d/scripts/convert/snapshot_scenes.sh E062 box025_person1 box023_person1

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
  echo "Mode: parallel — box025 GPU $GPU_BOX025, box023 GPU $GPU_BOX023"
  run_one E062_box025_sphere_autopalm box025_person1 core4d_e062_box025 "$GPU_BOX025" &
  PID0=$!
  run_one E062_box023_sphere_autopalm box023_person1 core4d_e062_box023 "$GPU_BOX023" &
  PID1=$!
  wait $PID0 || { echo "box025 failed"; exit 1; }
  wait $PID1 || { echo "box023 failed"; exit 1; }
else
  run_one E062_box025_sphere_autopalm box025_person1 core4d_e062_box025 "$GPU_BOX025"
  run_one E062_box023_sphere_autopalm box023_person1 core4d_e062_box023 "$GPU_BOX023"
fi

echo
echo "=== E062 train done ==="
ls -lh "$RESULTS"/E062_*.{npz,mp4} 2>/dev/null
