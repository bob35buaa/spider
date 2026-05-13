#!/usr/bin/env bash
# E060.0: Baseline (E041c, no warmstart) on box023 + bucket005_s2 with
# data-layer-fixed scenes (3-box hand + box023 margin 0.90).
#
# This validates whether E041c reward stack works once data-layer bugs
# (sphere hand collision + box023 oversized margin) are fixed.
# See workspace/core4d/plan/70_E060_data_layer_fix_plan.md.
#
# Default: parallel — box023 on GPU 0, bucket005_s2 on GPU 1.
# Override: `bash train_E060_0.sh serial 0` for both serially on GPU 0.
#
# Output (under workspace/core4d/results/E060/):
#   E060_0_box023.{npz,mp4,log}
#   E060_0_bucket005_s2.{npz,mp4,log}

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-parallel}"
GPU_BOX="${2:-0}"
GPU_BUCKET="${3:-1}"
RESULTS=workspace/core4d/results/E060
LOGS=logs/E060
mkdir -p "$RESULTS" "$LOGS"

# Per dual-safeguard rule (.claude/rules/experiment.md §7): snapshot scenes first
# (idempotent — already snapshot in Phase 1, but re-running locks the post-train state)
echo "[$(date '+%H:%M:%S')] === scene snapshot ==="
bash workspace/core4d/scripts/convert/snapshot_scenes.sh E060 \
  box023_person1 bucket005_s2_person1

run_one() {
  local name=$1 task=$2 gpu=$3
  local out_dir="$RESULTS/${name}_outdir"
  mkdir -p "$out_dir"
  echo "[$(date '+%H:%M:%S')] === $name (task=$task, GPU $gpu, output_dir=$out_dir) ==="
  CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl PYTHONUNBUFFERED=1 .venv/bin/python -u examples/run_mjwp.py \
    +override=core4d_e041c \
    task=$task \
    +use_torch_compile=false \
    output_dir="$out_dir" \
    video_output_path="$RESULTS/${name}.mp4" \
    > "$LOGS/${name}.log" 2>&1
  cp "$out_dir/trajectory_mjwp_act.npz" "$RESULTS/${name}.npz"
  echo "[$(date '+%H:%M:%S')] $name done. npz + mp4 at $RESULTS/"
}

if [[ "$MODE" == "parallel" ]]; then
  echo "Mode: parallel — box023 on GPU $GPU_BOX, bucket005_s2 on GPU $GPU_BUCKET"
  run_one E060_0_box023 box023_person1 "$GPU_BOX" &
  PID0=$!
  run_one E060_0_bucket005_s2 bucket005_s2_person1 "$GPU_BUCKET" &
  PID1=$!
  wait $PID0 || { echo "box023 failed"; exit 1; }
  wait $PID1 || { echo "bucket005_s2 failed"; exit 1; }
else
  echo "Mode: serial on GPU $GPU_BOX"
  run_one E060_0_box023 box023_person1 "$GPU_BOX"
  run_one E060_0_bucket005_s2 bucket005_s2_person1 "$GPU_BOX"
fi

echo
echo "=== E060.0 train done. Results: ==="
ls -lh "$RESULTS"/*.{npz,mp4} 2>/dev/null
