#!/usr/bin/env bash
# E060.1: Phase 3 step 1 — kill ori reward (contact_hdmi_ori_weight=0.0)
# Tests audit log §2.2 finding: hardcoded palm_normal [0,∓1,0] is noise on
# box023 L (mean dot 0.04, 50/50 sign), suboptimal on R (+0.51 vs +x +0.61).
#
# Same data layer as E060.0 (3-box hand + box023 margin 0.90).
# Same parallel scheme as train_E060_0.sh.
#
# Output (under workspace/core4d/results/E060/):
#   E060_1_box023.{npz,mp4,log}
#   E060_1_bucket005_s2.{npz,mp4,log}
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-parallel}"
GPU_BOX="${2:-0}"
GPU_BUCKET="${3:-1}"
RESULTS=workspace/core4d/results/E060
LOGS=logs/E060
mkdir -p "$RESULTS" "$LOGS"

# Snapshot is idempotent — already done by E060.0; rerunning to update manifest with current HEAD
echo "[$(date '+%H:%M:%S')] === scene snapshot (idempotent) ==="
bash workspace/core4d/scripts/convert/snapshot_scenes.sh E060 \
  box023_person1 bucket005_s2_person1

run_one() {
  local name=$1 task=$2 gpu=$3
  local out_dir="$RESULTS/${name}_outdir"
  mkdir -p "$out_dir"
  echo "[$(date '+%H:%M:%S')] === $name (task=$task, GPU $gpu, ori_weight=0.0) ==="
  CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl PYTHONUNBUFFERED=1 .venv/bin/python -u examples/run_mjwp.py \
    +override=core4d_e041c \
    task=$task \
    contact_hdmi_ori_weight=0.0 \
    +use_torch_compile=false \
    output_dir="$out_dir" \
    video_output_path="$RESULTS/${name}.mp4" \
    > "$LOGS/${name}.log" 2>&1
  cp "$out_dir/trajectory_mjwp_act.npz" "$RESULTS/${name}.npz"
  echo "[$(date '+%H:%M:%S')] $name done."
}

if [[ "$MODE" == "parallel" ]]; then
  echo "Mode: parallel — box023 on GPU $GPU_BOX, bucket005_s2 on GPU $GPU_BUCKET"
  run_one E060_1_box023 box023_person1 "$GPU_BOX" &
  PID0=$!
  run_one E060_1_bucket005_s2 bucket005_s2_person1 "$GPU_BUCKET" &
  PID1=$!
  wait $PID0 || { echo "box023 failed"; exit 1; }
  wait $PID1 || { echo "bucket005_s2 failed"; exit 1; }
else
  echo "Mode: serial on GPU $GPU_BOX"
  run_one E060_1_box023 box023_person1 "$GPU_BOX"
  run_one E060_1_bucket005_s2 bucket005_s2_person1 "$GPU_BOX"
fi

echo
echo "=== E060.1 train done. Results: ==="
ls -lh "$RESULTS"/E060_1_*.{npz,mp4} 2>/dev/null
