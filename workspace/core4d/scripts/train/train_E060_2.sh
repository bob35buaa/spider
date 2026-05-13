#!/usr/bin/env bash
# E060.2: Phase 3 step 2 — case-correct palm_normal (+x = audit §2.2 true best).
#
# Per-case yamls (since Hydra list overrides via CLI are fragile):
#   box023:       core4d_e060_2_box023.yaml       L=[1,0,0]  R=[1,0,0]
#   bucket005_s2: core4d_e060_2_bucket005_s2.yaml L=[0,-1,0] R=[1,0,0]
#
# Tests revised hypothesis (E060.1 log §4): hardcoded palm normal is an
# implicit wrist-orientation prior — case-correct prior should help BOTH
# cases (preserves prior + aligns with actual contact geometry).
#
# Same data layer as E060.0/.1 (3-box hand + box023 margin 0.90).
# Same parallel scheme as train_E060_0.sh / train_E060_1.sh.
#
# Output (under workspace/core4d/results/E060/):
#   E060_2_box023.{npz,mp4,log}
#   E060_2_bucket005_s2.{npz,mp4,log}
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-parallel}"
GPU_BOX="${2:-0}"
GPU_BUCKET="${3:-1}"
RESULTS=workspace/core4d/results/E060
LOGS=logs/E060
mkdir -p "$RESULTS" "$LOGS"

echo "[$(date '+%H:%M:%S')] === scene snapshot (idempotent) ==="
bash workspace/core4d/scripts/convert/snapshot_scenes.sh E060 \
  box023_person1 bucket005_s2_person1

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
  echo "Mode: parallel — box023 on GPU $GPU_BOX, bucket005_s2 on GPU $GPU_BUCKET"
  run_one E060_2_box023 box023_person1 core4d_e060_2_box023 "$GPU_BOX" &
  PID0=$!
  run_one E060_2_bucket005_s2 bucket005_s2_person1 core4d_e060_2_bucket005_s2 "$GPU_BUCKET" &
  PID1=$!
  wait $PID0 || { echo "box023 failed"; exit 1; }
  wait $PID1 || { echo "bucket005_s2 failed"; exit 1; }
else
  echo "Mode: serial on GPU $GPU_BOX"
  run_one E060_2_box023 box023_person1 core4d_e060_2_box023 "$GPU_BOX"
  run_one E060_2_bucket005_s2 bucket005_s2_person1 core4d_e060_2_bucket005_s2 "$GPU_BOX"
fi

echo
echo "=== E060.2 train done. Results: ==="
ls -lh "$RESULTS"/E060_2_*.{npz,mp4} 2>/dev/null
