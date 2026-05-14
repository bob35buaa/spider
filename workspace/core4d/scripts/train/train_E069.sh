#!/usr/bin/env bash
# E069: first-tick ref-control warmup on box023.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-parallel}"
GPU_W02="${2:-0}"
GPU_W05="${3:-1}"
RESULTS=workspace/core4d/results/E069
LOGS=logs/E069
mkdir -p "$RESULTS" "$LOGS"

run_one() {
  local name=$1
  local override=$2
  local gpu=$3
  local out_dir="$RESULTS/${name}_outdir"
  mkdir -p "$out_dir"
  echo "[$(date '+%H:%M:%S')] === $name (override=$override, GPU $gpu) ==="
  CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl PYTHONUNBUFFERED=1 .venv/bin/python -u examples/run_mjwp.py \
    +override="$override" \
    task=box023_person1 \
    +use_torch_compile=false \
    output_dir="$out_dir" \
    video_output_path="$RESULTS/${name}.mp4" \
    > "$LOGS/${name}.log" 2>&1
  cp "$out_dir/trajectory_mjwp_act.npz" "$RESULTS/${name}.npz"
  echo "[$(date '+%H:%M:%S')] $name done."
}

case "$MODE" in
  parallel)
    echo "[$(date '+%H:%M:%S')] === scene snapshot (idempotent) ==="
    bash workspace/core4d/scripts/convert/snapshot_scenes.sh E069 box023_person1
    run_one E069W02_box023 core4d_e069w02_box023 "$GPU_W02" &
    PID0=$!
    run_one E069W05_box023 core4d_e069w05_box023 "$GPU_W05" &
    PID1=$!
    wait $PID0 || { echo "E069-W02 failed"; exit 1; }
    wait $PID1 || { echo "E069-W05 failed"; exit 1; }
    ;;
  single_w02)
    echo "[$(date '+%H:%M:%S')] === scene snapshot (idempotent) ==="
    bash workspace/core4d/scripts/convert/snapshot_scenes.sh E069 box023_person1
    run_one E069W02_box023 core4d_e069w02_box023 "$GPU_W02"
    ;;
  single_w05)
    echo "[$(date '+%H:%M:%S')] === scene snapshot (idempotent) ==="
    bash workspace/core4d/scripts/convert/snapshot_scenes.sh E069 box023_person1
    run_one E069W05_box023 core4d_e069w05_box023 "$GPU_W02"
    ;;
  eval)
    .venv/bin/python workspace/core4d/scripts/eval/eval_E069.py
    exit 0
    ;;
  *)
    echo "Unknown MODE=$MODE (expected: parallel | single_w02 | single_w05 | eval)" >&2
    exit 1
    ;;
esac

.venv/bin/python workspace/core4d/scripts/eval/eval_E069.py
echo "=== E069 done ==="
