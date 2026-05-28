#!/usr/bin/env bash
# E089: G1-Feasibility gate validation (A path: box021_person1).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-smoke}"   # smoke | local
GPU="${2:-0}"
VARIANT="${VARIANT:-E089A_box021_person1_upperobj}"
RESULTS="${RESULTS:-workspace/core4d/results/E089/A}"
LOGS="${LOGS:-logs/E089}"
mkdir -p "$RESULTS/keyframes" "$LOGS"

OVERRIDE="core4d_${VARIANT}"
OUT_DIR="$RESULTS/${VARIANT}_outdir"
mkdir -p "$OUT_DIR"

# Snapshot is the first responsibility of any training script per project rules
bash workspace/core4d/scripts/convert/snapshot_scenes.sh E089 \
  box021_person1 box021_person1_upperobj_e089 > "$LOGS/snapshot.log" 2>&1

if [ "$MODE" = "smoke" ]; then
  echo "[$(date '+%H:%M:%S')] === E089A SMOKE (24 step) GPU=${GPU} ==="
  CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_GL=egl PYTHONUNBUFFERED=1 .venv/bin/python -u examples/run_mjwp.py \
    +override="$OVERRIDE" \
    +use_torch_compile=false \
    max_num_iterations=4 \
    output_dir="${OUT_DIR}_smoke" \
    video_output_path="$RESULTS/${VARIANT}_smoke.mp4" \
    > "$LOGS/${VARIANT}_smoke.log" 2>&1
  echo "[$(date '+%H:%M:%S')] === smoke done ==="
elif [ "$MODE" = "local" ]; then
  echo "[$(date '+%H:%M:%S')] === E089A FULL CEM GPU=${GPU} ==="
  CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_GL=egl PYTHONUNBUFFERED=1 .venv/bin/python -u examples/run_mjwp.py \
    +override="$OVERRIDE" \
    +use_torch_compile=false \
    output_dir="$OUT_DIR" \
    video_output_path="$RESULTS/${VARIANT}.mp4" \
    > "$LOGS/${VARIANT}.log" 2>&1
  # copy primary trajectory file
  if [ -f "$OUT_DIR/trajectory_mjwp_act.npz" ]; then
    cp "$OUT_DIR/trajectory_mjwp_act.npz" "$RESULTS/${VARIANT}.npz"
  elif [ -f "$OUT_DIR/trajectory_mjwp.npz" ]; then
    cp "$OUT_DIR/trajectory_mjwp.npz" "$RESULTS/${VARIANT}.npz"
  fi
  echo "[$(date '+%H:%M:%S')] === full CEM done ==="
else
  echo "Unknown MODE: $MODE (use smoke|local)" >&2
  exit 2
fi
