#!/usr/bin/env bash
# E073: dynamic contact target uses ref wrist+eef_offset for box023.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

GPU="${1:-0}"
RESULTS=workspace/core4d/results/E073
LOGS=logs/E073
NAME=E073_box023
OVERRIDE=core4d_e073_box023
mkdir -p "$RESULTS/keyframes" "$LOGS"

echo "[$(date '+%H:%M:%S')] === scene snapshot (idempotent) ==="
bash workspace/core4d/scripts/convert/snapshot_scenes.sh E073 box023_person1

echo "[$(date '+%H:%M:%S')] === $NAME (override=$OVERRIDE, GPU $GPU) ==="
CUDA_VISIBLE_DEVICES=$GPU MUJOCO_GL=egl PYTHONUNBUFFERED=1 .venv/bin/python -u examples/run_mjwp.py \
  +override="$OVERRIDE" \
  task=box023_person1 \
  +use_torch_compile=false \
  output_dir="$RESULTS/${NAME}_outdir" \
  video_output_path="$RESULTS/${NAME}.mp4" \
  > "$LOGS/${NAME}.log" 2>&1

cp "$RESULTS/${NAME}_outdir/trajectory_mjwp_act.npz" "$RESULTS/${NAME}.npz"

echo "[$(date '+%H:%M:%S')] === E073 eval ==="
.venv/bin/python workspace/core4d/scripts/eval/eval_E073.py | tee "$LOGS/eval_E073.log"

if command -v ffmpeg >/dev/null 2>&1 && [ -f "$RESULTS/${NAME}.mp4" ]; then
  echo "[$(date '+%H:%M:%S')] === extracting keyframes by frame index ==="
  for f in 100 115 130 145 160 166 168 180; do
    ffmpeg -y -loglevel error -i "$RESULTS/${NAME}.mp4" \
      -vf "select=eq(n\\,$f)" -frames:v 1 -vsync 0 \
      "$RESULTS/keyframes/f${f}.jpg"
  done
else
  echo "WARNING: ffmpeg or E073 video missing; keyframe extraction skipped."
fi

echo "=== E073 done ==="
