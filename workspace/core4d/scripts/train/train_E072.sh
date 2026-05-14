#!/usr/bin/env bash
# E072: analysis-only replay diagnosis for E071 box023 post-2s hold/place failure.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

RESULTS=workspace/core4d/results/E072
LOGS=logs/E072
VIDEO=workspace/core4d/results/E071/E071W02_box023.mp4
mkdir -p "$RESULTS/keyframes" "$LOGS"

echo "[$(date '+%H:%M:%S')] === E072 replay diagnosis ==="
.venv/bin/python workspace/core4d/scripts/eval/eval_E072.py | tee "$LOGS/eval_E072.log"

if command -v ffmpeg >/dev/null 2>&1 && [ -f "$VIDEO" ]; then
  echo "[$(date '+%H:%M:%S')] === extracting post-2s keyframes by frame index ==="
  for f in 100 115 130 145 160 166 168 180; do
    ffmpeg -y -loglevel error -i "$VIDEO" \
      -vf "select=eq(n\\,$f)" -frames:v 1 -vsync 0 \
      "$RESULTS/keyframes/f${f}.jpg"
  done
else
  echo "WARNING: ffmpeg or E071 video missing; keyframe extraction skipped."
fi

echo "=== E072 done ==="
