#!/usr/bin/env bash
# Extract A/B keyframes from E059 baseline.mp4 + warm.mp4 for visual comparison.
# 5 timestamps each, covering pre/start/mid/end/post intent.
# box023 intent (21, 78) at 30 fps → t = 0.70s start, 1.65s mid, 2.60s end.
#
# Usage:
#   bash workspace/core4d/scripts/eval/extract_E059_keyframes.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

RESULTS=workspace/core4d/results/E059
OUT=$RESULTS/keyframes
FRAME_SH=/root/.cc-mirror/codewiz-cc/config/skills/video-frames/scripts/frame.sh

mkdir -p "$OUT"

declare -a STAMPS=("0.40" "0.70" "1.65" "2.60" "3.00")

for src in E059_baseline E059_warm; do
  vid="$RESULTS/${src}.mp4"
  if [[ ! -f "$vid" ]]; then
    echo "[!] missing $vid"; continue
  fi
  for t in "${STAMPS[@]}"; do
    out="$OUT/${src}_t${t}s.jpg"
    bash "$FRAME_SH" "$vid" --time "$t" --out "$out" 2>/dev/null
    echo "  $out"
  done
done

echo "Done. Frames in $OUT/"
