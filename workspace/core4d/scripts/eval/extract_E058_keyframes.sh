#!/usr/bin/env bash
# Extract A/B keyframes from E058 baseline.mp4 + warm.mp4 for visual comparison.
# 5 timestamps each, covering pre/start/mid/end/post intent.
#
# Usage:
#   bash workspace/core4d/scripts/eval/extract_E058_keyframes.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

RESULTS=workspace/core4d/results/E058
OUT=$RESULTS/keyframes
FRAME_SH=/root/.cc-mirror/codewiz-cc/config/skills/video-frames/scripts/frame.sh

mkdir -p "$OUT"

declare -a STAMPS=("0.40" "0.70" "2.10" "3.55" "4.20")

for src in E058_baseline E058_warm; do
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
