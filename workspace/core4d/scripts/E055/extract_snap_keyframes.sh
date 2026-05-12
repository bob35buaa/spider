#!/usr/bin/env bash
# Extract keyframes from E055 snap visualization for visual sanity-check.
# Reuses the /video-frames skill's frame.sh, hard-wired to box023_person1.
#
# Picks frames around the intent window: pre-window, window-start, mid, end, post.
#
# Usage:
#   bash workspace/core4d/scripts/E055/extract_snap_keyframes.sh
set -euo pipefail

VID="workspace/core4d/results/E055/box023_person1/snap_visualization.mp4"
OUT_DIR="workspace/core4d/results/E055/box023_person1/keyframes"
FRAME_SH="/root/.cc-mirror/codewiz-cc/config/skills/video-frames/scripts/frame.sh"

mkdir -p "$OUT_DIR"

# intent_window=(21, 78) at 30 fps → t = 0.7s (start), 1.65s (mid), 2.6s (end)
declare -a STAMPS=(
  "00 0.40"   # pre-window: approach blend
  "01 0.70"   # intent start
  "02 1.65"   # intent mid
  "03 2.60"   # intent end
  "04 3.00"   # post: release blend
)

for entry in "${STAMPS[@]}"; do
  read -r idx t <<< "$entry"
  out="$OUT_DIR/frame_${idx}_t${t}s.jpg"
  bash "$FRAME_SH" "$VID" --time "$t" --out "$out" 2>/dev/null
  echo "  $out"
done

echo "Done. Frames in $OUT_DIR/"
