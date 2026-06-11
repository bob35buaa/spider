#!/usr/bin/env bash
# Extract keyframes from E057 snap visualization for visual sanity-check.
# Reuses the /video-frames skill's frame.sh, hard-wired to bucket005_s2_person1.
#
# bucket005_s2 intent_window=(20, 107) at 30 fps → t = 0.67s (start), 2.12s (mid), 3.57s (end).
#
# Usage:
#   bash workspace/core4d/scripts/E057/extract_snap_keyframes.sh
set -euo pipefail

VID="workspace/core4d/results/E057/bucket005_s2_person1/snap_visualization.mp4"
OUT_DIR="workspace/core4d/results/E057/bucket005_s2_person1/keyframes"
FRAME_SH="/root/.cc-mirror/codewiz-cc/config/skills/video-frames/scripts/frame.sh"

mkdir -p "$OUT_DIR"

# 5 keyframes covering pre-window / start / mid / end / post
declare -a STAMPS=(
  "00 0.40"   # pre-window: approach blend
  "01 0.70"   # intent start (t≈0.67s)
  "02 2.10"   # intent mid (t≈2.12s)
  "03 3.55"   # intent end (t≈3.57s)
  "04 4.20"   # post: release blend
)

for entry in "${STAMPS[@]}"; do
  read -r idx t <<< "$entry"
  out="$OUT_DIR/frame_${idx}_t${t}s.jpg"
  bash "$FRAME_SH" "$VID" --time "$t" --out "$out" 2>/dev/null
  echo "  $out"
done

echo "Done. Frames in $OUT_DIR/"
