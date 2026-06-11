#!/usr/bin/env bash
# Extract 5 keyframes per case from E060.0 baseline videos.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
RESULTS=workspace/core4d/results/E060

# box023 timestamps (per E059 keyframes: pre, intent_start, intent_mid, intent_end, post)
# box023 ref T=136 @ 30fps → 4.53s; intent (21, 78) → 0.70 - 2.60s
BOX023_TS=(0.40 0.70 1.65 2.60 3.00)

# bucket005_s2 ref T=148 @ 30fps → 4.93s; intent (19, 107) → 0.63 - 3.57s
BUCKET005_S2_TS=(0.40 0.65 2.10 3.55 4.00)

extract() {
  local name=$1; shift
  local mp4="$RESULTS/${name}.mp4"
  if [ ! -f "$mp4" ]; then
    echo "missing $mp4"; return
  fi
  local i=0
  for ts in "$@"; do
    i=$((i+1))
    out="$RESULTS/${name}_kf${i}_t${ts}s.jpg"
    ffmpeg -loglevel error -y -ss "$ts" -i "$mp4" -frames:v 1 -q:v 2 "$out"
    echo "  $out"
  done
}

echo "=== E060_0_box023 keyframes ==="
extract E060_0_box023 "${BOX023_TS[@]}"

echo "=== E060_0_bucket005_s2 keyframes ==="
extract E060_0_bucket005_s2 "${BUCKET005_S2_TS[@]}"

echo
echo "Total extracted:"
ls "$RESULTS"/E060_0_*_kf*.jpg | wc -l
