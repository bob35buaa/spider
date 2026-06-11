#!/usr/bin/env bash
# Extract dense keyframes for E064.
# box023 carry → place → recovery window (same as E063 to enable A/B comparison).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
RESULTS=workspace/core4d/results/E064
KF_DIR="$RESULTS/keyframes"
mkdir -p "$KF_DIR"

BOX023_TS=(0.40 1.00 1.67 2.00 2.33 2.66 3.00 3.33 4.00)
BOX025_TS=(0.40 1.50 2.50 3.50)

extract() {
  local name=$1; shift
  local mp4="$RESULTS/${name}.mp4"
  if [ ! -f "$mp4" ]; then echo "missing $mp4"; return; fi
  local i=0
  for ts in "$@"; do
    i=$((i+1))
    out="$KF_DIR/${name}_kf${i}_t${ts}s.jpg"
    ffmpeg -loglevel error -y -ss "$ts" -i "$mp4" -frames:v 1 -q:v 2 "$out"
    echo "  $out"
  done
}

echo "=== E064_box023 dense keyframes (carry→fall→recovery) ==="
extract E064_box023 "${BOX023_TS[@]}"

echo "=== E064_box025 keyframes (regression guard) ==="
extract E064_box025 "${BOX025_TS[@]}"

echo
echo "Total extracted:"
ls "$KF_DIR"/E064_*_kf*.jpg | wc -l
