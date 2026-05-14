#!/usr/bin/env bash
# Extract dense keyframes for E067-A and E067-D on box023.
# Timestamps cover pre-contact (B1 window) → carry → place → recovery.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
RESULTS=workspace/core4d/results/E067
KF_DIR="$RESULTS/keyframes"
mkdir -p "$KF_DIR"

BOX023_TS=(0.40 0.60 0.80 1.00 1.50 2.00 2.50 3.00 4.00)

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

echo "=== E067N_box023 keyframes (pre-contact + carry + place) ==="
extract E067N_box023 "${BOX023_TS[@]}"
echo "=== E067NS_box023 keyframes (pre-contact + carry + place) ==="
extract E067NS_box023 "${BOX023_TS[@]}"

echo
echo "Total extracted: $(ls "$KF_DIR"/E067*_kf*.jpg 2>/dev/null | wc -l)"
