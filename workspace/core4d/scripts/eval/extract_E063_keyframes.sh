#!/usr/bin/env bash
# Extract dense keyframes for E063, focused on box023 carry → place → recovery
# transition window (t=2.0-2.7s in log 79 §3 was the fall window in E062).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
RESULTS=workspace/core4d/results/E063
KF_DIR="$RESULTS/keyframes"
mkdir -p "$KF_DIR"

# box023 dense around fall window: pre-carry, lift, transition, fall window, recovery, end
# (matches log 79 §2 reported timestamps)
BOX023_TS=(0.40 1.00 1.67 2.00 2.33 2.66 3.00 3.33 4.00)

# box025 sparse: just verify no regression — start, mid-carry, end
BOX025_TS=(0.40 1.50 2.50 3.50)

extract() {
  local name=$1; shift
  local mp4="$RESULTS/${name}.mp4"
  if [ ! -f "$mp4" ]; then
    echo "missing $mp4"; return
  fi
  local i=0
  for ts in "$@"; do
    i=$((i+1))
    out="$KF_DIR/${name}_kf${i}_t${ts}s.jpg"
    ffmpeg -loglevel error -y -ss "$ts" -i "$mp4" -frames:v 1 -q:v 2 "$out"
    echo "  $out"
  done
}

echo "=== E063_box023 dense keyframes (carry→fall→recovery window) ==="
extract E063_box023 "${BOX023_TS[@]}"

echo "=== E063_box025 keyframes (regression guard) ==="
extract E063_box025 "${BOX025_TS[@]}"

echo
echo "Total extracted:"
ls "$KF_DIR"/E063_*_kf*.jpg | wc -l
