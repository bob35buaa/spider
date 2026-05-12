#!/usr/bin/env bash
# Extract N evenly-spaced keyframes from CORE4D processed visualization videos.
#
# Wraps the /video-frames skill's frame.sh into a multi-case multi-frame helper.
#
# Usage:
#   bash workspace/core4d/scripts/analyze/extract_case_keyframes.sh \
#       [--n 5] [--out workspace/core4d/results/E054/video_frames] \
#       <case_dir1> [<case_dir2> ...]
#
# Defaults to 5 frames; output goes to workspace/core4d/results/E054/video_frames/{case}/
# Each frame is named frame_{idx}_t{time}s.jpg.
#
# Example:
#   bash workspace/core4d/scripts/analyze/extract_case_keyframes.sh \
#       box023_person1 bucket001_person1
set -euo pipefail

PROC_DIR="example_datasets/processed/core4d/unitree_g1/humanoid_object"
FRAME_SH="/root/.cc-mirror/codewiz-cc/config/skills/video-frames/scripts/frame.sh"
N=5
OUT_BASE="workspace/core4d/results/E054/video_frames"
CASES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --n) N="$2"; shift 2 ;;
    --out) OUT_BASE="$2"; shift 2 ;;
    --) shift; while [[ $# -gt 0 ]]; do CASES+=("$1"); shift; done ;;
    *) CASES+=("$1"); shift ;;
  esac
done

if [[ ${#CASES[@]} -eq 0 ]]; then
  echo "Usage: $0 [--n N] [--out DIR] <case_name> [<case_name> ...]" >&2
  exit 2
fi

mkdir -p "$OUT_BASE"

for case in "${CASES[@]}"; do
  video="$PROC_DIR/$case/0/visualization_kinematic.mp4"
  if [[ ! -f "$video" ]]; then
    echo "[!] Missing video: $video" >&2
    continue
  fi
  out_dir="$OUT_BASE/$case"
  mkdir -p "$out_dir"

  duration=$(ffprobe -v error -show_entries format=duration \
    -of default=noprint_wrappers=1:nokey=1 "$video")
  echo "=== $case  duration=${duration}s ==="

  # N evenly-spaced timestamps: i/(N-1) * duration, i = 0..N-1
  python3 -c "
N=$N; D=$duration
for i in range(N):
    t = (i/(N-1))*D if N > 1 else D/2
    print(f'{i:02d} {t:.2f}')
" | while read -r idx t; do
    out_file="$out_dir/frame_${idx}_t${t}s.jpg"
    bash "$FRAME_SH" "$video" --time "$t" --out "$out_file" 2>/dev/null
    echo "  $out_file"
  done
done

echo "Done. Frames in $OUT_BASE/"
