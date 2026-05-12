#!/usr/bin/env bash
# E056: Multi-case hand-face diagnosis pipeline.
#
#   1. Run multi_case_face_diagnosis.py → 6 face_dist.png + summary csv + summary md
#   2. Extract video verification frames for 3 representative cases
#
# Output (under workspace/core4d/results/E056/):
#   case_grasp_type_summary.csv   — 6 cases × 15 columns classification
#   E056_summary.md               — human-readable summary + E057 recommendation
#   <case>_face_dist.png          — per-case time-series plot (×6)
#   video_verify/<case>_t<t>s.jpg — visual ground-truth frames (×9)
#
# Usage:
#   bash workspace/core4d/scripts/run_E056_diagnosis.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

echo "=== E056 step 1/2: multi_case_face_diagnosis.py ==="
.venv/bin/python workspace/core4d/scripts/E056/multi_case_face_diagnosis.py

echo
echo "=== E056 step 2/2: video verification frames ==="
PROC=example_datasets/processed/core4d/unitree_g1/humanoid_object
FRAME_SH=/root/.cc-mirror/codewiz-cc/config/skills/video-frames/scripts/frame.sh
OUT_DIR=workspace/core4d/results/E056/video_verify
mkdir -p "$OUT_DIR"

# 3 representative cases × 3 timestamps each:
#   box023        → expected 垂直 (lift bottom + pull far)
#   bucket005_s2  → expected 对侧 (squeeze both sides) ← ⭐ E057 recommended
#   box021        → expected 同面 (both on top, anomaly)
declare -A CASE_TIMES=(
  [box023_person1]="1.0 1.65 2.3"
  [bucket005_s2_person1]="1.5 2.5 3.5"
  [box021_person1]="1.0 2.0 3.0"
)

for case in "${!CASE_TIMES[@]}"; do
  vid="$PROC/$case/0/visualization_kinematic.mp4"
  if [[ ! -f "$vid" ]]; then
    echo "[!] missing $vid"; continue
  fi
  for t in ${CASE_TIMES[$case]}; do
    out="$OUT_DIR/${case%_person*}_t${t}s.jpg"
    bash "$FRAME_SH" "$vid" --time "$t" --out "$out" 2>/dev/null
    echo "  $out"
  done
done

echo
echo "=== E056 done. Inspect: ==="
ls -lh workspace/core4d/results/E056/ workspace/core4d/results/E056/video_verify/
