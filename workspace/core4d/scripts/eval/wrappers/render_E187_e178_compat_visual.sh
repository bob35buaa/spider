#!/usr/bin/env bash
# Render and sample fixed frames for the failed E187 Gate S0 bucket007 replay.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
FRAME_SCRIPT="/home/ubuntu/.codex/skills/video-frames/scripts/frame.sh"
CASE_ROOT="example_datasets/processed/core4d/unitree_g1/humanoid_object/dcv3_omnirt_v2_ref_fk_bucket007_20231020_055_p1"
RESULT_ROOT="workspace/core4d/results/E187/s0_environment/e178_compat"
SCENE="$CASE_ROOT/scene_act_E178_contactAlignedTop.xml"
KIN="$CASE_ROOT/0/trajectory_kinematic.npz"
PHYS="$RESULT_ROOT/cem/E187_e178_compat_bucket007_20231020_055_p1.npz"
CURRENT_VIDEO="$RESULT_ROOT/render/E187_e178_compat_bucket007_20231020_055_p1_kin_vs_phys.mp4"
HISTORICAL_VIDEO="workspace/core4d/results/E178/s6_downstream/render/full/E178_bucket007_20231020_055_p1_contactAlignedTop_full.mp4"
FRAME_ROOT="$RESULT_ROOT/eval/visual"

test -f "$FRAME_SCRIPT"
test -f "$SCENE"
test -f "$KIN"
test -f "$PHYS"
test -f "$HISTORICAL_VIDEO"
mkdir -p "$(dirname "$CURRENT_VIDEO")" "$FRAME_ROOT"

if [[ ! -f "$CURRENT_VIDEO" ]]; then
  MUJOCO_GL=egl "$PYTHON_BIN" \
    workspace/hdmi_reproduce/scripts/render_trajectory_video.py \
    --scene "$SCENE" \
    --kin "$KIN" \
    --phys "$PHYS" \
    --output "$CURRENT_VIDEO" \
    --width 720 \
    --height 480 \
    --fps 25
fi

for sample in start:00:00:00.20 middle:00:00:01.66 end:00:00:03.10; do
  label="${sample%%:*}"
  timestamp="${sample#*:}"
  bash "$FRAME_SCRIPT" "$CURRENT_VIDEO" --time "$timestamp" \
    --out "$FRAME_ROOT/e187_${label}.jpg"
  bash "$FRAME_SCRIPT" "$HISTORICAL_VIDEO" --time "$timestamp" \
    --out "$FRAME_ROOT/e178_${label}.jpg"
done

ffprobe -v error \
  -show_entries format=duration \
  -show_entries stream=width,height,r_frame_rate,nb_frames \
  -of json "$CURRENT_VIDEO" > "$FRAME_ROOT/e187_video_probe.json"
ffprobe -v error \
  -show_entries format=duration \
  -show_entries stream=width,height,r_frame_rate,nb_frames \
  -of json "$HISTORICAL_VIDEO" > "$FRAME_ROOT/e178_video_probe.json"

sha256sum "$CURRENT_VIDEO" "$FRAME_ROOT"/*.jpg > "$FRAME_ROOT/sha256sum.txt"
