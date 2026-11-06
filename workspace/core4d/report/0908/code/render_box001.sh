#!/usr/bin/env bash
# Render the three white-bg deliverables for box001_20231023_108_p2, mirroring the
# box021_20231011_037_p2 reference set (smplx / mixed / robot_only).
#
# box001 differs from the box021-hardcoded defaults of the viz scripts:
#   - CEM rollout is E198 G1A2 (npz suffix `_G1A2`, dir `cem/full_g1a2_box001`),
#     not E170 `_PRG` under `cem/full`;
#   - scene_act file is `scene_act_E194_G1_expansion_rubberHull_PRG_gravcomp.xml`,
#     not `scene_act_E170_lowerbody_physics.xml`;
#   - p2 lives under the `dcv3_omnirt_v1_ref_fk` scene prefix, p1 under `_v2_`;
#   - p1 CEM (119 frames) is LONGER than p2 (107) and starts earlier in mocap
#     (p1 mocap trim 19, p2 mocap trim 33 -> p1[t] <-> p2[t-12]).
#
# We adapt WITHOUT editing the (box021-tuned) viz scripts: idempotent symlinks make
# the hardcoded `_PRG` / `scene_act_E198_lowerbody_physics.xml` / single-prefix names
# resolve to box001's real files, and the derived trim/offset are passed via CLI.
# Derived empirically (object-motion speed xcorr, ~0.92-0.95 corr):
#   p1 mocap trim start = 19 ; p2 mocap trim start = 33 ; p1<->p2 offset = -12.
set -euo pipefail

REPO=/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/spider
cd "$REPO"
PY=.venv/bin/python
CODE=workspace/core4d/report/0908/code
CASE=box001_20231023_108
OUT="$REPO/workspace/core4d/report/0908/paper_results/viz/${CASE}_p2"
PREVIEW="${1:-0}"            # pass an integer N for an N-frame preview; 0 = full clip

CEMDIR="$REPO/workspace/core4d/results/E198/s6_downstream/cem/full_g1a2_box001"
PROC="$REPO/example_datasets/processed/core4d/unitree_g1/humanoid_object"
SCENE_REAL=scene_act_E194_G1_expansion_rubberHull_PRG_gravcomp.xml
SCENE_LINK=scene_act_E198_lowerbody_physics.xml      # name the scripts build for --exp E198

echo "=== [setup] idempotent symlinks (box001 real files -> box021-style names) ==="
# CEM npz: _G1A2 -> _PRG so `{exp=E198}_{case}_{person}_PRG.npz` resolves
for p in p1 p2; do
  ln -sfn "E198_${CASE}_${p}_G1A2.npz" "$CEMDIR/E198_${CASE}_${p}_PRG.npz"
done
# scene_act file symlink inside each real scene dir
ln -sfn "$SCENE_REAL" "$PROC/dcv3_omnirt_v1_ref_fk_${CASE}_p2/$SCENE_LINK"
ln -sfn "$SCENE_REAL" "$PROC/dcv3_omnirt_v2_ref_fk_${CASE}_p1/$SCENE_LINK"
# mixed render uses ONE --scene-prefix for both persons; expose p1 (v2) under the v1 name
ln -sfn "dcv3_omnirt_v2_ref_fk_${CASE}_p1" "$PROC/dcv3_omnirt_v1_ref_fk_${CASE}_p1"

mkdir -p "$OUT"

echo "=== [1/3] mixed: p2 robot + p1 SMPLX human + object  (writes camera_align.json) ==="
PYOPENGL_PLATFORM=osmesa MUJOCO_GL=osmesa "$PY" "$CODE/viz_mixed_robot_smplx.py" \
  --case "$CASE" --primary p2 --partner p1 --exp E198 --cem-dir "$CEMDIR" \
  --scene-prefix dcv3_omnirt_v1_ref_fk \
  --p1-trim-start 19 --p1-to-p2 -12 --window 0,107 --human-scale 0.8 \
  --white-only --out "$OUT" --preview "$PREVIEW"

echo "=== [2/3] smplx reference: both persons + object (reads OUT/camera_align.json) ==="
PYOPENGL_PLATFORM=osmesa "$PY" "$CODE/viz_smplx_reference.py" \
  --case "$CASE" --primary p2 --window 33,140 --white-only --out "$OUT" --preview "$PREVIEW"

echo "=== [3/3] robot_only: ONLY p2's G1 + object, white bg (same cam as mixed) ==="
PYOPENGL_PLATFORM=osmesa MUJOCO_GL=osmesa "$PY" "$CODE/viz_robot_only_p2.py" \
  --case "$CASE" --primary p2 --exp E198 --cem-dir "$CEMDIR" --cem-suffix G1A2 \
  --scene-prefix dcv3_omnirt_v1_ref_fk --window 0,107 \
  --out "$OUT" --preview "$PREVIEW"

echo "=== done -> $OUT ==="
ls -d "$OUT"/{smplx,mixed,robot}_frames_white 2>/dev/null
