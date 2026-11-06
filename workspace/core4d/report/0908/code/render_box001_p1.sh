#!/usr/bin/env bash
# Render the three white-bg deliverables for the two box001 *p1* cases, mirroring the
# box001_20231023_108_p2 set (smplx / mixed / robot_only), all white-only, human-scale 0.8.
#
#   box001_20231023_108_p1 : both persons retargeted -> standard robot-anchored mixed
#       (primary=p1 robot, partner=p2 human anchored to p2's robot). trims derived by
#       object-motion xcorr: p2 mocap trim=33, p1<->p2 offset +12, window [12,119).
#   box001_20231003_2_039_p1 : ONLY p1 retargeted (no p2 robot) -> OBJECT-anchored mixed
#       (partner p2 human mapped into p1's CEM frame through the shared box). p1 mocap
#       trim=14, window [0,100). 4-token case id (date=20231003_2) now handled by parse_case.
#
# Non-invasive adaptation (no edits to the proven box021 logic beyond optional flags):
# idempotent symlinks resolve box001's real E198/G1A2 names, derived trims via CLI.
set -euo pipefail

REPO=/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/spider
cd "$REPO"
PY=.venv/bin/python
CODE=workspace/core4d/report/0908/code
VIZ="$REPO/workspace/core4d/report/0908/paper_results/viz"
PREVIEW="${1:-0}"

CEMDIR="$REPO/workspace/core4d/results/E198/s6_downstream/cem/full_g1a2_box001"
PROC="$REPO/example_datasets/processed/core4d/unitree_g1/humanoid_object"
SMPLX="/mnt/ali-sh-1/usr/xiayibo/xyb_data_tidal_alsh/other-datasets/mocap_data/human_model_files/smplx/SMPLX_NEUTRAL.npz"
SCENE_REAL=scene_act_E194_G1_expansion_rubberHull_PRG_gravcomp.xml
SCENE_LINK=scene_act_E198_lowerbody_physics.xml

echo "=== [setup] idempotent symlinks ==="
# -- box001_20231023_108 (both persons) --
C=box001_20231023_108
for p in p1 p2; do ln -sfn "E198_${C}_${p}_G1A2.npz" "$CEMDIR/E198_${C}_${p}_PRG.npz"; done
ln -sfn "$SCENE_REAL" "$PROC/dcv3_omnirt_v1_ref_fk_${C}_p2/$SCENE_LINK"
ln -sfn "$SCENE_REAL" "$PROC/dcv3_omnirt_v2_ref_fk_${C}_p1/$SCENE_LINK"
ln -sfn "dcv3_omnirt_v2_ref_fk_${C}_p1" "$PROC/dcv3_omnirt_v1_ref_fk_${C}_p1"   # p1 under v1 prefix
# -- box001_20231003_2_039 (p1 only) --
ln -sfn "$SCENE_REAL" "$PROC/dcv3_omnirt_v2_ref_fk_box001_20231003_2_039_p1/$SCENE_LINK"

# ------------------------------------------------------------------ 108_p1 (robot-anchored)
C=box001_20231023_108; OUT="$VIZ/${C}_p1"; mkdir -p "$OUT"
echo "=== [108_p1 1/3] mixed (robot-anchored): p1 robot + p2 human + box ==="
PYOPENGL_PLATFORM=osmesa MUJOCO_GL=osmesa "$PY" "$CODE/viz_mixed_robot_smplx.py" \
  --case "$C" --primary p1 --partner p2 --exp E198 --cem-dir "$CEMDIR" \
  --scene-prefix dcv3_omnirt_v1_ref_fk \
  --p1-trim-start 33 --p1-to-p2 12 --window 12,119 --human-scale 0.8 \
  --white-only --out "$OUT" --preview "$PREVIEW"
echo "=== [108_p1 2/3] smplx reference (aligned to mixed window: mocap [31,138)) ==="
PYOPENGL_PLATFORM=osmesa "$PY" "$CODE/viz_smplx_reference.py" \
  --case "$C" --primary p1 --blue-person p2 --window 31,138 --white-only --out "$OUT" --preview "$PREVIEW"
echo "=== [108_p1 3/3] robot_only: p1 robot + box (aligned window [12,119)) ==="
PYOPENGL_PLATFORM=osmesa MUJOCO_GL=osmesa "$PY" "$CODE/viz_robot_only_p2.py" \
  --case "$C" --primary p1 --exp E198 --cem-dir "$CEMDIR" --cem-suffix G1A2 \
  --scene-prefix dcv3_omnirt_v2_ref_fk --window 12,119 --out "$OUT" --preview "$PREVIEW"

# ------------------------------------------------------------------ 039_p1 (object-anchored)
C=box001_20231003_2_039; OUT="$VIZ/${C}_p1"; mkdir -p "$OUT"
echo "=== [039_p1 1/3] mixed (object-anchored): p1 robot + p2 human(box-anchored) + box ==="
PYOPENGL_PLATFORM=osmesa MUJOCO_GL=osmesa "$PY" "$CODE/viz_mixed_obj_anchor.py" \
  --case "$C" --primary p1 --partner p2 --exp E198 --cem-dir "$CEMDIR" --cem-suffix G1A2 \
  --smplx-model "$SMPLX" --scene-prefix dcv3_omnirt_v2_ref_fk --primary-trim 14 \
  --window 0,100 --human-scale 0.8 --out "$OUT" --preview "$PREVIEW"
echo "=== [039_p1 2/3] smplx reference (aligned to mixed window: mocap [14,114)) ==="
PYOPENGL_PLATFORM=osmesa "$PY" "$CODE/viz_smplx_reference.py" \
  --case "$C" --primary p1 --blue-person p2 --window 14,114 --smplx-model "$SMPLX" --white-only --out "$OUT" --preview "$PREVIEW"
echo "=== [039_p1 3/3] robot_only: p1 robot + box ==="
PYOPENGL_PLATFORM=osmesa MUJOCO_GL=osmesa "$PY" "$CODE/viz_robot_only_p2.py" \
  --case "$C" --primary p1 --exp E198 --cem-dir "$CEMDIR" --cem-suffix G1A2 \
  --scene-prefix dcv3_omnirt_v2_ref_fk --window 0,100 --out "$OUT" --preview "$PREVIEW"

echo "=== done ==="
for C in box001_20231023_108_p1 box001_20231003_2_039_p1; do
  echo "$C:"; ls -d "$VIZ/$C"/*_frames_white 2>/dev/null
done
