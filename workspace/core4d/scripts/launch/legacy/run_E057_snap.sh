#!/usr/bin/env bash
# E057 Path B (bucket005_s2_person1): one-shot pipeline.
#
#   1. Snap both palms onto bucket005_s2 surface inside intent window
#   2. Render side-by-side ref vs snap mp4
#   3. Extract 5 keyframes for visual sanity-check
#   4. Verify post-snap main faces match E056 (L on -yz, R on +yz)
#
# Output (under workspace/core4d/results/E057/bucket005_s2_person1/):
#   warmstart_qpos.npz       — qpos_ref + qpos_snap + intent_window + snap_mask
#   snap_diagnostics.csv     — per-frame palm-to-surface, IK iter, joint limits
#   snap_visualization.mp4   — ref (top) vs snap (bottom), front+side
#   keyframes/frame_*.jpg    — 5 keyframes around intent window
#   face_verification.csv    — REF vs SNAP main face per hand
#   face_dist_snap.png       — 2x2 face dist time series (REF/SNAP × L/R)
#
# Usage:
#   bash workspace/core4d/scripts/run_E057_snap.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

echo "=== E057 step 1/4: snap_bucket005_s2.py ==="
.venv/bin/python workspace/core4d/scripts/E057/snap_bucket005_s2.py

echo
echo "=== E057 step 2/4: visualize_snap.py (MUJOCO_GL=egl) ==="
MUJOCO_GL=egl .venv/bin/python workspace/core4d/scripts/E057/visualize_snap.py

echo
echo "=== E057 step 3/4: extract_snap_keyframes.sh ==="
bash workspace/core4d/scripts/E057/extract_snap_keyframes.sh

echo
echo "=== E057 step 4/4: verify_snap_face.py (C6) ==="
.venv/bin/python workspace/core4d/scripts/E057/verify_snap_face.py

echo
echo "=== E057 done. Inspect: ==="
ls -lh workspace/core4d/results/E057/bucket005_s2_person1/
