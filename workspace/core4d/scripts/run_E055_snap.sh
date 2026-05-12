#!/usr/bin/env bash
# E055 Path B (box023_person1): one-shot pipeline
#   1. Snap both palms onto box023 surface inside intent window
#   2. Render side-by-side ref vs snap mp4
#   3. Extract 5 keyframes for visual sanity-check
#
# Output (under workspace/core4d/results/E055/box023_person1/):
#   warmstart_qpos.npz       — qpos_ref + qpos_snap + intent_window + snap_mask
#   snap_diagnostics.csv     — per-frame palm-to-surface, IK iter, joint limits
#   snap_visualization.mp4   — 2x2 grid: ref (top) vs snap (bottom), front+side
#   keyframes/frame_*.jpg    — 5 keyframes around intent window
#
# Usage:
#   bash workspace/core4d/scripts/run_E055_snap.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

echo "=== E055 step 1/3: snap_box023.py ==="
.venv/bin/python workspace/core4d/scripts/E055/snap_box023.py

echo
echo "=== E055 step 2/3: visualize_snap.py (MUJOCO_GL=egl) ==="
MUJOCO_GL=egl .venv/bin/python workspace/core4d/scripts/E055/visualize_snap.py

echo
echo "=== E055 step 3/3: extract_snap_keyframes.sh ==="
bash workspace/core4d/scripts/E055/extract_snap_keyframes.sh

echo
echo "=== E055 done. Inspect: ==="
ls -lh workspace/core4d/results/E055/box023_person1/
