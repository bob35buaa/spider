#!/usr/bin/env bash
# E061: sphere baseline verification.
# Question: does box025 + E041c on sphere hand recover historical pelvis_min ~0.575m / Stable 100%?
# Diagnostic intent: validate that 3-box port (commit fa2e181) is the SOLE regression source
# behind log 74's box025 collapse (3-box pelvis_min 0.253m vs sphere 0.575m).
#
# REQUIRES: Phase 1 (snapshot + git checkout fa2e181~1 of robot.xml + box025 scenes) already done.
# Verify sphere state with smoke test before invoking this script.
#
# Output:
#   workspace/core4d/results/E061/box025_e041c_sphere.{npz,mp4}
#   logs/E061/box025_e041c_sphere.log
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

GPU="${1:-0}"
RESULTS=workspace/core4d/results/E061
LOGS=logs/E061
mkdir -p "$RESULTS" "$LOGS"

out_dir="$RESULTS/box025_e041c_sphere_outdir"
mkdir -p "$out_dir"

echo "[$(date '+%H:%M:%S')] === E061 box025 + E041c sphere (GPU $GPU) ==="
CUDA_VISIBLE_DEVICES=$GPU MUJOCO_GL=egl PYTHONUNBUFFERED=1 .venv/bin/python -u examples/run_mjwp.py \
  +override=core4d_e041c \
  task=box025_person1 \
  +use_torch_compile=false \
  output_dir="$out_dir" \
  video_output_path="$RESULTS/box025_e041c_sphere.mp4" \
  > "$LOGS/box025_e041c_sphere.log" 2>&1

cp "$out_dir/trajectory_mjwp_act.npz" "$RESULTS/box025_e041c_sphere.npz"
echo "[$(date '+%H:%M:%S')] done. npz/mp4 under $RESULTS, log under $LOGS"
ls -lh "$RESULTS/box025_e041c_sphere.npz" "$RESULTS/box025_e041c_sphere.mp4"
