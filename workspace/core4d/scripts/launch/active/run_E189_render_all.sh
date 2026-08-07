#!/usr/bin/env bash
# Render all 43 E189 Full trajectories and their E172/E173-vs-E189 paired
# videos (box004<-E172, box024/box001<-E173).
#
# IMPORTANT, DO NOT "FIX" THIS TO unset MUJOCO_GL:
# run_E189_local_8gpu.sh / run_cem_queue.py (CEM launch, save_video=false)
# unset MUJOCO_GL on purpose, because glfw's ImportError-on-missing-DISPLAY
# is safely swallowed by mujoco/__init__.py when no renderer is ever
# constructed. THIS SCRIPT IS DIFFERENT: it calls mujoco.Renderer() to
# produce real frames, which needs an actual GL context. Verified
# empirically in this shell: MUJOCO_GL unset -> glfw tries to open a
# DISPLAY and mujoco.Renderer() raises mujoco.FatalError (not swallowed);
# MUJOCO_GL=egl succeeds. This matches the already-working E179 render
# entry (run_E179_render_all.sh), which also exports MUJOCO_GL=egl. Keep
# egl here even if a future edit tries to align this file with the CEM
# scripts' "unset" convention -- that convention does not apply to
# rendering.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec "$PYTHON_BIN" \
  workspace/core4d/scripts/experiments/E189/render_paired_results.py \
  "$@"
