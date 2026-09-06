#!/usr/bin/env bash
# E210 aug x G1only -- render the 15 full-CEM rollouts to mp4 for the rule-9 review.
#
# GL backend: this host has a broken EGL and no GLFW context; only osmesa renders
# (E207 F7, E202 C7). The CEM runs themselves are headless and unaffected.
#
# WHAT THESE VIDEOS CAN AND CANNOT SHOW (E208 F13, and it applies verbatim here):
# `spider/viewers/__init__.py::_auto_video_camera` recomputes lookat/radius EVERY
# FRAME from the union bbox of sim+ref bodies, so the camera tracks the object.
# Comparing absolute pose / on-screen position / size ACROSS videos is meaningless.
# On top of that an aug video's reference pane is the AUGMENTED reference, so
# "did the augmentation happen" is not visible either -- that is established
# numerically (manifest sha chain + the effective-offset column).
#   => Safe A/B here: E210 vs E202 for the SAME variant. Identical reference
#      trajectory, so the only thing that differs is the robot. That is also the
#      only strictly single-variable contrast in the experiment.
#   => NOT safe: E210 aug vs E207 orig. Different references -> different camera.
#      That contrast is numbers-only.
#
# Usage:
#   bash workspace/core4d/scripts/launch/active/run_E210_render_all.sh
#   DRY_RUN=1 bash workspace/core4d/scripts/launch/active/run_E210_render_all.sh
#   CASES="bucket007_20231003_2_021_p1" bash .../run_E210_render_all.sh
# Env overrides: CASES, MAX_FRAMES, OVERWRITE, DRY_RUN, MUJOCO_GL, MANIFEST
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

MANIFEST="${MANIFEST:-workspace/core4d/results/E210/s6_downstream/manifests/aug_g1only_full_manifest.tsv}"
RENDERER="workspace/core4d/scripts/experiments/E168/render_a100_cem_videos.py"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"

[[ -f "$MANIFEST" ]] || { echo "missing manifest: $MANIFEST" >&2; exit 2; }
head -1 "$MANIFEST" | grep -q $'\tvariant\t' || {
  echo "manifest lacks the 'variant' column the renderer needs; re-run build_manifest.py" >&2
  exit 2; }

ARGS=(--manifest "$MANIFEST" --pool all)
[[ -n "${CASES:-}" ]] && ARGS+=(--cases $CASES)
[[ -n "${MAX_FRAMES:-}" ]] && ARGS+=(--max-frames "$MAX_FRAMES")
[[ "${OVERWRITE:-0}" == "1" ]] && ARGS+=(--overwrite)
[[ "${DRY_RUN:-0}" == "1" ]] && ARGS+=(--dry-run)

echo "[run_E210_render] MUJOCO_GL=$MUJOCO_GL manifest=$MANIFEST ${ARGS[*]}"
exec .venv/bin/python "$RENDERER" "${ARGS[@]}"
