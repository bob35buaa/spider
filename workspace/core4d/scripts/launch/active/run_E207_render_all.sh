#!/usr/bin/env bash
# E207 G1only -- render the 9 full-CEM rollouts to mp4 for the rule-9 visual review.
#
# GL backend: this host has a broken EGL and no GLFW context; only osmesa renders
# (verified 2026-09-05). The CEM runs themselves are headless and unaffected.
#
# Usage:
#   bash workspace/core4d/scripts/launch/active/run_E207_render_all.sh
#   DRY_RUN=1 bash workspace/core4d/scripts/launch/active/run_E207_render_all.sh
#   CASES="bucket003_20231018_001_p2" bash .../run_E207_render_all.sh
# Env overrides: CASES, MAX_FRAMES, OVERWRITE, DRY_RUN, MUJOCO_GL
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

MANIFEST="workspace/core4d/results/E207/s6_downstream/manifests/g1only_full_manifest.tsv"
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

echo "[run_E207_render] MUJOCO_GL=$MUJOCO_GL manifest=$MANIFEST ${ARGS[*]}"
exec .venv/bin/python "$RENDERER" "${ARGS[@]}"
