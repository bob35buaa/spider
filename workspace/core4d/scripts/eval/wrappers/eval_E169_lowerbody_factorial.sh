#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
STAGE="${1:-available}"
shift || true

args=(--stage "$STAGE")
if [ "$STAGE" = "full" ]; then
  args+=(--require-all)
fi
MUJOCO_GL="${MUJOCO_GL:-egl}" .venv/bin/python \
  workspace/core4d/scripts/eval/runners/eval_E169_lowerbody_factorial.py \
  "${args[@]}" "$@"
