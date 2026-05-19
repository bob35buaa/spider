#!/usr/bin/env bash
# E022: generate box023_p1 contact-mask repair tasks and overrides.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

FORCE_ARGS=()
if [ "${1:-}" = "--force" ]; then
  FORCE_ARGS=(--force)
fi

mkdir -p workspace/core4d_collab_retarget/results/E022
.venv/bin/python workspace/core4d_collab_retarget/scripts/E022/generate_e022_masks.py "${FORCE_ARGS[@]}"
.venv/bin/python workspace/core4d_collab_retarget/scripts/E022/generate_e022_overrides.py
echo "E022 preprocess done."
