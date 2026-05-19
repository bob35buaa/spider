#!/usr/bin/env bash
# E023: generate lower-body geometry repair tasks and overrides.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

FORCE_ARGS=()
if [ "${1:-}" = "--force" ]; then
  FORCE_ARGS=(--force)
fi

mkdir -p workspace/core4d_collab_retarget/results/E023
.venv/bin/python workspace/core4d_collab_retarget/scripts/E023/generate_e023_assets.py "${FORCE_ARGS[@]}"
.venv/bin/python workspace/core4d_collab_retarget/scripts/E023/generate_e023_overrides.py
echo "E023 preprocess done."
