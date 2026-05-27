#!/usr/bin/env bash
# E029: preprocess COLA-style dynamic D6 support body assets.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

FORCE=0
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

RESULTS="workspace/core4d_collab_retarget/results/E029/d6"
mkdir -p "$RESULTS"

ARGS=()
if [ "$FORCE" = "1" ]; then
  ARGS+=(--force)
fi

.venv/bin/python workspace/core4d_collab_retarget/scripts/E029/generate_e029_d6_assets.py "${ARGS[@]}"
.venv/bin/python workspace/core4d_collab_retarget/scripts/E029/generate_e029_overrides.py

echo "E029 D6 preprocess done."

