#!/usr/bin/env bash
# E028: D003 Box021 13-case preprocess for local serial SPIDER dynamic retarget.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

FORCE=0
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

RESULTS="workspace/core4d_collab_retarget/results/E028"
mkdir -p "$RESULTS"

.venv/bin/python workspace/core4d_collab_retarget/scripts/E028/build_e028_manifest.py
if [ "$FORCE" = "1" ]; then
  .venv/bin/python workspace/core4d_collab_retarget/scripts/E028/generate_e028_assets.py --force
else
  .venv/bin/python workspace/core4d_collab_retarget/scripts/E028/generate_e028_assets.py
fi
.venv/bin/python workspace/core4d_collab_retarget/scripts/E028/generate_e028_overrides.py

echo "E028 preprocess done."
