#!/usr/bin/env bash
# E028b: rebuild D003 Box021 candidate support-proxy anchors from contact centroids.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

FORCE=0
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

RESULTS="workspace/core4d_collab_retarget/results/E028b_anchor_refit"
MANIFEST="$RESULTS/manifest.tsv"
mkdir -p "$RESULTS"

.venv/bin/python workspace/core4d_collab_retarget/scripts/E028b/build_e028b_manifest.py \
  --manifest "$MANIFEST"

if [ "$FORCE" = "1" ]; then
  .venv/bin/python workspace/core4d_collab_retarget/scripts/E028/generate_e028_assets.py \
    --manifest "$MANIFEST" \
    --force
else
  .venv/bin/python workspace/core4d_collab_retarget/scripts/E028/generate_e028_assets.py \
    --manifest "$MANIFEST"
fi

.venv/bin/python workspace/core4d_collab_retarget/scripts/E028/generate_e028_overrides.py \
  --manifest "$MANIFEST" \
  --result-root "$RESULTS"

echo "E028b preprocess done."
