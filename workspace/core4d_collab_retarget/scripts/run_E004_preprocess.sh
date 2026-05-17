#!/usr/bin/env bash
# E004 preprocessing: generate true-freejoint virtual-partner Hydra overrides.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

RESULTS="${RESULTS:-workspace/core4d_collab_retarget/results/E004}"
VARIANTS_FILE="${VARIANTS_FILE:-workspace/core4d_collab_retarget/scripts/E004/variants.tsv}"

mkdir -p "$RESULTS/contact_masks"

.venv/bin/python workspace/core4d_collab_retarget/scripts/E004/generate_e004_overrides.py \
  --variants "$VARIANTS_FILE" \
  --result-root "$RESULTS"

echo "=== E004 preprocess done: $RESULTS ==="
