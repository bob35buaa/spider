#!/usr/bin/env bash
# E005 preprocessing: generate corrected partner-force support-site overrides.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

RESULTS="${RESULTS:-workspace/core4d_collab_retarget/results/E005}"
VARIANTS_FILE="${VARIANTS_FILE:-workspace/core4d_collab_retarget/scripts/E005/variants.tsv}"

mkdir -p "$RESULTS/contact_masks"

.venv/bin/python workspace/core4d_collab_retarget/scripts/E005/generate_e005_overrides.py \
  --variants "$VARIANTS_FILE" \
  --result-root "$RESULTS"

echo "=== E005 preprocess done: $RESULTS ==="
