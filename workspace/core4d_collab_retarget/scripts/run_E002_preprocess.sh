#!/usr/bin/env bash
# E002 preprocessing: create freejoint derived tasks and Hydra overrides.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

RESULTS="${RESULTS:-workspace/core4d_collab_retarget/results/E002}"
VARIANTS_FILE="${VARIANTS_FILE:-workspace/core4d_collab_retarget/scripts/E002/variants.tsv}"

mkdir -p "$RESULTS/contact_masks"

.venv/bin/python workspace/core4d_collab_retarget/scripts/E002/create_freejoint_legobj_cases.py \
  --variants "$VARIANTS_FILE" \
  --force

.venv/bin/python workspace/core4d_collab_retarget/scripts/E002/generate_e002_overrides.py \
  --variants "$VARIANTS_FILE" \
  --result-root "$RESULTS"

echo "=== E002 preprocess done: $RESULTS ==="
