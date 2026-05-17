#!/usr/bin/env bash
# E003 preprocessing: create physics-sweep freejoint tasks and Hydra overrides.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

RESULTS="${RESULTS:-workspace/core4d_collab_retarget/results/E003}"
VARIANTS_FILE="${VARIANTS_FILE:-workspace/core4d_collab_retarget/scripts/E003/variants.tsv}"

mkdir -p "$RESULTS/contact_masks"

.venv/bin/python workspace/core4d_collab_retarget/scripts/E003/create_physics_sweep_cases.py \
  --variants "$VARIANTS_FILE" \
  --force

.venv/bin/python workspace/core4d_collab_retarget/scripts/E003/generate_e003_overrides.py \
  --variants "$VARIANTS_FILE" \
  --result-root "$RESULTS"

echo "=== E003 preprocess done: $RESULTS ==="
