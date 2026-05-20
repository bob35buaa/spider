#!/usr/bin/env bash
# E030: generate lower-body geometry / surface-control overrides.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

mkdir -p workspace/core4d_collab_retarget/results/E030
.venv/bin/python workspace/core4d_collab_retarget/scripts/E030/generate_e030_assets.py --force
echo "E030 preprocess done."
