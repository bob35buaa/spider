#!/usr/bin/env bash
# E028: generate hard penetration overrides.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

mkdir -p workspace/core4d_collab_retarget/results/E028
.venv/bin/python workspace/core4d_collab_retarget/scripts/E028/generate_e028_overrides.py
echo "E028 preprocess done."
