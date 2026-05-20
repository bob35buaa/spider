#!/usr/bin/env bash
# E025: generate contact/collision repair overrides.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

mkdir -p workspace/core4d_collab_retarget/results/E025
.venv/bin/python workspace/core4d_collab_retarget/scripts/E025/generate_e025_overrides.py
echo "E025 preprocess done."
