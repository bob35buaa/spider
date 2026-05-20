#!/usr/bin/env bash
# E024: generate bucket001 stability repair overrides.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

mkdir -p workspace/core4d_collab_retarget/results/E024
.venv/bin/python workspace/core4d_collab_retarget/scripts/E024/generate_e024_overrides.py
echo "E024 preprocess done."
