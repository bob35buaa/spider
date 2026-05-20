#!/usr/bin/env bash
# E029: generate bucket001 stability / posture-valid contact overrides.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

mkdir -p workspace/core4d_collab_retarget/results/E029
.venv/bin/python workspace/core4d_collab_retarget/scripts/E029/generate_e029_overrides.py
echo "E029 preprocess done."
