#!/usr/bin/env bash
# E013: generate Hydra overrides for true-freejoint object oracle.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

.venv/bin/python workspace/core4d_collab_retarget/scripts/E013/generate_e013_overrides.py "$@"
