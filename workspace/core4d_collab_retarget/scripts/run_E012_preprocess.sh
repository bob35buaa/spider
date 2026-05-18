#!/usr/bin/env bash
# E012: generate Hydra overrides for dual-point partner pose closure.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

.venv/bin/python workspace/core4d_collab_retarget/scripts/E012/generate_e012_overrides.py "$@"
