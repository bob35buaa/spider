#!/usr/bin/env bash
# E008: generate COLA-style support proxy overrides.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

.venv/bin/python workspace/core4d_collab_retarget/scripts/E008/generate_e008_overrides.py "$@"
