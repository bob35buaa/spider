#!/usr/bin/env bash
# E009: generate COLA-style support proxy overrides.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

.venv/bin/python workspace/core4d_collab_retarget/scripts/E009/generate_e009_overrides.py "$@"
