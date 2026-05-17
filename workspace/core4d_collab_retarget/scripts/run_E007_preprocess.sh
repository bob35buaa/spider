#!/usr/bin/env bash
# E007: generate COLA-style support proxy overrides.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

.venv/bin/python workspace/core4d_collab_retarget/scripts/E007/generate_e007_overrides.py "$@"
