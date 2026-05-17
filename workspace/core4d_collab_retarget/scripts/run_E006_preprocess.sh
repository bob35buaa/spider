#!/usr/bin/env bash
# E006: generate COLA-style support proxy overrides.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

.venv/bin/python workspace/core4d_collab_retarget/scripts/E006/generate_e006_overrides.py "$@"
