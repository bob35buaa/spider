#!/usr/bin/env bash
# E018b: generate 13-case canonical support-proxy anchor scenes and overrides.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

.venv/bin/python workspace/core4d_collab_retarget/scripts/E018b/generate_e018b_assets.py "$@"
.venv/bin/python workspace/core4d_collab_retarget/scripts/E018b/generate_e018b_overrides.py
