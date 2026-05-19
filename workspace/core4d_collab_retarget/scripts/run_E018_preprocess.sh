#!/usr/bin/env bash
# E018: generate canonical support-proxy anchor scenes and overrides.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

.venv/bin/python workspace/core4d_collab_retarget/scripts/E018/generate_e018_assets.py "$@"
.venv/bin/python workspace/core4d_collab_retarget/scripts/E018/generate_e018_overrides.py
