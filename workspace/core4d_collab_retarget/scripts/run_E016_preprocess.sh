#!/usr/bin/env bash
# E016: generate freejoint derived tasks, support-weld scenes and overrides.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

.venv/bin/python workspace/core4d_collab_retarget/scripts/E016/generate_e016_assets.py "$@"
.venv/bin/python workspace/core4d_collab_retarget/scripts/E016/generate_e016_overrides.py
