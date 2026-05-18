#!/usr/bin/env bash
# E015: generate dynamic-support scenes, augmented data, and Hydra overrides.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

.venv/bin/python workspace/core4d_collab_retarget/scripts/E015/generate_e015_assets.py "$@"
.venv/bin/python workspace/core4d_collab_retarget/scripts/E015/generate_e015_overrides.py "$@"
