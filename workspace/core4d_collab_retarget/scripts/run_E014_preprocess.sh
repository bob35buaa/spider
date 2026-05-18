#!/usr/bin/env bash
# E014: generate scene XMLs and Hydra overrides for COLA-B soft weld.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

.venv/bin/python workspace/core4d_collab_retarget/scripts/E014/generate_e014_scenes.py "$@"
.venv/bin/python workspace/core4d_collab_retarget/scripts/E014/generate_e014_overrides.py "$@"
