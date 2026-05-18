#!/usr/bin/env bash
# E010: generate contact-pad scenes and Hydra overrides.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

.venv/bin/python workspace/core4d_collab_retarget/scripts/E010/generate_contact_pad_scenes.py
.venv/bin/python workspace/core4d_collab_retarget/scripts/E010/generate_e010_overrides.py "$@"
