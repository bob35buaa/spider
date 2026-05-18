#!/usr/bin/env bash
# E011: generate Hydra overrides for soft object tether diagnostics.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

.venv/bin/python workspace/core4d_collab_retarget/scripts/E011/generate_e011_overrides.py "$@"
