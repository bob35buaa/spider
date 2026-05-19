#!/usr/bin/env bash
# E017: audit anchors, generate selected support-weld scenes and overrides.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

.venv/bin/python workspace/core4d_collab_retarget/scripts/E017/audit_select_anchors.py --write "$@"
.venv/bin/python workspace/core4d_collab_retarget/scripts/E017/generate_e017_overrides.py
