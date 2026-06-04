#!/usr/bin/env bash
# Fixed E112 evaluation entry.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-full}"
shift || true

.venv/bin/python workspace/core4d/scripts/eval/eval_E112_contact_aware_cem.py --stage "$STAGE" "$@"
