#!/usr/bin/env bash
# Fixed entry for E150 contact-anchor eef_offset sweep eval.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-full}"
shift || true

.venv/bin/python workspace/core4d/scripts/eval/eval_E150_eef_offset_sweep.py --stage "$STAGE" "$@"
