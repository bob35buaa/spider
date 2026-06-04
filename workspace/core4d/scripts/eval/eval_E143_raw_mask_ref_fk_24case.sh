#!/usr/bin/env bash
# E143 fixed evaluation entry.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-full}"
shift || true

.venv/bin/python workspace/core4d/scripts/eval/eval_E143_raw_mask_ref_fk_24case.py \
  --stage "$STAGE" "$@"
