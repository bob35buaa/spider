#!/usr/bin/env bash
# E144 fixed evaluation entry.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-full}"
shift || true

.venv/bin/python workspace/core4d/scripts/eval/eval_E144_raw_mask_ref_fk_full_cem.py \
  --stage "$STAGE" "$@"
