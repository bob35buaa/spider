#!/usr/bin/env bash
# E145 fixed evaluation entry for high-priority full-CEM rows.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-full}"
shift || true

.venv/bin/python workspace/core4d/scripts/eval/eval_E145_full_nonbox_raw_mask_ref_fk.py \
  --stage "$STAGE" "$@"
