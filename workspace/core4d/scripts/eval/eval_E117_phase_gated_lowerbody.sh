#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-full}"
shift || true

.venv/bin/python workspace/core4d/scripts/eval/eval_E117_phase_gated_lowerbody.py \
  --stage "$STAGE" "$@"
