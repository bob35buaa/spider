#!/usr/bin/env bash
# Evaluate E123 two-stage carry curriculum outputs.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-full}"
shift || true

if [ "$STAGE" != "smoke" ] && [ "$STAGE" != "full" ]; then
  echo "Invalid STAGE=$STAGE (use smoke|full)" >&2
  exit 2
fi

.venv/bin/python workspace/core4d/scripts/eval/eval_E123_two_stage_curriculum.py \
  --stage "$STAGE" "$@"
