#!/usr/bin/env bash
# Fixed entry for E147 rubber hand collision A/B eval.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-full}"
shift || true

.venv/bin/python workspace/core4d/scripts/eval/eval_E147_rubber_hand_collision.py --stage "$STAGE" "$@"
