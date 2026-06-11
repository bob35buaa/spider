#!/usr/bin/env bash
# Fixed entry for E148 E143-24 rubber hand collision comparison eval.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-full}"
shift || true

.venv/bin/python workspace/core4d/scripts/eval/eval_E148_e143_rubber_hand_collision.py --stage "$STAGE" "$@"
