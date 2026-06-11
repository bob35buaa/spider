#!/usr/bin/env bash
# E116 fixed eval entrypoint.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-full}"
shift || true

python3 workspace/core4d/scripts/eval/eval_E116_surface_target_upright.py --stage "$STAGE" "$@"
