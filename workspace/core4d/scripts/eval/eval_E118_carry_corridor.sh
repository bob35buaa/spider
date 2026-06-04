#!/usr/bin/env bash
# Fixed entrypoint for E118 carry-corridor evaluation.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-smoke}"
shift || true

python workspace/core4d/scripts/eval/eval_E118_carry_corridor.py --stage "$STAGE" "$@"
