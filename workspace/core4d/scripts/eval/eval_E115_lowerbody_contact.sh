#!/usr/bin/env bash
# E115 fixed eval entrypoint.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-full}"
shift || true

python3 workspace/core4d/scripts/eval/eval_E115_lowerbody_contact.py --stage "$STAGE" "$@"
