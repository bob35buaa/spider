#!/usr/bin/env bash
# E202: score every completed bucket aug full-CEM run with the public core
# evaluator; emit per-case orig(E178)-vs-aug deltas, per-object distribution,
# and the feasibility distribution (C5/C6).
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
exec .venv/bin/python workspace/core4d/scripts/eval/runners/eval_E202_bucket_augmentation.py "$@"
