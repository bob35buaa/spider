#!/usr/bin/env bash
# E199 full-scale (plan229): score every completed translation-augmented box
# full-CEM run with the public core evaluator, pair each against its same-case
# reused A0/PRG orig baseline (re-scored under the identical contract), and emit
# per-object strata + per-case deltas + translation feasibility distribution.
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
exec .venv/bin/python workspace/core4d/scripts/eval/runners/eval_E199_fullscale_augmentation.py "$@"
