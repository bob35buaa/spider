#!/usr/bin/env bash
# E199: score every completed augmented full-CEM run with the public core
# evaluator and emit orig-vs-aug deltas + full distribution (C4).
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
exec .venv/bin/python workspace/core4d/scripts/eval/runners/eval_E199_augmentation.py "$@"
