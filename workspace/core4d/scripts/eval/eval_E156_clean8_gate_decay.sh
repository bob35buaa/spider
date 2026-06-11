#!/usr/bin/env bash
# Compatibility wrapper for E156 clean8 gate/decay evaluation.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
exec bash workspace/core4d/scripts/eval/wrappers/eval_E156_clean8_gate_decay.sh "$@"
