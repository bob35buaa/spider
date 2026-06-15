#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$ROOT"

STAGE="${1:-full}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" workspace/core4d/scripts/experiments/E162/build_post_e147_reeval_manifest.py
"$PYTHON_BIN" workspace/core4d/scripts/eval/runners/eval_E162_post_e147_rl_safe_reeval.py "$STAGE" --allow-missing
