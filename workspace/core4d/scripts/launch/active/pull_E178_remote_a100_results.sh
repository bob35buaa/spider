#!/usr/bin/env bash
# Pull and validate only artifacts registered by E178.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

export EXPERIMENT_ID=E178
export RESULT_ROOT=workspace/core4d/results/E178
export CANARY_MANIFEST_BASENAME=semantic_bucket_canary_manifest.tsv
export FULL_MANIFEST_BASENAME=semantic_bucket_full_manifest.tsv
export EXPECTED_CANARY_ROWS=3
export EXPECTED_FULL_ROWS=27
export EVALUATOR=workspace/core4d/scripts/eval/runners/eval_E178_lowgeom.py

exec bash \
  workspace/core4d/scripts/launch/active/pull_E176_remote_a100_results.sh \
  "$@"
