#!/usr/bin/env bash
# E178 A100 launcher: fixed user-authorized GPUs 2,3,6,7.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

export EXPERIMENT_ID=E178
export RESULT_ROOT=workspace/core4d/results/E178
export CANARY_MANIFEST_BASENAME=semantic_bucket_canary_manifest.tsv
export FULL_MANIFEST_BASENAME=semantic_bucket_full_manifest.tsv
export EXPECTED_CANARY_ROWS=3
export EXPECTED_FULL_ROWS=27
export E176_FIXED_GPUS=1
export A100_POLICY_GPUS="${A100_POLICY_GPUS:-2,3,6,7}"

exec bash workspace/core4d/scripts/launch/active/run_E176_remote_a100.sh "$@"
