#!/usr/bin/env bash
# Monitor E178 A100 and trigger its scoped pull/gates.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

export EXPERIMENT_ID=E178
export RESULT_ROOT=workspace/core4d/results/E178
export PULL_SCRIPT=workspace/core4d/scripts/launch/active/pull_E178_remote_a100_results.sh

exec bash workspace/core4d/scripts/launch/active/watch_E176_remote_a100.sh "$@"
