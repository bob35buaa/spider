#!/usr/bin/env bash
# Compatibility wrapper for E157 gateA RL export.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
exec bash workspace/core4d/scripts/launch/active/run_E157_gateA_rl_export.sh "$@"
