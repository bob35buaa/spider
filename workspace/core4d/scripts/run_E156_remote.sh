#!/usr/bin/env bash
# Compatibility wrapper for E156 remote launcher.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
exec bash workspace/core4d/scripts/launch/active/run_E156_remote.sh "$@"
