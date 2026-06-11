#!/usr/bin/env bash
# Compatibility wrapper for E156 remote result pull.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
exec bash workspace/core4d/scripts/launch/active/pull_E156_remote_results.sh "$@"
