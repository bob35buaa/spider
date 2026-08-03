#!/usr/bin/env bash
# E187 A1 bucket007 2.5mm grid candidate; run preflight before smoke/formal.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-uv run python}"
$PYTHON_BIN workspace/core4d/scripts/experiments/E187/test_bucket007_grid_candidate.py
$PYTHON_BIN workspace/core4d/scripts/experiments/E187/bake_bucket007_grid_candidate.py "$@"
