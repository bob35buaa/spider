#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
cd "${REPO_ROOT}"

UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/e186_uv_cache}" \
  uv run python \
  workspace/core4d/scripts/experiments/E186/test_production_grid_backend.py
