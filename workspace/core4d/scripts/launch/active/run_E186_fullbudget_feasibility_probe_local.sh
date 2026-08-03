#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
cd "${REPO_ROOT}"

CASE_ID="${1:?usage: $0 CASE_ID [GPU_ID]}"
GPU_ID="${2:-0}"

UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/e186_uv_cache}" \
  uv run python workspace/core4d/scripts/experiments/E186/run_fullbudget_feasibility_probe.py \
  --case-id "${CASE_ID}" --gpu-id "${GPU_ID}"
