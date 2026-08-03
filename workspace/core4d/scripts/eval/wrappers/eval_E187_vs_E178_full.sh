#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$repo_root"

mode="${1:-run}"
builder="workspace/core4d/scripts/experiments/E187/build_evaluation_manifest.py"
test_file="workspace/core4d/scripts/experiments/E187/test_evaluation_manifest.py"
runner="workspace/core4d/scripts/eval/runners/eval_E187_vs_E178_full.py"

python_bin="${PYTHON_BIN:-uv run python}"
$python_bin "$test_file"

case "$mode" in
  preflight)
    $python_bin "$builder" --check
    ;;
  run)
    $python_bin "$builder"
    $python_bin "$runner"
    ;;
  *)
    echo "usage: $0 {preflight,run}" >&2
    exit 2
    ;;
esac
