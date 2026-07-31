#!/usr/bin/env bash
# Run one E182 S1 case serially on one visible physical GPU.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
cd "$REPO_ROOT"

CASE_ID="${1:?usage: $0 CASE_ID PHYSICAL_GPU_ID}"
PHYSICAL_GPU_ID="${2:?usage: $0 CASE_ID PHYSICAL_GPU_ID}"
PYTHON_BIN="${E182_PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
RESULT_ROOT="${E182_QUERY_RESULT_ROOT:-workspace/core4d/results/E182/s1_query_tape}"
SCRIPT_ROOT="workspace/core4d/scripts/experiments/E182"

case "$PHYSICAL_GPU_ID" in
  0|1) ;;
  *)
    echo "E182 S1 worker only accepts physical GPU 0 or 1." >&2
    exit 2
    ;;
esac
case "$CASE_ID" in
  bucket003_20231018_001_p1|bucket004_20231002_021_p1|bucket007_20231020_055_p1) ;;
  *)
    echo "E182 S1 worker case is outside frozen dev3: $CASE_ID" >&2
    exit 2
    ;;
esac

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="$REPO_ROOT/.e182_deps:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

for mode in off on_a on_b; do
  runner_args=(
    "$SCRIPT_ROOT/run_query_tape_replay.py"
    "--mode" "$mode"
    "--case-id" "$CASE_ID"
    "--python-bin" "$PYTHON_BIN"
    "--gpu-id" "0"
    "--result-root" "$RESULT_ROOT"
  )
  CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU_ID" "$PYTHON_BIN" "${runner_args[@]}"
done

CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU_ID" "$PYTHON_BIN" \
  "$SCRIPT_ROOT/audit_query_tape_replay.py" \
  --case-id "$CASE_ID" \
  --result-root "$RESULT_ROOT"

echo "E182_QUERY_WORKER=PASS case=$CASE_ID physical_gpu=$PHYSICAL_GPU_ID"
