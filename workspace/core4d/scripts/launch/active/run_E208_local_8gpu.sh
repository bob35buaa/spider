#!/usr/bin/env bash
# E208 (R294) · desk/chair object augmentation -> PRG full CEM, local 8 GPUs.
#
# 105 runs = 21 cases (22 minus the excluded chair005) x 5 aug variants
# (trans_0/1/2 + rot_0/1).  Budget is frozen at 1024 samples x 32 iterations,
# seed 0, use_torch_compile=false -- inherited from E206's admission decision,
# never re-typed here.
#
# Preconditions, all enforced by the runner itself (it exits rather than warns):
#   * results/E208/s6_downstream/cem/throughput/admission_decision.json  A1+A2 pass
#   * results/E208/s6_downstream/manifests/freeze.json                   set sha matches
#   * a flock single-instance lock -- the runner rewrites the whole manifest on
#     every status change, so two instances would revert each other
#
# GPU sharing: E207 (R293) and E209 (R295) run from other sessions on the same 8
# cards.  The memory admission waits for a free slot and never preempts, so
# coexisting is safe; it only stretches wall clock.  Check before starting:
#   nvidia-smi --query-gpu=index,memory.free --format=csv
#
# Usage:
#   DRY_RUN=1 bash workspace/core4d/scripts/launch/active/run_E208_local_8gpu.sh
#   bash workspace/core4d/scripts/launch/active/run_E208_local_8gpu.sh
#   GPUS=0,1,2,3 bash .../run_E208_local_8gpu.sh          # yield cards to E207/E209
#   CASES=desk023_20231030_019_p1 VARIANTS=trans0 bash ... # smoke one run
#   E208_FORCE=1 bash ...                                  # ignore existing outputs
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PY="${PY:-.venv/bin/python}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
PER_GPU_MEM_MIB="${PER_GPU_MEM_MIB:-5000}"
PER_TASK_TIMEOUT_MIN="${PER_TASK_TIMEOUT_MIN:-180}"
CASES="${CASES:-}"
VARIANTS="${VARIANTS:-}"
LIMIT="${LIMIT:-0}"

RUNNER=workspace/core4d/scripts/experiments/E208/run_e208_cem.py
LOG_DIR=logs/E208
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"

args=(
  --gpus "$GPUS"
  --per-gpu-mem-mib "$PER_GPU_MEM_MIB"
  --per-task-timeout-min "$PER_TASK_TIMEOUT_MIN"
)
[ -n "$CASES" ] && args+=(--cases "$CASES")
[ -n "$VARIANTS" ] && args+=(--variants "$VARIANTS")
[ "$LIMIT" != "0" ] && args+=(--limit "$LIMIT")

if [ "${DRY_RUN:-0}" = "1" ]; then
  exec "$PY" "$RUNNER" "${args[@]}" --dry-run
fi

echo "E208 R294 full CEM: gpus=$GPUS timeout=${PER_TASK_TIMEOUT_MIN}min"
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader || true
exec "$PY" "$RUNNER" "${args[@]}" 2>&1 | tee "$LOG_DIR/queue_${STAMP}.log"
