#!/usr/bin/env bash
# E216 chair/desk (Phase 2) full CEM -- one SHARD of the frozen 59-row queue.
# 3-machine run over the shared /mnt filesystem: each machine runs a disjoint
# shard (A/B/C), round-robin over ordinal so each gets a balanced tier slice.
# run_E216cd_cem rewrites only its own shard TSV + takes a per-shard flock, and
# validates its rows are a SUBSET of the frozen set.  Output paths are keyed by
# variant_id (unique per case+variant) so rollouts/logs never collide.
#
# IMPORTANT: kill this machine's GPU keepalive first (frees the 8 GPUs); the
# scheduler only dispatches when free mem >= PER_GPU_MEM_MIB.
#
# Usage (per machine):
#   SHARD=A bash workspace/core4d/scripts/experiments/E216/run_E216cd_shard.sh   # local
#   SHARD=B bash .../run_E216cd_shard.sh                                         # remote 1
#   SHARD=C bash .../run_E216cd_shard.sh                                         # remote 2
# Env: SHARD (A/B/C, required), GPUS, PER_GPU_MEM_MIB, PER_TASK_TIMEOUT_MIN, DRY_RUN
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

SHARD="${SHARD:?set SHARD=A|B|C}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
PER_GPU_MEM_MIB="${PER_GPU_MEM_MIB:-5000}"
PER_TASK_TIMEOUT_MIN="${PER_TASK_TIMEOUT_MIN:-180}"
DRY_RUN="${DRY_RUN:-0}"
PY=".venv/bin/python"
Q="workspace/core4d/scripts/experiments/E216/run_E216cd_cem.py"
MAN="workspace/core4d/results/E216/chair_desk/s6_downstream/manifests/e216cd_priority_manifest.shard${SHARD}.tsv"

[[ -f "$MAN" ]] || { echo "ERROR: shard manifest not found: $MAN" >&2; exit 1; }
EXTRA=""; [[ "$DRY_RUN" == "1" ]] && EXTRA="--dry-run"

echo "[run_E216cd shard$SHARD] GPUS=$GPUS mem>=$PER_GPU_MEM_MIB timeout=${PER_TASK_TIMEOUT_MIN}min  $MAN"
exec "$PY" "$Q" --manifest "$MAN" --gpus "$GPUS" --per-gpu-mem-mib "$PER_GPU_MEM_MIB" \
  --per-task-timeout-min "$PER_TASK_TIMEOUT_MIN" $EXTRA
