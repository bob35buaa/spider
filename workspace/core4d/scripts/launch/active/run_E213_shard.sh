#!/bin/bash
# E213 Phase A: selected-arm object-augmentation Full CEM, one shard per machine.
#
# Four 8-GPU machines share this /mnt + .venv.  Each machine runs ONE shard:
#   A = local (Claude)   B/C/D = the three remote 8-GPU boxes (user launches each).
# Each machine reads/writes only its own shard manifest; merge_shards.py reconciles.
#
# Usage (on each machine, from repo root):
#   bash workspace/core4d/scripts/launch/active/run_E213_shard.sh A
#   GPUS=0,1,2,3,4,5,6,7 bash .../run_E213_shard.sh B
#
# Env overrides: GPUS (default all 8), DRY_RUN=1, E213_FORCE=1, LIMIT=N.
set -euo pipefail

SHARD="${1:?usage: run_E213_shard.sh <A|B|C|D>}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$REPO"

PY="${PY:-$REPO/.venv/bin/python}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
RUNNER="workspace/core4d/scripts/experiments/E213/run_source_cem.py"

args=(--shard "$SHARD" --gpus "$GPUS")
[[ "${DRY_RUN:-0}" == "1" ]] && args+=(--dry-run)
[[ -n "${LIMIT:-}" ]] && args+=(--limit "$LIMIT")
[[ -n "${CASES:-}" ]] && args+=(--cases "$CASES")

echo "[E213] host=$(hostname) shard=$SHARD gpus=$GPUS"
# Rule 7 snapshot: only shard A (local) writes it; B/C/D share /mnt and see it.
if [[ "$SHARD" == "A" && "${DRY_RUN:-0}" != "1" && "${SKIP_SNAPSHOT:-0}" != "1" ]]; then
  echo "[E213] snapshot scenes before first CEM (rule 7)..."
  bash workspace/core4d/scripts/experiments/E213/snapshot_E213_scenes.sh || \
    echo "[E213] WARN snapshot step failed (continuing)"
fi
exec "$PY" "$RUNNER" "${args[@]}"
