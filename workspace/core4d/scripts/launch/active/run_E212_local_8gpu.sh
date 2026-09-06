#!/usr/bin/env bash
# E212 Stage A -- desk023 partial-gravcomp sweep (g = 0.4 / 0.6 / 0.8) on 8 GPUs.
#
# Reuses the E199 manifest-driven priority queue (all run params live in the
# manifest rows). Coexists with other GPU jobs: only dispatches when free mem
# >= PER_GPU_MEM_MIB; never kills/preempts foreign processes. Resume-safe for
# *finished* work.
#
# Must be the E199 queue, NOT E200's: e200_common.TIER_RANK is {"P1": 1} and the
# E212 manifest is tier P0, which would KeyError on dispatch.
#
# TWO-MACHINE SPLIT (SHARD=A|B)
#   The queue rewrites the WHOLE manifest TSV on every status change, so two
#   machines sharing this /mnt must never point at the same file -- the second
#   writer would clobber the first's status from its own stale snapshot. Each
#   shard is a disjoint manifest; A is 8 rows (local), B is 4 (remote). The
#   split is a Latin-square diagonal (e212_common.shard_of), so both shards span
#   all 4 cases and all 3 g values and a dead machine still leaves an
#   interpretable cross-section.
#
#   Only shard A snapshots the scenes (rules/experiment.md §7). Shard B must run
#   with SKIP_SNAPSHOT=1; the script then ASSERTS the snapshot exists and was
#   taken at the current git HEAD, so B can never start against unfrozen or
#   since-modified scenes.
#
# Usage:
#   local  (8 rows):  SHARD=A bash workspace/core4d/scripts/launch/active/run_E212_local_8gpu.sh
#   remote (4 rows):  SHARD=B SKIP_SNAPSHOT=1 bash workspace/core4d/scripts/launch/active/run_E212_local_8gpu.sh
#   dry run:          SHARD=B DRY_RUN=1 bash .../run_E212_local_8gpu.sh
#   smoke:            STAGE=smoke SHARD=A GPUS=0 bash .../run_E212_local_8gpu.sh
#
# Env overrides: STAGE, SHARD, GPUS, PER_GPU_MEM_MIB, MAX_PER_GPU, POLL_INTERVAL,
#                DRY_RUN, SKIP_SNAPSHOT
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

STAGE="${STAGE:-full}"
SHARD="${SHARD:-A}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
PER_GPU_MEM_MIB="${PER_GPU_MEM_MIB:-5000}"
MAX_PER_GPU="${MAX_PER_GPU:-1}"
POLL_INTERVAL="${POLL_INTERVAL:-15}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_SNAPSHOT="${SKIP_SNAPSHOT:-0}"
PY=".venv/bin/python"
Q="workspace/core4d/scripts/experiments/E199/run_local_priority_queue.py"
MANIFEST_DIR="workspace/core4d/results/E212/s6_downstream/manifests"
SNAPSHOT_MANIFEST="workspace/core4d/results/E212/scene_snapshot/manifest.txt"

case "$STAGE" in
  full|smoke) ;;
  *) echo "unknown STAGE=$STAGE (want full|smoke)" >&2; exit 2 ;;
esac
case "$SHARD" in
  A|B) ;;
  *) echo "unknown SHARD=$SHARD (want A|B)" >&2; exit 2 ;;
esac

MANIFEST="$MANIFEST_DIR/e212_stageA_${STAGE}_manifest.shard${SHARD}.tsv"
[[ -f "$MANIFEST" ]] || {
  echo "missing shard manifest: $MANIFEST (run E212/build_manifest.py --stage $STAGE)" >&2
  exit 2
}

# rules/experiment.md §7 Safeguard 2: freeze the scenes before spending GPU.
if [[ "$SKIP_SNAPSHOT" != "1" && "$DRY_RUN" != "1" ]]; then
  bash workspace/core4d/scripts/experiments/E212/snapshot_E212_scenes.sh
elif [[ "$DRY_RUN" != "1" ]]; then
  # Skipping is only legitimate when the OTHER machine already froze the scenes
  # at this exact commit. Anything else and we would be running against scenes
  # nobody recorded.
  [[ -f "$SNAPSHOT_MANIFEST" ]] || {
    echo "SKIP_SNAPSHOT=1 but $SNAPSHOT_MANIFEST is absent -- run shard A first" >&2
    exit 2
  }
  # snapshot_scenes.sh writes "# Git HEAD: <short-sha> (<branch>)". Match that
  # shape explicitly -- a pattern that silently matches nothing is a guard that
  # never fires, which is worse than no guard at all.
  SNAP_HEAD="$(sed -n 's/^# Git HEAD: \([0-9a-f]\{7,40\}\).*/\1/p' "$SNAPSHOT_MANIFEST" | head -1)"
  [[ -n "$SNAP_HEAD" ]] || {
    echo "cannot parse '# Git HEAD:' out of $SNAPSHOT_MANIFEST -- refusing to run blind" >&2
    exit 2
  }
  HEAD_SHA="$(git rev-parse --short="${#SNAP_HEAD}" HEAD)"
  if [[ "$SNAP_HEAD" != "$HEAD_SHA" ]]; then
    echo "snapshot was taken at $SNAP_HEAD but HEAD is $HEAD_SHA -- scenes may have moved" >&2
    exit 2
  fi
  echo "[run_E212] snapshot check OK (HEAD=$HEAD_SHA)"
fi

# Stamp the machine into the rows this instance owns, before dispatch. A
# two-machine run must be able to answer "where did this row run" after the
# fact; the queue preserves unknown columns, so a pre-write survives.
if [[ "$DRY_RUN" != "1" ]]; then
  "$PY" - "$MANIFEST" "$(hostname)" <<'PY'
import csv, sys
from pathlib import Path
path, host = Path(sys.argv[1]), sys.argv[2]
with path.open(encoding="utf-8", newline="") as stream:
    reader = csv.DictReader(stream, delimiter="\t")
    fields, rows = list(reader.fieldnames), list(reader)
for row in rows:
    row["host"] = host
with path.open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
print(f"[run_E212] stamped host={host} on {len(rows)} rows")
PY
fi

# triton JIT needs python3.12-dev, which is not guaranteed on this host.
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
# CEM is headless; osmesa is only needed by the renderer.
export MUJOCO_GL="${MUJOCO_GL:-disable}"

EXTRA=""
[[ "$DRY_RUN" == "1" ]] && EXTRA="--dry-run"

echo "[run_E212] STAGE=$STAGE SHARD=$SHARD GPUS=$GPUS PER_GPU_MEM_MIB=$PER_GPU_MEM_MIB MAX_PER_GPU=$MAX_PER_GPU"
echo "[run_E212] manifest=$MANIFEST $EXTRA"
exec "$PY" "$Q" --manifest "$MANIFEST" --gpus "$GPUS" \
  --per-gpu-mem-mib "$PER_GPU_MEM_MIB" --max-per-gpu "$MAX_PER_GPU" \
  --poll-interval "$POLL_INTERVAL" $EXTRA
