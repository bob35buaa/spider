#!/usr/bin/env bash
# E216 data build: per-object-arm trans 0.1m augmentation -> SPIDER tasks -> frozen CEM queue.
#
# Stages (idempotent; safe to re-run):
#   1. seed_warmstart        copy each case's `_original` (+converted+trim_window) into the
#                            E216 tree.  trans_* are NOT seeded -> upstream recomputes at 0.1m.
#   2. run_upstream_retarget drive pipeline.sh aug branch with OMNIRT_TRANS_OFFSET_M=0.1
#                            (patched holosoma 3fa6d90); produces trans_0/1/2 npz.  GPU (IK).
#   3. build_augmented_tasks fixed-window trim + SPIDER task + per-object arm CEM scene.
#   4. build_gravcomp_sidecars  re-verify the object-gravcomp sidecars (box_prg_g1a2).
#   5. build_aug_manifest    overrides + single-variable compose audit + (optional) --freeze.
#   6. snapshot_E216_scenes  rules/experiment.md §7 scene snapshot (before CEM).
#
# Usage:
#   bash workspace/core4d/scripts/train/train_E216.sh                 # all 104 box+bucket
#   OBJECTS=box021 bash .../train_E216.sh                             # subset / canary
#   FREEZE=1 bash .../train_E216.sh                                   # freeze the manifest
# Env: OBJECTS, CASES, RETARGET_MAX_WORKERS (object groups in parallel), FREEZE
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

PY=".venv/bin/python"
E="workspace/core4d/scripts/experiments/E216"
OBJECTS="${OBJECTS:-}"
CASES="${CASES:-}"
FREEZE="${FREEZE:-0}"
RMW="${RETARGET_MAX_WORKERS:-5}"

SEL=""
[[ -n "$OBJECTS" ]] && SEL="$SEL --objects $OBJECTS"
[[ -n "$CASES" ]] && SEL="$SEL --cases $CASES"

echo "=== [E216 S1] seed warm-starts ($SEL) ==="
$PY "$E/seed_warmstart.py" $SEL

echo "=== [E216 S2a] upstream omnirt_v2 @0.1m retarget ($SEL, max-workers=$RMW) ==="
E216_RETARGET_MAX_WORKERS="$RMW" $PY "$E/run_upstream_retarget.py" $SEL --max-workers "$RMW"

echo "=== [E216 S2b] build SPIDER tasks + per-object arm scenes ($SEL) ==="
# build returns non-zero if ANY variant is a legit geometric exclusion
# (runtime_initial_overlap); that must not abort the pipeline -- the manifest
# step consumes only the "built" rows.  Tolerate, the summary records failures.
$PY "$E/build_augmented_tasks.py" $SEL || echo "WARN: build_augmented_tasks reported excluded variants (see artifacts tsv)"

echo "=== [E216 S2c] verify object-gravcomp sidecars ==="
$PY "$E/build_gravcomp_sidecars.py" || true

echo "=== [E216 S2d] build overrides + single-variable audit (FREEZE=$FREEZE) ==="
if [[ "$FREEZE" == "1" ]]; then
  $PY "$E/build_aug_manifest.py" --freeze
else
  $PY "$E/build_aug_manifest.py"
fi

echo "=== [E216 S2e] scene snapshot (rules/experiment.md §7) ==="
bash "$E/snapshot_E216_scenes.sh" || echo "WARN: snapshot skipped (no built tasks yet?)"

echo "=== [E216] data build done.  Next: run_E216_local_8gpu.sh ==="
