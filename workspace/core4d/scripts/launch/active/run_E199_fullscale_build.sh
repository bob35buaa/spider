#!/usr/bin/env bash
# E199 full-scale (plan229) DATA BUILD: build translation-augmented SPIDER tasks
# for all s6-full-CEM box cases, sharded across N parallel CPU workers.
#
# Each shard processes a disjoint set of case_ids and writes its OWN artifacts
# TSV (no cross-shard race: aug task dirs, converted/retargeted dirs, scene
# snapshots are all keyed by unique case_id). After all shards finish, their
# artifact rows are merged into the canonical FULLSCALE_ARTIFACTS, then the CEM
# override + priority manifest is built and the scene snapshot manifest written.
#
# This is CPU-only (holosoma IK + MuJoCo kinematic trajectory); it does NOT use
# GPUs. Launch the CEM queue afterwards with:
#   SCOPE=box_fullscale bash workspace/core4d/scripts/launch/active/run_E199_local_8gpu.sh
#
# Usage:
#   bash workspace/core4d/scripts/launch/active/run_E199_fullscale_build.sh
# Env overrides: SHARDS (default 6)
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

PY=".venv/bin/python"
SHARDS="${SHARDS:-6}"
DRIVER="workspace/core4d/scripts/experiments/E199/build_augmented_tasks.py"
MANIFEST_PY="workspace/core4d/scripts/experiments/E199/build_aug_manifest.py"
SHARD_DIR="workspace/core4d/results/E199/data_preprocess/manifests/shards"
LOG_DIR="logs/E199/fullscale"
SNAP_DIR="workspace/core4d/results/E199/scene_snapshot"
mkdir -p "$SHARD_DIR" "$LOG_DIR" "$SNAP_DIR"

echo "[fullscale-build] computing $SHARDS balanced case shards"
mapfile -t SHARD_CSVS < <("$PY" - "$SHARDS" <<'PYEOF'
import sys
sys.path.insert(0, 'workspace/core4d/scripts/experiments/E199')
import e199_common as C
n = int(sys.argv[1])
cases = sorted(c["case_id"] for c in C.load_fullscale_cases())
buckets = [[] for _ in range(n)]
for i, cid in enumerate(cases):
    buckets[i % n].append(cid)
for b in buckets:
    print(",".join(b))
PYEOF
)

pids=()
for i in "${!SHARD_CSVS[@]}"; do
  csv="${SHARD_CSVS[$i]}"
  [[ -z "$csv" ]] && continue
  af="$SHARD_DIR/shard_${i}.tsv"
  nohup "$PY" "$DRIVER" --scope box_fullscale --cases "$csv" --artifacts "$af" \
    > "$LOG_DIR/build_shard_${i}.log" 2>&1 &
  pid=$!
  pids+=("$pid")
  echo "[shard $i] pid=$pid ncases=$(tr ',' ' ' <<<"$csv" | wc -w) -> $af"
done

echo "[fullscale-build] waiting for ${#pids[@]} shards ..."
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "[fullscale-build] all shards finished (fail=$fail)"

echo "[fullscale-build] merging shard artifacts -> canonical"
"$PY" - <<'PYEOF'
import glob, sys
sys.path.insert(0, 'workspace/core4d/scripts/experiments/E199')
import e199_common as C
rows, seen = [], set()
for f in sorted(glob.glob(str(C.RESULTS / 'data_preprocess/manifests/shards/shard_*.tsv'))):
    for r in C.read_tsv(f):
        key = r.get('target_task')
        if key in seen:
            continue
        seen.add(key)
        rows.append(r)
rows.sort(key=lambda r: (r.get('object_key', ''), r.get('case_id', ''), r.get('aug_variant', '')))
C.write_tsv(C.FULLSCALE_ARTIFACTS, rows)
print(f"[merge] {len(rows)} variant rows across {len({r['case_id'] for r in rows})} cases -> {C.rel(C.FULLSCALE_ARTIFACTS)}")
PYEOF

echo "[fullscale-build] building CEM overrides + priority manifest"
"$PY" "$MANIFEST_PY" --scope box_fullscale

echo "[fullscale-build] snapshot manifest (git HEAD + sha256)"
{
  echo "# E199 full-scale scene snapshot manifest"
  echo "git_head: $(git rev-parse HEAD)"
  echo "created_at: $(date -Iseconds)"
  echo ""
  find "$SNAP_DIR/cem_sidecars" -name '*.xml' 2>/dev/null | sort | while read -r f; do
    echo "$(sha256sum "$f" | cut -d' ' -f1)  ${f#workspace/}"
  done
} > "$SNAP_DIR/manifest.txt"

echo "[fullscale-build] done. launch CEM with:"
echo "  SCOPE=box_fullscale GPUS=0,1,2,3,4,5,6,7 bash workspace/core4d/scripts/launch/active/run_E199_local_8gpu.sh"
