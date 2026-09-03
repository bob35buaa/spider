#!/usr/bin/env bash
# E206 Stage2b parallel retarget driver (adapted from run_E173_stage2b_parallel.sh).
#
# The shared run_stage2b_queue.py runs cases SERIALLY. E206 has 65 in-scope cases
# and this host has 192 cores; a serial run wastes hours. Cases are independent
# and deterministic (fixed OmniRetarget seed + SPIDER preprocess), so sharding
# the manifest across N runners changes only wall-clock, not results.
#
# Retargeting is CPU-only (CVXPY/IK -- no CUDA anywhere in pipeline.sh or
# robot_retarget.py), so this is safe to run while the GPUs are contended.
#
# Each shard runner writes per-case outputs keyed by case_id (no cross-shard
# collision) and persists status into its own shard manifest; shard rows are then
# merged back into the canonical manifest for downstream S4/S5.
#
# Usage:
#   VARIANT=omnirt_v1 NSHARDS=24 bash .../run_E206_stage2b_parallel.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

export HOLOSOMA_DEPS_DIR="${HOLOSOMA_DEPS_DIR:-/mnt/ali-sh-1/dataset/zeus/xiayb/.holosoma_deps}"
CORE4D_RAW_ROOT="${CORE4D_RAW_ROOT:-/mnt/ali-sh-1/usr/xiayibo/xyb_data_tidal_alsh/other-datasets/mocap_data/CORE4D/CORE4D_Real}"
SMPLX_MODEL_DIR="${SMPLX_MODEL_DIR:-/mnt/ali-sh-1/usr/xiayibo/xyb_data_tidal_alsh/other-datasets/mocap_data/human_model_files}"
RETARGET_PYTHON_BIN="${RETARGET_PYTHON_BIN:-$HOLOSOMA_DEPS_DIR/miniconda3/envs/hsretargeting/bin/python}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
HOLOSOMA_REPO="${HOLOSOMA_REPO:-../holosoma}"
VARIANT="${VARIANT:-omnirt_v1}"
NSHARDS="${NSHARDS:-24}"
# KEEP_GOING=1 isolates each case: an OmniRetarget CVXPY infeasible is logged and
# the batch continues (pipeline.sh:19-22). Without it one bad case aborts a shard.
export KEEP_GOING="${KEEP_GOING:-1}"

E=workspace/core4d/results/E206
VDIR="$E/s3_retarget/$VARIANT/ref_fk"
MANIFEST="$VDIR/stage2b_manifest_${VARIANT}_ref_fk.tsv"
SHARD_ROOT="$VDIR/shards"
QUEUE=workspace/core4d/scripts/experiments/E174/run_stage2b_queue.py
[ -f "$MANIFEST" ] || { echo "manifest missing: $MANIFEST" >&2; exit 2; }
mkdir -p "$SHARD_ROOT" "logs/E206/stage2b/$VARIANT"

# --- shard eligible rows round-robin into NSHARDS manifests -------------------
echo "=== sharding $MANIFEST into $NSHARDS shards (eligible-only) ==="
$PYTHON_BIN - "$MANIFEST" "$SHARD_ROOT" "$NSHARDS" <<'PYEOF'
import csv, sys
from pathlib import Path
manifest, shard_root, n = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
rows = list(csv.DictReader(open(manifest), delimiter="\t"))
fields = list(rows[0].keys())
elig = [r for r in rows if r.get("pipeline_enabled") == "1"
        and r.get("stage2b_status") not in ("pass", "omniretarget_infeasible")]
n = min(n, max(1, len(elig)))
buckets = [[] for _ in range(n)]
for i, r in enumerate(elig):
    buckets[i % n].append(r)
made = 0
for i, b in enumerate(buckets):
    if not b:
        continue
    d = shard_root / f"shard{i:02d}"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"stage2b_manifest_shard{i:02d}.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        w.writeheader(); w.writerows(b)
    made += 1
print(f"eligible={len(elig)} shards_made={made}")
PYEOF

# --- launch one queue runner per shard ---------------------------------------
echo "=== launching parallel shard runners ==="
pids=()
for sd in "$SHARD_ROOT"/shard*/; do
  sm=$(ls "$sd"stage2b_manifest_shard*.tsv 2>/dev/null | head -1)
  [ -f "$sm" ] || continue
  sid=$(basename "$sd")
  nohup $PYTHON_BIN "$QUEUE" --manifest-tsv "$sm" \
    --holosoma-repo "$HOLOSOMA_REPO" --core4d-raw-root "$CORE4D_RAW_ROOT" \
    --smplx-model-dir "$SMPLX_MODEL_DIR" --retarget-python-bin "$RETARGET_PYTHON_BIN" \
    > "logs/E206/stage2b/$VARIANT/${sid}.log" 2>&1 &
  pids+=($!)
done
echo "launched ${#pids[@]} shard runners: ${pids[*]}"

# --- wait for all ------------------------------------------------------------
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "=== all shard runners done (nonzero_exits=$fail) ==="

# --- merge shard statuses back into canonical manifest -----------------------
echo "=== merging shard rows into $MANIFEST ==="
$PYTHON_BIN - "$MANIFEST" "$SHARD_ROOT" <<'PYEOF'
import collections, csv, sys
from pathlib import Path
manifest, shard_root = sys.argv[1], Path(sys.argv[2])
rows = list(csv.DictReader(open(manifest), delimiter="\t"))
fields = list(rows[0].keys())
by_id = {r["case_id"]: r for r in rows}
merged = 0
for sm in sorted(shard_root.glob("shard*/stage2b_manifest_shard*.tsv")):
    for r in csv.DictReader(open(sm), delimiter="\t"):
        cid = r["case_id"]
        if cid in by_id:
            for k in fields:
                if r.get(k, "") != "":
                    by_id[cid][k] = r[k]
            merged += 1
with open(manifest, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
    w.writeheader(); w.writerows(rows)
print(f"merged_rows={merged} "
      f"status={dict(collections.Counter(r.get('stage2b_status','') for r in rows))}")
PYEOF
echo "=== done: $VARIANT ==="
