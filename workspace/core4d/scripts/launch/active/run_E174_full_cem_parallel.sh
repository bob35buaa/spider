#!/usr/bin/env bash
# E174 Full CEM parallel driver (CPU-bound cases packed across GPUs).
#
# The per-GPU serial launcher (run_E174_local_8gpu_cem.sh) runs one case at a
# time per GPU. Each CEM case (use_torch_compile=false) is CPU-bound (~1 core at
# ~97%, ~1.7GB GPU), so on this 192-core / 8x81GB host we can run ~all 53 cases
# concurrently and finish in ~one case-time instead of ~7 serial rounds.
# Numerics are unchanged (same frozen samples/steps/seed/reward per row); only
# scheduling changes. Each case keeps its manifest assigned_gpu.
#
# Usage: MAXJOBS=48 bash .../run_E174_full_cem_parallel.sh
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
MAXJOBS="${MAXJOBS:-48}"
NS="${NS:-1024}"; NI="${NI:-32}"

E=workspace/core4d/results/E174
RUNNER=workspace/core4d/scripts/experiments/E174/run_cem_queue.py
MANIFEST="$E/s6_downstream/manifests/cem_full_manifest.tsv"
SHARD_DIR="$E/s6_downstream/cem/shards/full_par"
LOGD="logs/E174/cem/full_par"
mkdir -p "$SHARD_DIR" "$LOGD"
[ -f "$MANIFEST" ] || { echo "manifest missing: $MANIFEST" >&2; exit 2; }

# free GPUs (authorized filler)
pkill -f "/mnt/ali-sh-1/usr/xiayibo/.cache/run.py" 2>/dev/null || true
sleep 2

# --- split into 1-case shards (skip already-completed) ------------------------
echo "=== splitting $MANIFEST into per-case shards ==="
mapfile -t CASES < <($PYTHON_BIN - "$MANIFEST" "$SHARD_DIR" <<'PY'
import csv, sys
from pathlib import Path
manifest, shard_dir = Path(sys.argv[1]), Path(sys.argv[2])
rows = list(csv.DictReader(open(manifest), delimiter="\t"))
fields = list(rows[0].keys())
scol = "status" if "status" in fields else None
done = {"run_complete_pending_eval", "run_complete", "pass"}
for r in rows:
    if scol and r.get(scol, "") in done:
        continue
    cid = r["case_id"]; gpu = r.get("assigned_gpu", "0") or "0"
    out = shard_dir / f"{cid}.tsv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        w.writeheader(); w.writerow(r)
    print(f"{cid}\t{gpu}\t{out}")
PY
)
echo "cases to run: ${#CASES[@]} (cap=$MAXJOBS)"

# --- launch with concurrency cap ---------------------------------------------
running=0
declare -a PIDS=()
for line in "${CASES[@]}"; do
  cid="${line%%$'\t'*}"; rest="${line#*$'\t'}"; gpu="${rest%%$'\t'*}"; shard="${rest#*$'\t'}"
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 3; done
  nohup $PYTHON_BIN "$RUNNER" --mode full --manifest-tsv "$shard" \
    --gpu-id "$gpu" --num-samples "$NS" --max-num-iterations "$NI" --all \
    > "$LOGD/${cid}.log" 2>&1 &
  PIDS+=($!)
done
echo "launched ${#PIDS[@]} case runners"
FAIL=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=$((FAIL+1)); done
echo "=== all case runners done (nonzero=$FAIL) ==="

# --- merge per-case shards back into full manifest ---------------------------
$PYTHON_BIN - "$MANIFEST" "$SHARD_DIR" <<'PY'
import csv, sys, collections
from pathlib import Path
manifest, shard_dir = Path(sys.argv[1]), Path(sys.argv[2])
rows = list(csv.DictReader(open(manifest), delimiter="\t"))
fields = list(rows[0].keys())
by = {r["case_id"]: r for r in rows}
for sh in shard_dir.glob("*.tsv"):
    for r in csv.DictReader(open(sh), delimiter="\t"):
        cid = r["case_id"]
        if cid in by:
            for k in fields:
                if r.get(k, "") != "":
                    by[cid][k] = r[k]
with manifest.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
    w.writeheader(); w.writerows(rows)
print("status:", dict(collections.Counter(r.get("status", "") for r in rows)))
PY
echo "=== full parallel CEM done (fail=$FAIL) ==="
exit $FAIL
