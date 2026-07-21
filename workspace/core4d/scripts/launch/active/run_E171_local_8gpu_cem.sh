#!/usr/bin/env bash
# E171 local 8-GPU CEM launcher (rubber_hull + E170 PRG cross-object candidate).
#
# Runs the frozen S5_READY CEM manifest on the local GPUs, sharded by the
# per-row assigned_gpu, one E169/E171 queue runner per GPU. Before launching it
# frees the GPUs by killing the utilization-filler /mnt/.../.cache/run.py (the
# user explicitly authorized this; no other jobs are touched).
#
# Usage:
#   MODE=canary bash workspace/core4d/scripts/launch/active/run_E171_local_8gpu_cem.sh
#   MODE=full   bash workspace/core4d/scripts/launch/active/run_E171_local_8gpu_cem.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${MODE:-${1:-canary}}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
E=workspace/core4d/results/E171
RUNNER=workspace/core4d/scripts/experiments/E171/run_cem_queue.py
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"   # CEM sim uses MuJoCo Warp (GPU compute); egl is broken on this host, osmesa is a safe fallback

if [ "$MODE" = "canary" ]; then
  MANIFEST="$E/s6_downstream/manifests/cem_canary_manifest.tsv"
  NS="${NS:-64}"; NI="${NI:-4}"; RUN_MODE=canary
elif [ "$MODE" = "full" ]; then
  MANIFEST="$E/s6_downstream/manifests/cem_full_manifest.tsv"
  NS="${NS:-1024}"; NI="${NI:-32}"; RUN_MODE=full
else
  echo "MODE must be canary|full" >&2; exit 2
fi
[ -f "$MANIFEST" ] || { echo "manifest missing: $MANIFEST" >&2; exit 2; }

SHARD_DIR="$E/s6_downstream/cem/shards/$MODE"
mkdir -p "$SHARD_DIR" "logs/E171/cem/$MODE"

# --- free GPUs: kill the utilization-filler run.py (user-authorized) ----------
echo "=== freeing GPUs: killing utilization-filler .cache/run.py (authorized) ==="
pkill -f "/mnt/ali-sh-1/usr/xiayibo/.cache/run.py" 2>/dev/null || true
sleep 5
echo "GPU free memory after kill:"
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | head -8

# --- shard manifest by assigned_gpu ------------------------------------------
echo "=== sharding $MANIFEST by assigned_gpu ==="
GPUS=$($PYTHON_BIN - "$MANIFEST" "$SHARD_DIR" <<'PY'
import csv, sys
from pathlib import Path
manifest, shard_dir = Path(sys.argv[1]), Path(sys.argv[2])
rows = list(csv.DictReader(open(manifest), delimiter='\t'))
fields = list(rows[0].keys()) if rows else []
by_gpu = {}
for r in rows:
    by_gpu.setdefault(r.get('assigned_gpu', '0'), []).append(r)
for gpu, grp in sorted(by_gpu.items()):
    out = shard_dir / f"gpu{gpu}.tsv"
    with out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter='\t', lineterminator='\n')
        w.writeheader(); w.writerows(grp)
print(" ".join(sorted(by_gpu)))
PY
)
echo "GPUs in shard: $GPUS"

# --- launch one runner per GPU -----------------------------------------------
declare -a PIDS=()
for g in $GPUS; do
  shard="$SHARD_DIR/gpu${g}.tsv"
  log="logs/E171/cem/$MODE/gpu${g}.log"
  echo "launch gpu=$g shard=$shard -> $log"
  nohup $PYTHON_BIN "$RUNNER" --mode "$RUN_MODE" --manifest-tsv "$shard" \
    --gpu-id "$g" --num-samples "$NS" --max-num-iterations "$NI" --all \
    > "$log" 2>&1 &
  PIDS+=($!)
done
echo "launched ${#PIDS[@]} runners: ${PIDS[*]}"

# --- wait for all shards ------------------------------------------------------
FAIL=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || FAIL=1
done

# --- merge shard rows back into the main manifest ----------------------------
$PYTHON_BIN - "$MANIFEST" "$SHARD_DIR" "$GPUS" <<'PY'
import csv, sys
from pathlib import Path
manifest, shard_dir, gpus = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3].split()
rows = list(csv.DictReader(open(manifest), delimiter='\t'))
fields = list(rows[0].keys()) if rows else []
merged = {}
for g in gpus:
    for r in csv.DictReader(open(shard_dir / f"gpu{g}.tsv"), delimiter='\t'):
        merged[r['case_id']] = r
for r in rows:
    if r['case_id'] in merged:
        r.update({k: merged[r['case_id']].get(k, r.get(k, '')) for k in r})
with manifest.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields, delimiter='\t', lineterminator='\n')
    w.writeheader(); w.writerows(rows)
from collections import Counter
print("status:", dict(Counter(r.get('status','') for r in rows)))
PY

echo "=== $MODE CEM done (launcher fail=$FAIL) ==="
exit $FAIL
