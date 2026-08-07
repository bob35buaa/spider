#!/usr/bin/env bash
# E189 local 8-GPU CEM launcher (E167A_zOnlyBody, PRG intentionally OFF).
#
# Runs the frozen 43-row box004/box024/box001 no-PRG manifest on the local
# 8xL20Y GPUs, sharded by the per-row assigned_gpu (one E189 queue runner per
# GPU). Before launching it frees the GPUs by killing the utilization-filler
# /mnt/.../.cache/run.py (the user explicitly authorized this in E171-E174/
# E186; no other jobs are touched).
#
# Usage:
#   MODE=canary bash workspace/core4d/scripts/launch/active/run_E189_local_8gpu.sh
#   MODE=full   bash workspace/core4d/scripts/launch/active/run_E189_local_8gpu.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${MODE:-${1:-canary}}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
E=workspace/core4d/results/E189
RUNNER=workspace/core4d/scripts/experiments/E189/run_cem_queue.py
AUDITOR=workspace/core4d/scripts/experiments/E189/audit_e167a_no_prg.py
unset MUJOCO_GL   # DO NOT force egl/osmesa: both crash `import mujoco` with a raw AttributeError
                  # (not the ImportError mujoco's own try/except expects) depending on the
                  # invoking shell's GL/driver state -- verified 2026-08-06 to vary between
                  # shells on this exact host. Leaving MUJOCO_GL unset lets mujoco fall back to
                  # glfw, whose import failure is a clean ImportError that mujoco swallows, so
                  # `import mujoco` succeeds either way. Neither canary nor Full renders video
                  # (save_video=false, viewer disabled), so no GL context is actually needed here.

if [ "$MODE" = "canary" ]; then
  MANIFEST="$E/s6_downstream/manifests/cem_canary_manifest.tsv"
  RUN_MODE=canary
  EXPECTED_ROWS=9
elif [ "$MODE" = "full" ]; then
  MANIFEST="$E/s6_downstream/manifests/cem_full_manifest.tsv"
  RUN_MODE=full
  EXPECTED_ROWS=43
else
  echo "MODE must be canary|full" >&2; exit 2
fi
[ -f "$MANIFEST" ] || { echo "manifest missing: $MANIFEST" >&2; exit 2; }

ROWS=$(($(wc -l < "$MANIFEST") - 1))
[ "$ROWS" -eq "$EXPECTED_ROWS" ] || {
  echo "manifest row count drift: $MANIFEST has $ROWS rows, expected $EXPECTED_ROWS" >&2
  exit 2
}

echo "=== preflight: E167A/no-PRG audit against Full manifest ==="
"$PYTHON_BIN" "$AUDITOR" --require-all >/dev/null
echo "preflight audit passed (43/43 method parity, 43/43 no-PRG)"

SHARD_DIR="$E/s6_downstream/cem/shards/$MODE"
mkdir -p "$SHARD_DIR" "logs/E189/cem/$MODE"

# --- free GPUs: kill the utilization-filler run.py (user-authorized) ----------
echo "=== freeing GPUs: killing utilization-filler .cache/run.py (authorized) ==="
pkill -f "/mnt/ali-sh-1/usr/xiayibo/.cache/run.py" 2>/dev/null || true
sleep 5
echo "GPU free memory after kill:"
nvidia-smi --query-gpu=index,memory.free,memory.used,utilization.gpu --format=csv,noheader

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
for gpu, grp in sorted(by_gpu.items(), key=lambda kv: int(kv[0])):
    out = shard_dir / f"gpu{gpu}.tsv"
    with out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter='\t', lineterminator='\n')
        w.writeheader(); w.writerows(grp)
print(" ".join(sorted(by_gpu, key=int)))
PY
)
echo "GPUs in shard: $GPUS"

# --- launch one runner per GPU -----------------------------------------------
declare -a PIDS=()
declare -a PID_GPUS=()
for g in $GPUS; do
  shard="$SHARD_DIR/gpu${g}.tsv"
  log="logs/E189/cem/$MODE/gpu${g}.log"
  echo "launch gpu=$g shard=$shard -> $log"
  nohup "$PYTHON_BIN" "$RUNNER" --mode "$RUN_MODE" --manifest-tsv "$shard" \
    --python-bin "$PYTHON_BIN" --gpu-id "$g" --all \
    > "$log" 2>&1 &
  PIDS+=($!)
  PID_GPUS+=("$g")
done
echo "launched ${#PIDS[@]} runners: ${PIDS[*]}"

# --- wait for all shards ------------------------------------------------------
FAIL=0
for idx in "${!PIDS[@]}"; do
  pid="${PIDS[$idx]}"
  gpu="${PID_GPUS[$idx]}"
  if ! wait "$pid"; then
    echo "runner on gpu=$gpu (pid=$pid) failed; see logs/E189/cem/$MODE/gpu${gpu}.log" >&2
    FAIL=1
  fi
done

# --- merge shard rows back into the canonical manifest ------------------------
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
