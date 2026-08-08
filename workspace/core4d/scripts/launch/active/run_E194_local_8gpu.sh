#!/usr/bin/env bash
# E194 local 8-GPU gravity-compensation 2x2 CEM launcher.
#
# Runs the frozen G1/G2/G3 gravcomp manifest (PRG ON, E167A_zOnlyBody) on the
# local 8xL20Y GPUs, sharded by each row's assigned_gpu (one queue runner per
# GPU). Before launching it frees the GPUs by killing the utilization-filler
# .cache/run.py (user-authorized in E171-E174/E186/E189; no other jobs touched).
#
# Usage:
#   MODE=canary bash workspace/core4d/scripts/launch/active/run_E194_local_8gpu.sh
#   MODE=full   bash workspace/core4d/scripts/launch/active/run_E194_local_8gpu.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${MODE:-${1:-canary}}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
E=workspace/core4d/results/E194
RUNNER=workspace/core4d/scripts/experiments/E194/run_cem_queue.py
AUDITOR=workspace/core4d/scripts/experiments/E194/audit_gravcomp_scenes.py
unset MUJOCO_GL   # DO NOT force egl/osmesa: both crash `import mujoco` with a raw
                  # AttributeError (not the ImportError mujoco's try/except expects)
                  # depending on the shell's GL/driver state -- verified 2026-08-06 to
                  # vary between shells on this host. Unset lets mujoco fall back to
                  # glfw (clean ImportError, swallowed). CEM renders no video here.

if [ "$MODE" = "canary" ]; then
  MANIFEST="$E/s6_downstream/manifests/cem_canary_manifest.tsv"
  RUN_MODE=canary
  EXPECTED_ROWS=9   # 3 canary cases x 3 arms (G1/G2/G3)
elif [ "$MODE" = "full" ]; then
  MANIFEST="$E/s6_downstream/manifests/cem_full_manifest.tsv"
  RUN_MODE=full
  EXPECTED_ROWS=45  # 15 cases x 3 arms
else
  echo "MODE must be canary|full" >&2; exit 2
fi
[ -f "$MANIFEST" ] || { echo "manifest missing: $MANIFEST" >&2; exit 2; }

ROWS=$(($(wc -l < "$MANIFEST") - 1))
[ "$ROWS" -eq "$EXPECTED_ROWS" ] || {
  echo "manifest row count drift: $MANIFEST has $ROWS rows, expected $EXPECTED_ROWS" >&2
  exit 2
}

echo "=== preflight: per-arm gravcomp/gain audit against Full manifest ==="
"$PYTHON_BIN" "$AUDITOR" --require-all >/dev/null
echo "preflight audit passed (per-arm gravcomp + kp + rot parity)"

SHARD_DIR="$E/s6_downstream/cem/shards/$MODE"
mkdir -p "$SHARD_DIR" "logs/E194/cem/$MODE"

echo "=== GPU free check ==="
if [ "${FREE_GPUS:-0}" = "1" ]; then
  # Opt-in only (FREE_GPUS=1): kill the utilization-filler .cache/run.py to free
  # the GPUs. Default is OFF so a normal launch never touches jobs it did not
  # create -- if the GPUs are already free (the usual case here) this is skipped.
  echo "FREE_GPUS=1: killing utilization-filler .cache/run.py"
  pkill -f "/mnt/ali-sh-1/usr/xiayibo/.cache/run.py" 2>/dev/null || true
  sleep 5
fi
nvidia-smi --query-gpu=index,memory.free,memory.used,utilization.gpu --format=csv,noheader

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

declare -a PIDS=()
declare -a PID_GPUS=()
for g in $GPUS; do
  shard="$SHARD_DIR/gpu${g}.tsv"
  log="logs/E194/cem/$MODE/gpu${g}.log"
  echo "launch gpu=$g shard=$shard -> $log"
  nohup "$PYTHON_BIN" "$RUNNER" --mode "$RUN_MODE" --manifest-tsv "$shard" \
    --python-bin "$PYTHON_BIN" --gpu-id "$g" --all \
    > "$log" 2>&1 &
  PIDS+=($!)
  PID_GPUS+=("$g")
done
echo "launched ${#PIDS[@]} runners: ${PIDS[*]}"

FAIL=0
for idx in "${!PIDS[@]}"; do
  pid="${PIDS[$idx]}"
  gpu="${PID_GPUS[$idx]}"
  if ! wait "$pid"; then
    echo "runner on gpu=$gpu (pid=$pid) failed; see logs/E194/cem/$MODE/gpu${gpu}.log" >&2
    FAIL=1
  fi
done

$PYTHON_BIN - "$MANIFEST" "$SHARD_DIR" "$GPUS" <<'PY'
import csv, sys
from pathlib import Path
manifest, shard_dir, gpus = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3].split()
rows = list(csv.DictReader(open(manifest), delimiter='\t'))
fields = list(rows[0].keys()) if rows else []
merged = {}
for g in gpus:
    for r in csv.DictReader(open(shard_dir / f"gpu{g}.tsv"), delimiter='\t'):
        merged[r['variant']] = r
for r in rows:
    if r['variant'] in merged:
        r.update({k: merged[r['variant']].get(k, r.get(k, '')) for k in r})
with manifest.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields, delimiter='\t', lineterminator='\n')
    w.writeheader(); w.writerows(rows)
from collections import Counter
print("status:", dict(Counter(r.get('status','') for r in rows)))
PY

echo "=== $MODE CEM done (launcher fail=$FAIL) ==="
exit $FAIL
