#!/usr/bin/env bash
# E194 render: self MP4 for every completed G1/G2/G3 rollout, then the 4-cell
# (A0/G1/G2/G3) keyframe montages for the key cases. CPU render via osmesa,
# sharded across cores (each shard renders a disjoint set of variants).
#
# Usage: [NSHARDS=8] bash workspace/core4d/scripts/launch/active/run_E194_render_all.sh
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
NSHARDS="${NSHARDS:-8}"

E=workspace/core4d/results/E194
RUNNER=workspace/core4d/scripts/experiments/E194/render_E194_arms.py
KEYFRAMES=workspace/core4d/scripts/experiments/E194/render_E194_keyframes.py
MANIFEST="$E/s6_downstream/manifests/cem_full_manifest.tsv"
LOGD="logs/E194/render/full"
mkdir -p "$LOGD"
[ -f "$MANIFEST" ] || { echo "manifest missing: $MANIFEST" >&2; exit 2; }

# eligible = rows whose rollout npz exists
mapfile -t ALL < <($PYTHON_BIN - "$MANIFEST" <<'PY'
import csv, sys
from pathlib import Path
REPO = Path.cwd()
for r in csv.DictReader(open(sys.argv[1]), delimiter="\t"):
    p = Path(r["outdir_npz"])
    if (p if p.is_absolute() else REPO / p).is_file():
        print(r["variant"])
PY
)
echo "eligible variants: ${#ALL[@]} (nshards=$NSHARDS)"
[ "${#ALL[@]}" -gt 0 ] || { echo "no completed rollouts to render" >&2; exit 0; }

declare -a PIDS=()
for ((s=0; s<NSHARDS; s++)); do
  shard=()
  for ((i=s; i<${#ALL[@]}; i+=NSHARDS)); do shard+=("${ALL[$i]}"); done
  [ ${#shard[@]} -eq 0 ] && continue
  nohup $PYTHON_BIN "$RUNNER" --variants "${shard[@]}" > "$LOGD/shard_${s}.log" 2>&1 &
  PIDS+=($!)
  echo "shard $s: ${#shard[@]} variants -> pid $!"
done
FAIL=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=$((FAIL+1)); done
echo "=== self-MP4 render done (nonzero shards=$FAIL) ==="

echo "=== building 4-cell keyframe montages (key cases) ==="
$PYTHON_BIN "$KEYFRAMES" > "$LOGD/keyframes.log" 2>&1 || echo "keyframes step returned nonzero (see $LOGD/keyframes.log)"
echo "=== E194 render all done ==="
