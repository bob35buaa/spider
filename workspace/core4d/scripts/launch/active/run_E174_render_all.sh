#!/usr/bin/env bash
# E174 Full CEM视频渲染 — 全部 53 条 CEM-complete case 渲染 MP4（强制视觉复核，plan 8.3）。
#
# render_cem_results.py 是串行的且会 skip 已存在 MP4（resume 安全）。渲染走 osmesa
# 纯 CPU（config.device=cpu），因此按 case sharding 后可在本机 192 核上并行提速。
# 每个 shard 是一个独立进程，跑 render_cem_results.py --cases <该 shard 的 case_ids>。
#
# Usage: [NSHARDS=8] bash workspace/core4d/scripts/launch/active/run_E174_render_all.sh
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
NSHARDS="${NSHARDS:-8}"

E=workspace/core4d/results/E174
RUNNER=workspace/core4d/scripts/experiments/E174/render_cem_results.py
MANIFEST="$E/s6_downstream/manifests/cem_full_manifest.tsv"
LOGD="logs/E174/render/full"
mkdir -p "$LOGD"
[ -f "$MANIFEST" ] || { echo "manifest missing: $MANIFEST" >&2; exit 2; }

# --- collect eligible case_ids, split round-robin into NSHARDS ----------------
mapfile -t ALL_CASES < <($PYTHON_BIN - "$MANIFEST" <<'PY'
import csv, sys
from pathlib import Path
COMPLETE = {"run_complete_pending_eval", "run_complete", "eval_complete"}
rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
for r in rows:
    if r.get("status") in COMPLETE:
        print(r["case_id"])
PY
)
echo "eligible cases: ${#ALL_CASES[@]} (nshards=$NSHARDS)"

declare -a PIDS=()
for ((s=0; s<NSHARDS; s++)); do
  shard_cases=()
  for ((i=s; i<${#ALL_CASES[@]}; i+=NSHARDS)); do shard_cases+=("${ALL_CASES[$i]}"); done
  [ ${#shard_cases[@]} -eq 0 ] && continue
  nohup $PYTHON_BIN "$RUNNER" --cases "${shard_cases[@]}" \
    > "$LOGD/shard_${s}.log" 2>&1 &
  PIDS+=($!)
  echo "shard $s: ${#shard_cases[@]} cases -> pid $!"
done

FAIL=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=$((FAIL+1)); done
echo "=== all render shards done (nonzero=$FAIL) ==="

# --- final tally --------------------------------------------------------------
NMP4=$(find "$E/s6_downstream/render/full" -maxdepth 1 -name '*_full.mp4' | wc -l)
echo "total MP4s in render/full: $NMP4 (expected ${#ALL_CASES[@]})"
exit $FAIL
