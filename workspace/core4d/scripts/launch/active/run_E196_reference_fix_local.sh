#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
MODE="${MODE:-wave0}"
LOCAL_GPU_ID="${LOCAL_GPU_ID:-0}"
CHECK_ONLY="${CHECK_ONLY:-0}"
ROOT="workspace/core4d/results/E196"
RUNNER="workspace/core4d/scripts/experiments/E196/run_reference_fix_queue.py"
AUDITOR="workspace/core4d/scripts/experiments/E196/audit_reference_fix.py"
case "$MODE" in
  wave0) MANIFEST="$ROOT/s6_downstream/manifests/reference_fix_wave0_manifest.tsv"; EXPECTED=1 ;;
  remaining) MANIFEST="$ROOT/s6_downstream/manifests/reference_fix_remaining_manifest.tsv"; EXPECTED=9 ;;
  *) echo "unsupported MODE=$MODE" >&2; exit 2 ;;
esac

SHARD="$ROOT/s6_downstream/execution/local/$MODE/manifest.tsv"
mkdir -p "$(dirname "$SHARD")" "logs/E196/launch"
"$PYTHON_BIN" - "$MANIFEST" "$SHARD" "$EXPECTED" <<'PY'
import csv,sys
from pathlib import Path
source,target,expected=Path(sys.argv[1]),Path(sys.argv[2]),int(sys.argv[3])
rows=list(csv.DictReader(source.open(),delimiter="\t")); fields=list(rows[0])
rows=[row for row in rows if row["worker"]=="local-gpu0"]
if len(rows)!=expected: raise SystemExit(f"local rows={len(rows)} expected={expected}")
target.parent.mkdir(parents=True,exist_ok=True)
with target.open("w",newline="") as stream:
 writer=csv.DictWriter(stream,fieldnames=fields,delimiter="\t",lineterminator="\n")
 writer.writeheader(); writer.writerows(rows)
print(f"local rows={len(rows)}")
PY
"$PYTHON_BIN" "$AUDITOR" --manifest "$SHARD" --scope prelaunch --allow-subset --require-all

GPU_STATE="$(nvidia-smi -i "$LOCAL_GPU_ID" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)"
"$PYTHON_BIN" - "$GPU_STATE" <<'PY'
import sys
memory,utilization=(int(value.strip()) for value in sys.argv[1].split(","))
if memory>=2000 or utilization>=20:
    raise SystemExit(f"local GPU is busy: memory={memory}MB utilization={utilization}%")
print(f"local GPU available: memory={memory}MB utilization={utilization}%")
PY
COMPUTE_PIDS="$(nvidia-smi -i "$LOCAL_GPU_ID" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')"
if [ -n "$COMPUTE_PIDS" ]; then
  echo "local GPU has compute processes: $COMPUTE_PIDS" >&2
  exit 4
fi
if [ "$CHECK_ONLY" = "1" ]; then
  echo "E196 local $MODE check passed"
  exit 0
fi

GPU_SNAPSHOT="$ROOT/s6_downstream/execution/local/$MODE/gpu_snapshot_before_launch.txt"
{
  date --iso-8601=seconds
  printf 'mode=%s\ngpu_id=%s\nstate=%s\ncompute_pids=%s\n' \
    "$MODE" "$LOCAL_GPU_ID" "$GPU_STATE" "${COMPUTE_PIDS:-none}"
  git rev-parse HEAD
  sha256sum "$RUNNER" examples/run_mjwp.py spider/simulators/scene_act_reference.py
} > "$GPU_SNAPSHOT"
SESSION="E196_reference_${MODE}_local"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 5
fi
LOG="logs/E196/launch/${MODE}_local_gpu${LOCAL_GPU_ID}.log"
tmux new-session -d -s "$SESSION" -c "$PWD" \
  "PYTHONPATH='$PWD' CUDA_VISIBLE_DEVICES='$LOCAL_GPU_ID' '$PYTHON_BIN' '$RUNNER' --manifest-tsv '$SHARD' --python-bin '$PYTHON_BIN' --gpu-id '$LOCAL_GPU_ID' > '$LOG' 2>&1"
"$PYTHON_BIN" - "$ROOT/s6_downstream/execution/latest_${MODE}_local.json" "$SESSION" "$SHARD" "$LOCAL_GPU_ID" "$GPU_SNAPSHOT" <<'PY'
import json,sys
from datetime import datetime
from pathlib import Path
path=Path(sys.argv[1]); path.parent.mkdir(parents=True,exist_ok=True)
path.write_text(json.dumps({"created_at":datetime.now().astimezone().isoformat(timespec="seconds"),"session":sys.argv[2],"shard":sys.argv[3],"gpu_id":sys.argv[4],"gpu_snapshot":sys.argv[5]},indent=2)+"\n")
PY
echo "E196 local $MODE session started: $SESSION"
