#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
MODE="${MODE:-canary}"
SENTINEL_ONLY="${SENTINEL_ONLY:-0}"
ROOT="workspace/core4d/results/E194"
RUNNER="workspace/core4d/scripts/experiments/E194/run_g1_expansion_queue.py"
AUDITOR="workspace/core4d/scripts/experiments/E194/audit_g1_expansion.py"
case "$MODE:$SENTINEL_ONLY" in
  canary:*) TAG="canary"; MANIFEST="$ROOT/s6_downstream/manifests/g1_expansion_canary_manifest.tsv" ;;
  full:1) TAG="sentinel"; MANIFEST="$ROOT/s6_downstream/manifests/g1_expansion_sentinel_manifest.tsv" ;;
  full:0) TAG="full"; MANIFEST="$ROOT/s6_downstream/manifests/g1_expansion_full_manifest.tsv" ;;
  *) echo "unsupported MODE=$MODE SENTINEL_ONLY=$SENTINEL_ONLY" >&2; exit 2 ;;
esac
"$PYTHON_BIN" "$AUDITOR" --require-all
SHARD="$ROOT/remote_staging/local-gpu0/$TAG/manifest.tsv"
mkdir -p "$(dirname "$SHARD")" "logs/E194/launch"
"$PYTHON_BIN" - "$MANIFEST" "$SHARD" "$TAG" <<'PY'
import csv, sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter="\t")); fields=list(rows[0])
rows=[row for row in rows if row["worker"]=="local-gpu0"]
expected={"canary":3,"sentinel":4,"full":36}[sys.argv[3]]
if len(rows)!=expected: raise SystemExit(f"local rows={len(rows)} expected={expected}")
with open(sys.argv[2],"w",newline="") as stream:
 w=csv.DictWriter(stream,fieldnames=fields,delimiter="\t",lineterminator="\n"); w.writeheader(); w.writerows(rows)
print(f"local rows={len(rows)}")
PY
SESSION="E194_G1_${TAG}_local_$(date +%Y%m%d_%H%M%S)"
LOG="logs/E194/launch/${TAG}_local_gpu0.log"
tmux new-session -d -s "$SESSION" -c "$PWD" \
  "CUDA_VISIBLE_DEVICES=0 '$PYTHON_BIN' '$RUNNER' --manifest-tsv '$SHARD' --python-bin '$PYTHON_BIN' --gpu-id 0 > '$LOG' 2>&1"
"$PYTHON_BIN" - "$ROOT/remote_staging/latest_${TAG}_local.json" "$SESSION" "$SHARD" <<'PY'
import json,sys
from datetime import datetime
from pathlib import Path
p=Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True)
p.write_text(json.dumps({"created_at":datetime.now().astimezone().isoformat(timespec="seconds"),"session":sys.argv[2],"shard":sys.argv[3]},indent=2)+"\n")
PY
echo "E194 G1 $TAG local session started: $SESSION"
