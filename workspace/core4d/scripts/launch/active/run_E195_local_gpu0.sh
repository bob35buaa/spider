#!/usr/bin/env bash
# Run the seven E195 rows assigned to local GPU0 alongside existing workloads.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
ROOT="workspace/core4d/results/E195"
MANIFEST="$ROOT/s6_downstream/manifests/cem_full_manifest.tsv"
RUNNER="workspace/core4d/scripts/experiments/E195/run_cem_queue.py"
AUDITOR="workspace/core4d/scripts/experiments/E195/audit_config.py"
SHARD="$ROOT/remote_staging/local-gpu0/full/manifest.tsv"

"$PYTHON_BIN" "$AUDITOR" --require-all
mkdir -p "$(dirname "$SHARD")" "logs/E195/launch"
"$PYTHON_BIN" - "$MANIFEST" "$SHARD" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
fields = list(rows[0])
rows = [
    row for row in rows
    if row["worker_id"] == "local-gpu0" and row["host"] == "local" and row["gpu_id"] == "0"
]
if len(rows) != 7:
    raise SystemExit(f"local-gpu0 rows={len(rows)} expected=7")
with open(sys.argv[2], "w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
print("local-gpu0 rows=7")
PY

SESSION="E195_full_local_$(date +%Y%m%d_%H%M%S)"
LOG="logs/E195/launch/full_local_gpu0.log"
tmux new-session -d -s "$SESSION" -c "$PWD" \
  "CUDA_VISIBLE_DEVICES=0 '$PYTHON_BIN' '$RUNNER' --mode full --manifest-tsv '$SHARD' --python-bin '$PYTHON_BIN' --gpu-id 0 > '$LOG' 2>&1"
"$PYTHON_BIN" - "$ROOT/remote_staging/latest_full_local.json" "$SESSION" <<'PY'
import json, sys
from datetime import datetime
from pathlib import Path
path = Path(sys.argv[1]); path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps({
    "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "session": sys.argv[2], "worker": "local-gpu0", "gpu": "0",
}, indent=2) + "\n")
PY
echo "E195 local append-only session started: $SESSION"

