#!/usr/bin/env bash
# Run the E192 rows assigned to local-gpu0 without touching other processes.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
MODE="${MODE:-${1:-full}}"
PY="${PYTHON_BIN:-.venv/bin/python}"
ROOT="workspace/core4d/results/E192"
MANIFEST="$ROOT/s6_downstream/manifests/cem_${MODE}_manifest.tsv"
RUNNER="workspace/core4d/scripts/experiments/E192/run_cem_queue.py"
AUDITOR="workspace/core4d/scripts/experiments/E192/audit_gate_overrides.py"
case "$MODE" in canary|full) ;; *) echo "MODE must be canary|full" >&2; exit 2 ;; esac
"$PY" "$AUDITOR" --require-all
SHARD="$ROOT/remote_staging/local-gpu0/$MODE/manifest.tsv"
mkdir -p "$(dirname "$SHARD")" "logs/E192/launch"
"$PY" - "$MANIFEST" "$SHARD" <<'PY'
import csv, sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter="\t")); fields=list(rows[0])
rows=[r for r in rows if r["worker_id"]=="local-gpu0" and r["host"]=="local" and r["gpu_id"]=="0"]
if not rows: raise SystemExit("local-gpu0 shard is empty")
with open(sys.argv[2],"w",newline="") as stream:
    w=csv.DictWriter(stream,fieldnames=fields,delimiter="\t",lineterminator="\n")
    w.writeheader(); w.writerows(rows)
print(f"local-gpu0 rows={len(rows)}")
PY
SESSION="E192_${MODE}_local_$(date +%Y%m%d_%H%M%S)"
LOG="logs/E192/launch/${MODE}_local_gpu0.log"
tmux new-session -d -s "$SESSION" -c "$PWD" \
  "CUDA_VISIBLE_DEVICES=0 '$PY' '$RUNNER' --mode '$MODE' --manifest-tsv '$SHARD' --python-bin '$PY' --gpu-id 0 > '$LOG' 2>&1"
echo "E192 local append-only session started: $SESSION"
