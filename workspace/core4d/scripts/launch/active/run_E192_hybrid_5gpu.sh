#!/usr/bin/env bash
# E192 five-worker orchestrator. Append-only: never kills, waits for, or moves jobs.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${MODE:-${1:-baseline_sentinel}}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
PREFLIGHT_ONLY="${E192_PREFLIGHT_ONLY:-0}"
ROOT="workspace/core4d/results/E192"
RUNNER="workspace/core4d/scripts/experiments/E192/run_cem_queue.py"
AUDITOR="workspace/core4d/scripts/experiments/E192/audit_gate_overrides.py"
case "$MODE" in
  baseline_sentinel) MANIFEST="$ROOT/s6_downstream/manifests/cem_baseline_sentinel_manifest.tsv" ;;
  canary) MANIFEST="$ROOT/s6_downstream/manifests/cem_canary_manifest.tsv" ;;
  full) MANIFEST="$ROOT/s6_downstream/manifests/cem_full_manifest.tsv" ;;
  *) echo "MODE must be baseline_sentinel|canary|full" >&2; exit 2 ;;
esac

"$PYTHON_BIN" "$AUDITOR" --require-all
[ -f "$MANIFEST" ] || { echo "missing manifest: $MANIFEST" >&2; exit 2; }
SHARD="$ROOT/remote_staging/local-gpu0/$MODE/manifest.tsv"
mkdir -p "$(dirname "$SHARD")" "logs/E192/launch"
"$PYTHON_BIN" - "$MANIFEST" "$SHARD" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
fields=list(rows[0])
rows=[r for r in rows if r['worker_id']=='local-gpu0' and r['host']=='local' and r['gpu_id']=='0']
if not rows: raise SystemExit('local-gpu0 shard is empty')
with open(sys.argv[2],'w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields,delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rows)
print(f'local-gpu0 rows={len(rows)}')
PY

if [ "$PREFLIGHT_ONLY" = "1" ]; then
  "$PYTHON_BIN" "$RUNNER" --mode "$MODE" --manifest-tsv "$SHARD" --python-bin "$PYTHON_BIN" --gpu-id 0 --all --dry-run
  E192_PREFLIGHT_ONLY=1 MODE="$MODE" bash workspace/core4d/scripts/launch/active/run_E192_remote_A100.sh
  exit 0
fi

SESSION="E192_${MODE}_local_$(date +%Y%m%d_%H%M%S)"
REMOTE_LOG="logs/E192/launch/${MODE}_remote_launch.log"
LOCAL_LOG="logs/E192/launch/${MODE}_local_gpu0.log"
tmux new-session -d -s "$SESSION" -c "$PWD" \
  "CUDA_VISIBLE_DEVICES=0 '$PYTHON_BIN' '$RUNNER' --mode '$MODE' --manifest-tsv '$SHARD' --python-bin '$PYTHON_BIN' --gpu-id 0 --all > '$LOCAL_LOG' 2>&1"
MODE="$MODE" bash workspace/core4d/scripts/launch/active/run_E192_remote_A100.sh >"$REMOTE_LOG" 2>&1
echo "E192 append-only launch complete: local_session=$SESSION mode=$MODE"

