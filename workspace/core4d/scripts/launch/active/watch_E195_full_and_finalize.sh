#!/usr/bin/env bash
# Wait for E195's own sessions, then pull, evaluate and render. Never retries or moves rows.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
REMOTE="${E195_ADA_REMOTE:-spider-remote}"
ROOT="workspace/core4d/results/E195"
LOCAL_POINTER="$ROOT/remote_staging/latest_full_local.json"
REMOTE_POINTER="$ROOT/remote_staging/latest_full_ada6000.json"
LOG="logs/E195/monitor/full_finalize.log"
mkdir -p "$(dirname "$LOG")"

mapfile -t SESSIONS < <("$PYTHON_BIN" - "$LOCAL_POINTER" "$REMOTE_POINTER" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["session"])
print(json.load(open(sys.argv[2]))["session"])
PY
)
LOCAL_SESSION="${SESSIONS[0]}"
REMOTE_SESSION="${SESSIONS[1]}"

exec > >(tee -a "$LOG") 2>&1
echo "[$(date '+%F %T')] watcher start local=$LOCAL_SESSION remote=$REMOTE_SESSION"
while true; do
  local_active=0
  remote_active=0
  tmux has-session -t "$LOCAL_SESSION" 2>/dev/null && local_active=1 || true
  if ssh -o BatchMode=yes -o ConnectTimeout=20 "$REMOTE" \
      "tmux has-session -t '$REMOTE_SESSION' 2>/dev/null"; then
    remote_active=1
  else
    rc=$?
    if [ "$rc" -eq 255 ]; then
      echo "[$(date '+%F %T')] remote status unavailable; keep waiting"
      sleep 60
      continue
    fi
  fi
  local_done=$(find "$ROOT/s6_downstream/cem/full" -maxdepth 1 -type f -name 'E195_*_A3.npz' 2>/dev/null | wc -l)
  echo "[$(date '+%F %T')] active local=$local_active remote=$remote_active local_done=$local_done"
  if [ "$local_active" -eq 0 ] && [ "$remote_active" -eq 0 ]; then
    break
  fi
  sleep 60
done

echo "[$(date '+%F %T')] workers ended; pulling Ada results"
bash workspace/core4d/scripts/launch/active/pull_E195_remote_Ada6000_results.sh

"$PYTHON_BIN" - <<'PY'
import csv
from collections import Counter
path = "workspace/core4d/results/E195/s6_downstream/manifests/cem_full_manifest.tsv"
rows = list(csv.DictReader(open(path), delimiter="\t"))
counts = Counter(row["status"] for row in rows)
print(f"canonical manifest status={dict(counts)}")
if len(rows) != 15 or counts != {"run_complete_pending_eval": 15}:
    raise SystemExit("E195 Full did not close 15/15; stop before evaluation")
PY

echo "[$(date '+%F %T')] evaluating"
bash workspace/core4d/scripts/eval/wrappers/eval_E195_stricter_hand_gate.sh
echo "[$(date '+%F %T')] rendering"
bash workspace/core4d/scripts/launch/active/run_E195_render_all.sh --skip-existing
"$PYTHON_BIN" workspace/core4d/scripts/eval/reports/gen_E195_comparison.py
echo "[$(date '+%F %T')] automatic finalize complete; visual review remains"

