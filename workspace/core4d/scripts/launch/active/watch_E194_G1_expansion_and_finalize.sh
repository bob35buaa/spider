#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
TAG="${1:-full}"
REMOTE="${E194_ADA_REMOTE:-spider-remote}"
ROOT="workspace/core4d/results/E194"
LOCAL_POINTER="$ROOT/remote_staging/latest_${TAG}_local.json"
REMOTE_POINTER="$ROOT/remote_staging/latest_${TAG}_ada6000.json"
case "$TAG" in
  canary) EXPECTED=9; MANIFEST="$ROOT/s6_downstream/manifests/g1_expansion_canary_manifest.tsv" ;;
  sentinel) EXPECTED=6; MANIFEST="$ROOT/s6_downstream/manifests/g1_expansion_sentinel_manifest.tsv" ;;
  full) EXPECTED=72; MANIFEST="$ROOT/s6_downstream/manifests/g1_expansion_full_manifest.tsv" ;;
  *) echo "unsupported tag: $TAG" >&2; exit 2 ;;
esac
mapfile -t SESSIONS < <("$PYTHON_BIN" - "$LOCAL_POINTER" "$REMOTE_POINTER" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["session"]); print(json.load(open(sys.argv[2]))["session"])
PY
)
LOCAL_SESSION="${SESSIONS[0]}"; REMOTE_SESSION="${SESSIONS[1]}"
LOG="logs/E194/monitor/${TAG}_g1_expansion.log"; mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "[$(date '+%F %T')] watcher start tag=$TAG local=$LOCAL_SESSION remote=$REMOTE_SESSION"
while true; do
  local_active=0; remote_active=0
  tmux has-session -t "$LOCAL_SESSION" 2>/dev/null && local_active=1 || true
  if ssh -o BatchMode=yes -o ConnectTimeout=20 "$REMOTE" "tmux has-session -t '$REMOTE_SESSION' 2>/dev/null"; then
    remote_active=1
  else
    rc=$?; if [ "$rc" -eq 255 ]; then echo "remote unavailable; retry"; sleep 60; continue; fi
  fi
  echo "[$(date '+%F %T')] active local=$local_active remote=$remote_active"
  [ "$local_active" -eq 0 ] && [ "$remote_active" -eq 0 ] && break
  sleep 60
done
bash workspace/core4d/scripts/launch/active/pull_E194_G1_expansion_remote_Ada6000_results.sh "$TAG"
"$PYTHON_BIN" - "$MANIFEST" "$EXPECTED" <<'PY'
import csv,sys
from collections import Counter
from pathlib import Path
sys.path.insert(0,"workspace/core4d/scripts/experiments/E194")
import run_g1_expansion_queue as Q
rows=list(csv.DictReader(open(sys.argv[1]),delimiter="\t")); expected=int(sys.argv[2]); counts=Counter(r["status"] for r in rows)
failures={r["variant"]:Q.output_failures(r) for r in rows if r["status"]=="run_complete_pending_eval"}
bad={k:v for k,v in failures.items() if v}
print(f"rows={len(rows)} status={dict(counts)} validation_failures={bad}")
if len(rows)!=expected or counts!={"run_complete_pending_eval":expected} or bad:
 raise SystemExit("completion contract failed")
PY
if [ "$TAG" = full ]; then
  bash workspace/core4d/scripts/eval/wrappers/eval_E194_G1_expansion.sh
  "$PYTHON_BIN" workspace/core4d/scripts/eval/reports/gen_E194_G1_expansion_report.py
  bash workspace/core4d/scripts/launch/active/run_E194_G1_expansion_render_all.sh
  MUJOCO_GL=egl "$PYTHON_BIN" workspace/core4d/scripts/eval/reports/gen_E173_object_z_tracking_report.py
fi
echo "[$(date '+%F %T')] automatic $TAG closure complete"
