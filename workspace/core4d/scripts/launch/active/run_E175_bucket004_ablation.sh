#!/usr/bin/env bash
# E175 bucket004 five-arm CEM ablation.
#
# This host currently exposes one RTX 5090. The 20 cells run serially on GPU 0;
# the script intentionally does not kill, pause, or modify any other process.
#
# Usage:
#   bash workspace/core4d/scripts/launch/active/run_E175_bucket004_ablation.sh canary
#   bash workspace/core4d/scripts/launch/active/run_E175_bucket004_ablation.sh full
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-canary}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
GPU_ID="${GPU_ID:-0}"
ROOT="workspace/core4d/results/E175"
RUNNER="workspace/core4d/scripts/experiments/E175/run_cem_queue.py"
# This RTX 5090 host passes the EGL import probe; OSMesa is unavailable here.
export MUJOCO_GL="${MUJOCO_GL:-egl}"

if [ "$MODE" = "canary" ]; then
  MANIFEST="$ROOT/s6_downstream/manifests/bucket004_ablation_canary_manifest.tsv"
  NS="${NS:-64}"
  NI="${NI:-4}"
  RUN_MODE="canary"
elif [ "$MODE" = "full" ]; then
  MANIFEST="$ROOT/s6_downstream/manifests/bucket004_ablation_full_manifest.tsv"
  NS="${NS:-1024}"
  NI="${NI:-32}"
  RUN_MODE="full"
  "$PYTHON_BIN" - "$ROOT/s6_downstream/manifests/bucket004_ablation_canary_manifest.tsv" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
rows = list(csv.DictReader(path.open(), delimiter="\t"))
bad = [row["variant"] for row in rows if row["status"] != "run_complete_pending_eval"]
if len(rows) != 20 or bad:
    raise SystemExit(f"full blocked: canary rows={len(rows)} incomplete={bad}")
PY
else
  echo "MODE must be canary|full" >&2
  exit 2
fi

[ -f "$MANIFEST" ] || {
  echo "manifest missing: $MANIFEST" >&2
  exit 2
}

echo "=== E175 $MODE: 20 cells serial on visible GPU $GPU_ID ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader

"$PYTHON_BIN" "$RUNNER" \
  --mode "$RUN_MODE" \
  --manifest-tsv "$MANIFEST" \
  --gpu-id "$GPU_ID" \
  --num-samples "$NS" \
  --max-num-iterations "$NI" \
  --all

"$PYTHON_BIN" - "$MANIFEST" <<'PY'
import collections
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
rows = list(csv.DictReader(path.open(), delimiter="\t"))
status = collections.Counter(row.get("status", "") for row in rows)
print("rows:", len(rows), "status:", dict(status))
if len(rows) != 20 or status != {"run_complete_pending_eval": 20}:
    raise SystemExit("E175 manifest terminal contract failed")
PY

echo "=== E175 $MODE complete ==="
