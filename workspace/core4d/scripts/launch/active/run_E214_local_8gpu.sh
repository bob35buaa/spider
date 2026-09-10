#!/usr/bin/env bash
# E214 four-ablation CEM -- local 8-GPU entry (no remote / no shard).
#
# Step 1 (rule 7): snapshot the baseline scene XMLs used.
# Step 2: build the 200-row manifest (idempotent).
# Step 3: dispatch all 200 runs across 8 local GPUs (flock single-instance).
#
# The runner is resumable: re-invoking skips cem_ok rows and re-picks failed/
# pending ones.  Runs in the foreground here; wrap with nohup/tmux for a long run:
#   nohup bash workspace/core4d/scripts/launch/active/run_E214_local_8gpu.sh > logs/E214/run.log 2>&1 &
#
# Optional args are forwarded to run_ablation_cem.py, e.g.:
#   ... --ablations A1_contactHDMI_only,A3_softPenalty_only,A4_hardGate_only
#   ... --cases box021_20231011_034_p1 --limit 4
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PY=.venv/bin/python
E="workspace/core4d/scripts/experiments/E214"
GPUS="${E214_GPUS:-0,1,2,3,4,5,6,7}"

echo "=== [E214] step 1: snapshot baseline scenes ==="
bash "$E/snapshot_E214_scenes.sh"

echo "=== [E214] step 2: build manifest ==="
$PY "$E/build_manifest.py"

echo "=== [E214] step 3: dispatch 8-GPU CEM (gpus=$GPUS) ==="
$PY "$E/run_ablation_cem.py" --gpus "$GPUS" "$@"

echo "=== [E214] done. audit single-variable diffs: ==="
$PY "$E/audit_single_variable.py" || true
