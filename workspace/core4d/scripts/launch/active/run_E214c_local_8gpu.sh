#!/usr/bin/env bash
# E214c part-wise-threshold ablation CEM -- local 8-GPU entry (no remote / no shard).
#
# Step 1 (rule 7): snapshot the baseline scene XMLs used.
# Step 2: CPU contract test (fail fast before any GPU).
# Step 3: build the 100-row manifest (idempotent).
# Step 4: dispatch all 100 runs across 8 local GPUs (flock single-instance).
# Step 5: single-variable audit.
#
# The runner is resumable: re-invoking skips cem_ok rows and re-picks failed/pending.
# Runs in the foreground here; wrap with nohup/tmux for a long run:
#   nohup bash workspace/core4d/scripts/launch/active/run_E214c_local_8gpu.sh > logs/E214c/run.log 2>&1 &
#
# NOTE: kill the GPU keepalive script before launching, restart it after (see memory).
#
# Optional args are forwarded to run_ablation_cem.py, e.g.:
#   ... --ablations A5_uniform_hand --cases box021_20231011_034_p1 --limit 2   # smoke
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PY=.venv/bin/python
E="workspace/core4d/scripts/experiments/E214c"
GPUS="${E214C_GPUS:-0,1,2,3,4,5,6,7}"

echo "=== [E214c] step 1: snapshot baseline scenes ==="
bash "$E/snapshot_E214c_scenes.sh"

echo "=== [E214c] step 2: CPU contract test ==="
$PY "$E/test_e214c_contract.py"

echo "=== [E214c] step 3: build manifest ==="
$PY "$E/build_manifest.py"

echo "=== [E214c] step 4: dispatch 8-GPU CEM (gpus=$GPUS) ==="
$PY "$E/run_ablation_cem.py" --gpus "$GPUS" "$@"

echo "=== [E214c] step 5: audit single-variable diffs ==="
$PY "$E/audit_single_variable.py" || true
