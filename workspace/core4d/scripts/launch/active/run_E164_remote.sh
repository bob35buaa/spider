#!/usr/bin/env bash
# E164 remote launcher for bimanual global mask + bimanual reward.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-full}"
REMOTE="${REMOTE:-spider-remote}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/xiayb/pHRI_workspace/spider}"
SESSION="${SESSION:-E164_${STAGE}_$(date +%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
REMOTE_GPU0_WORK="${REMOTE_GPU0_WORK:-}"
REMOTE_GPU1_WORK="${REMOTE_GPU1_WORK:-}"
WAIT_FOR_GPU_IDLE="${WAIT_FOR_GPU_IDLE:-0}"
E164_GPU_IDLE_MAX_MEM_MB="${E164_GPU_IDLE_MAX_MEM_MB:-3000}"
E164_GPU_IDLE_MAX_UTIL_PCT="${E164_GPU_IDLE_MAX_UTIL_PCT:-20}"
E164_GPU_IDLE_POLL_SEC="${E164_GPU_IDLE_POLL_SEC:-120}"
E164_GPU_IDLE_STABLE_POLLS="${E164_GPU_IDLE_STABLE_POLLS:-2}"

if [ "$STAGE" != "smoke" ] && [ "$STAGE" != "full" ]; then
  echo "Invalid STAGE=$STAGE (use smoke|full)" >&2
  exit 2
fi

"$PYTHON_BIN" workspace/core4d/scripts/experiments/E164/build_bimanual_global_mask_reward_manifest.py >/dev/null

retry() {
  local attempt=1
  local max_attempts="${E164_REMOTE_RETRY_MAX:-4}"
  while true; do
    if "$@"; then
      return 0
    fi
    if [ "$attempt" -ge "$max_attempts" ]; then
      echo "Command failed after ${attempt} attempts: $*" >&2
      return 1
    fi
    echo "Retry ${attempt}/${max_attempts}: $*" >&2
    sleep "$((attempt * 3))"
    attempt=$((attempt + 1))
  done
}

mapfile -t TASK_DIRS < <("$PYTHON_BIN" - <<'PY'
import csv
from pathlib import Path
rows = list(csv.DictReader(open("workspace/core4d/scripts/experiments/E164/variants.tsv", newline=""), delimiter="\t"))
for p in sorted({str(Path(r["rubber_scene_act"]).parent) for r in rows}):
    print(p)
PY
)
mapfile -t OBJECT_DIRS < <("$PYTHON_BIN" - <<'PY'
import csv
from pathlib import Path
rows = list(csv.DictReader(open("workspace/core4d/scripts/experiments/E164/variants.tsv", newline=""), delimiter="\t"))
for p in sorted({str(Path(r["object_asset"]).parent) for r in rows}):
    print(p)
PY
)
mapfile -t MASK_DIRS < <("$PYTHON_BIN" - <<'PY'
import csv
from pathlib import Path
rows = list(csv.DictReader(open("workspace/core4d/scripts/experiments/E164/variants.tsv", newline=""), delimiter="\t"))
for p in sorted({str(Path(r["mask_path"]).parent) for r in rows}):
    print(p)
PY
)

echo "Syncing E164 artifacts to ${REMOTE}:${REMOTE_ROOT}"
retry ssh "$REMOTE" "mkdir -p \
  '$REMOTE_ROOT/spider/simulators' \
  '$REMOTE_ROOT/spider/optimizers' \
  '$REMOTE_ROOT/examples/config/override' \
  '$REMOTE_ROOT/workspace/core4d/scripts/experiments/E164' \
  '$REMOTE_ROOT/workspace/core4d/scripts/experiments/E156' \
  '$REMOTE_ROOT/workspace/core4d/scripts/launch/active' \
  '$REMOTE_ROOT/workspace/core4d/results/E164/bimanual_global_mask_reward/cem/${STAGE}' \
  '$REMOTE_ROOT/logs/E164/cem/${STAGE}'"

retry rsync -az examples/run_mjwp.py "${REMOTE}:${REMOTE_ROOT}/examples/run_mjwp.py"
retry rsync -az spider/config.py "${REMOTE}:${REMOTE_ROOT}/spider/config.py"
retry rsync -az spider/simulators/mjwp.py "${REMOTE}:${REMOTE_ROOT}/spider/simulators/mjwp.py"
retry rsync -az spider/optimizers/sampling.py "${REMOTE}:${REMOTE_ROOT}/spider/optimizers/sampling.py"
retry rsync -az spider/optimizers/sampling_fast.py "${REMOTE}:${REMOTE_ROOT}/spider/optimizers/sampling_fast.py"
retry rsync -az workspace/core4d/scripts/experiments/E164/ "${REMOTE}:${REMOTE_ROOT}/workspace/core4d/scripts/experiments/E164/"
retry rsync -az workspace/core4d/scripts/experiments/E156/ "${REMOTE}:${REMOTE_ROOT}/workspace/core4d/scripts/experiments/E156/"
retry rsync -az workspace/core4d/scripts/launch/active/run_E164_local.sh "${REMOTE}:${REMOTE_ROOT}/workspace/core4d/scripts/launch/active/run_E164_local.sh"
retry rsync -az examples/config/override/core4d_E164_*.yaml "${REMOTE}:${REMOTE_ROOT}/examples/config/override/"
retry rsync -az examples/config/override/core4d_E14*.yaml examples/config/override/core4d_E15*.yaml examples/config/override/core4d_E160_*.yaml "${REMOTE}:${REMOTE_ROOT}/examples/config/override/"

for dir in "${TASK_DIRS[@]}"; do
  [ -d "$dir" ] || { echo "Missing task dir: $dir" >&2; exit 1; }
  retry ssh "$REMOTE" "mkdir -p '$REMOTE_ROOT/$dir'"
  retry rsync -az "$dir/" "${REMOTE}:${REMOTE_ROOT}/${dir}/"
done

for dir in "${OBJECT_DIRS[@]}"; do
  [ -d "$dir" ] || { echo "Missing object dir: $dir" >&2; exit 1; }
  retry ssh "$REMOTE" "mkdir -p '$REMOTE_ROOT/$dir'"
  retry rsync -az "$dir/" "${REMOTE}:${REMOTE_ROOT}/${dir}/"
done

for dir in "${MASK_DIRS[@]}"; do
  [ -d "$dir" ] || { echo "Missing mask dir: $dir" >&2; exit 1; }
  retry ssh "$REMOTE" "mkdir -p '$REMOTE_ROOT/$dir'"
  retry rsync -az "$dir/" "${REMOTE}:${REMOTE_ROOT}/${dir}/"
done

retry ssh "$REMOTE" "cd '$REMOTE_ROOT' && \
  .venv/bin/python -m py_compile \
    examples/run_mjwp.py spider/config.py spider/simulators/mjwp.py \
    spider/optimizers/sampling.py spider/optimizers/sampling_fast.py \
    workspace/core4d/scripts/experiments/E164/build_bimanual_global_masks.py \
    workspace/core4d/scripts/experiments/E164/build_bimanual_global_mask_reward_manifest.py && \
  bash -n workspace/core4d/scripts/launch/active/run_E164_local.sh && \
  echo preflight-ok"

retry ssh "$REMOTE" "cd '$REMOTE_ROOT' && \
  tmux new-session -d -s '$SESSION' \"bash -lc 'set -euo pipefail; \
    echo E164 remote session $SESSION stage=$STAGE; \
    (WAIT_FOR_GPU_IDLE=$WAIT_FOR_GPU_IDLE E164_GPU_IDLE_MAX_MEM_MB=$E164_GPU_IDLE_MAX_MEM_MB E164_GPU_IDLE_MAX_UTIL_PCT=$E164_GPU_IDLE_MAX_UTIL_PCT E164_GPU_IDLE_POLL_SEC=$E164_GPU_IDLE_POLL_SEC E164_GPU_IDLE_STABLE_POLLS=$E164_GPU_IDLE_STABLE_POLLS E164_SPLIT=remote-gpu0 CASE_METHODS=\\\"$REMOTE_GPU0_WORK\\\" LOCAL_GPU=0 bash workspace/core4d/scripts/launch/active/run_E164_local.sh $STAGE) & PID0=\\\$!; \
    (WAIT_FOR_GPU_IDLE=$WAIT_FOR_GPU_IDLE E164_GPU_IDLE_MAX_MEM_MB=$E164_GPU_IDLE_MAX_MEM_MB E164_GPU_IDLE_MAX_UTIL_PCT=$E164_GPU_IDLE_MAX_UTIL_PCT E164_GPU_IDLE_POLL_SEC=$E164_GPU_IDLE_POLL_SEC E164_GPU_IDLE_STABLE_POLLS=$E164_GPU_IDLE_STABLE_POLLS E164_SPLIT=remote-gpu1 CASE_METHODS=\\\"$REMOTE_GPU1_WORK\\\" LOCAL_GPU=1 bash workspace/core4d/scripts/launch/active/run_E164_local.sh $STAGE) & PID1=\\\$!; \
    echo launched PID0=\\\$PID0 PID1=\\\$PID1; \
    wait \\\$PID0; echo remote-gpu0 done; \
    wait \\\$PID1; echo remote-gpu1 done; \
    echo E164 remote complete'\""

echo "Started remote tmux session: ${SESSION}"
echo "Remote GPU0 work: ${REMOTE_GPU0_WORK:-split remote-gpu0}"
echo "Remote GPU1 work: ${REMOTE_GPU1_WORK:-split remote-gpu1}"
echo "Monitor: ssh ${REMOTE} \"tmux capture-pane -t ${SESSION} -p | tail -80\""
