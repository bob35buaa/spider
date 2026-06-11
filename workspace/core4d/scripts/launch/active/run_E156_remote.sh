#!/usr/bin/env bash
# E156 remote launcher for clean8 +gateA / E155_decay CEM rows.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-full}"
REMOTE="${REMOTE:-spider-remote}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/xiayb/pHRI_workspace/spider}"
SESSION="${SESSION:-E156_${STAGE}_$(date +%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
REMOTE_GPU0_WORK="${REMOTE_GPU0_WORK:-}"
REMOTE_GPU1_WORK="${REMOTE_GPU1_WORK:-}"

if [ "$STAGE" != "smoke" ] && [ "$STAGE" != "full" ]; then
  echo "Invalid STAGE=$STAGE (use smoke|full)" >&2
  exit 2
fi

"$PYTHON_BIN" workspace/core4d/scripts/experiments/E156/build_clean8_gate_decay_manifest.py --legacy-copy >/dev/null

retry() {
  local attempt=1
  local max_attempts="${E156_REMOTE_RETRY_MAX:-4}"
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
rows = list(csv.DictReader(open("workspace/core4d/scripts/experiments/E156/variants.tsv", newline=""), delimiter="\t"))
for p in sorted({str(Path(r["rubber_scene_act"]).parent) for r in rows}):
    print(p)
PY
)
mapfile -t OBJECT_DIRS < <("$PYTHON_BIN" - <<'PY'
import csv
from pathlib import Path
rows = list(csv.DictReader(open("workspace/core4d/scripts/experiments/E156/variants.tsv", newline=""), delimiter="\t"))
for p in sorted({str(Path(r["object_asset"]).parent) for r in rows}):
    print(p)
PY
)
mapfile -t MASK_DIRS < <("$PYTHON_BIN" - <<'PY'
import csv
from pathlib import Path
rows = list(csv.DictReader(open("workspace/core4d/scripts/experiments/E156/variants.tsv", newline=""), delimiter="\t"))
for p in sorted({str(Path(r["mask_path"]).parent) for r in rows}):
    print(p)
PY
)

echo "Syncing E156 artifacts to ${REMOTE}:${REMOTE_ROOT}"
retry ssh "$REMOTE" "mkdir -p \
  '$REMOTE_ROOT/spider/optimizers' \
  '$REMOTE_ROOT/spider/simulators' \
  '$REMOTE_ROOT/examples/config/override' \
  '$REMOTE_ROOT/workspace/core4d/scripts/E156' \
  '$REMOTE_ROOT/workspace/core4d/scripts/experiments/E156' \
  '$REMOTE_ROOT/workspace/core4d/scripts/launch/active' \
  '$REMOTE_ROOT/workspace/core4d/results/E156/clean8_gate_decay/cem/${STAGE}' \
  '$REMOTE_ROOT/workspace/core4d/results/E143/contact_masks' \
  '$REMOTE_ROOT/logs/E156/cem/${STAGE}' \
  '$REMOTE_ROOT/example_datasets/processed/core4d/assets/objects' \
  '$REMOTE_ROOT/example_datasets/processed/core4d/unitree_g1/humanoid_object'"

retry rsync -az examples/run_mjwp.py "${REMOTE}:${REMOTE_ROOT}/examples/run_mjwp.py"
retry rsync -az spider/config.py "${REMOTE}:${REMOTE_ROOT}/spider/config.py"
retry rsync -az spider/simulators/mjwp.py "${REMOTE}:${REMOTE_ROOT}/spider/simulators/mjwp.py"
retry rsync -az spider/optimizers/sampling.py "${REMOTE}:${REMOTE_ROOT}/spider/optimizers/sampling.py"
retry rsync -az spider/optimizers/sampling_fast.py "${REMOTE}:${REMOTE_ROOT}/spider/optimizers/sampling_fast.py"
retry rsync -az workspace/core4d/scripts/experiments/E156/ "${REMOTE}:${REMOTE_ROOT}/workspace/core4d/scripts/experiments/E156/"
retry rsync -az workspace/core4d/scripts/E156/ "${REMOTE}:${REMOTE_ROOT}/workspace/core4d/scripts/E156/"
retry rsync -az workspace/core4d/scripts/launch/active/run_E156_local.sh "${REMOTE}:${REMOTE_ROOT}/workspace/core4d/scripts/launch/active/run_E156_local.sh"
retry rsync -az examples/config/override/core4d_E1*.yaml "${REMOTE}:${REMOTE_ROOT}/examples/config/override/"

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
    workspace/core4d/scripts/experiments/E156/build_clean8_gate_decay_manifest.py && \
  bash -n workspace/core4d/scripts/launch/active/run_E156_local.sh && \
  echo preflight-ok"

retry ssh "$REMOTE" "cd '$REMOTE_ROOT' && \
  tmux new-session -d -s '$SESSION' \"bash -lc 'set -euo pipefail; \
    echo E156 remote session $SESSION stage=$STAGE; \
    (E156_SPLIT=remote-gpu0 CASE_METHODS=\\\"$REMOTE_GPU0_WORK\\\" LOCAL_GPU=0 bash workspace/core4d/scripts/launch/active/run_E156_local.sh $STAGE) & PID0=\\\$!; \
    (E156_SPLIT=remote-gpu1 CASE_METHODS=\\\"$REMOTE_GPU1_WORK\\\" LOCAL_GPU=1 bash workspace/core4d/scripts/launch/active/run_E156_local.sh $STAGE) & PID1=\\\$!; \
    echo launched PID0=\\\$PID0 PID1=\\\$PID1; \
    wait \\\$PID0; echo remote-gpu0 done; \
    wait \\\$PID1; echo remote-gpu1 done; \
    echo E156 remote complete'\""

echo "Started remote tmux session: ${SESSION}"
echo "Remote GPU0 work: ${REMOTE_GPU0_WORK:-split remote-gpu0}"
echo "Remote GPU1 work: ${REMOTE_GPU1_WORK:-split remote-gpu1}"
echo "Monitor: ssh ${REMOTE} \"tmux capture-pane -t ${SESSION} -p | tail -80\""
