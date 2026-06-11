#!/usr/bin/env bash
# E153 remote launcher: run gate-threshold sweep on remote GPU0/GPU1.
#   GPU0: box004_083_p2 | GPU1: box023_person2 (box021 runs locally, not here).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-full}"
REMOTE="${REMOTE:-spider-remote}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/xiayb/pHRI_workspace/spider}"
SESSION="${SESSION:-E153_${STAGE}_$(date +%H%M%S)}"

if [ "$STAGE" != "smoke" ] && [ "$STAGE" != "full" ]; then
  echo "Invalid STAGE=$STAGE (use smoke|full)" >&2
  exit 2
fi

retry() {
  local attempt=1
  local max_attempts="${E153_REMOTE_RETRY_MAX:-4}"
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

# Hardcoded E153 case paths (no variants.tsv; box021 is local-only).
TASK_DIRS=(
  "example_datasets/processed/core4d/unitree_g1/humanoid_object/e091_box004_20231003_2_083_p2_e092_dyn"
  "example_datasets/processed/core4d/unitree_g1/humanoid_object/box023_person2_legobj"
)
OBJECT_DIRS=(
  "example_datasets/processed/core4d/assets/objects/box004"
  "example_datasets/processed/core4d/assets/objects/box023"
)
MASK_DIRS=(
  "workspace/core4d/results/E143/contact_masks/box004_083_p2"
  "workspace/core4d/results/E143/contact_masks/box023_person2"
)

echo "Syncing E153 artifacts to ${REMOTE}:${REMOTE_ROOT}"
retry ssh "$REMOTE" "mkdir -p \
  '$REMOTE_ROOT/spider/optimizers' \
  '$REMOTE_ROOT/spider/simulators' \
  '$REMOTE_ROOT/workspace/core4d/scripts/E153' \
  '$REMOTE_ROOT/workspace/core4d/scripts/convert' \
  '$REMOTE_ROOT/workspace/core4d/results/E153/gate_threshold_sweep/cem/${STAGE}' \
  '$REMOTE_ROOT/workspace/core4d/results/E143/contact_masks' \
  '$REMOTE_ROOT/logs/E153/cem/${STAGE}' \
  '$REMOTE_ROOT/example_datasets/processed/core4d/assets/objects' \
  '$REMOTE_ROOT/example_datasets/processed/core4d/unitree_g1/humanoid_object' \
  '$REMOTE_ROOT/examples/config/override'"

# Code increments (remote is rsync-pushed, not git-pulled).
retry rsync -az spider/config.py "${REMOTE}:${REMOTE_ROOT}/spider/config.py"
retry rsync -az spider/simulators/mjwp.py "${REMOTE}:${REMOTE_ROOT}/spider/simulators/mjwp.py"
retry rsync -az spider/optimizers/sampling.py "${REMOTE}:${REMOTE_ROOT}/spider/optimizers/sampling.py"
retry rsync -az spider/optimizers/sampling_fast.py "${REMOTE}:${REMOTE_ROOT}/spider/optimizers/sampling_fast.py"
retry rsync -az workspace/core4d/scripts/E153/ "${REMOTE}:${REMOTE_ROOT}/workspace/core4d/scripts/E153/"
retry rsync -az workspace/core4d/scripts/convert/snapshot_scenes.sh "${REMOTE}:${REMOTE_ROOT}/workspace/core4d/scripts/convert/snapshot_scenes.sh"
retry rsync -az examples/config/override/ "${REMOTE}:${REMOTE_ROOT}/examples/config/override/"

# Per-case task dirs (scene xml etc).
for dir in "${TASK_DIRS[@]}"; do
  [ -d "$dir" ] || { echo "Missing task dir: $dir" >&2; exit 1; }
  retry ssh "$REMOTE" "mkdir -p '$REMOTE_ROOT/$dir'"
  retry rsync -az "$dir/" "${REMOTE}:${REMOTE_ROOT}/${dir}/"
done

# Object assets.
for dir in "${OBJECT_DIRS[@]}"; do
  [ -d "$dir" ] || { echo "Missing object dir: $dir" >&2; exit 1; }
  retry ssh "$REMOTE" "mkdir -p '$REMOTE_ROOT/$dir'"
  retry rsync -az "$dir/" "${REMOTE}:${REMOTE_ROOT}/${dir}/"
done

# Contact masks.
for dir in "${MASK_DIRS[@]}"; do
  [ -d "$dir" ] || { echo "Missing mask dir: $dir" >&2; exit 1; }
  retry ssh "$REMOTE" "mkdir -p '$REMOTE_ROOT/$dir'"
  retry rsync -az "$dir/" "${REMOTE}:${REMOTE_ROOT}/${dir}/"
done

# Preflight: compile changed spider code, syntax-check the two case scripts.
retry ssh "$REMOTE" "cd '$REMOTE_ROOT' && \
  .venv/bin/python -m py_compile \
    spider/config.py spider/simulators/mjwp.py spider/optimizers/sampling.py spider/optimizers/sampling_fast.py && \
  bash -n workspace/core4d/scripts/E153/run_case_box004.sh && \
  bash -n workspace/core4d/scripts/E153/run_case_box023.sh && \
  echo preflight-ok"

# Launch both cases in parallel inside one tmux session: box004 on GPU0, box023 on GPU1.
retry ssh "$REMOTE" "cd '$REMOTE_ROOT' && \
  tmux new-session -d -s '$SESSION' \"bash -lc 'set -euo pipefail; \
    echo E153 remote session $SESSION stage=$STAGE; \
    (bash workspace/core4d/scripts/E153/run_case_box004.sh 0 $STAGE) & PID0=\\\$!; \
    (bash workspace/core4d/scripts/E153/run_case_box023.sh 1 $STAGE) & PID1=\\\$!; \
    echo launched PID0=\\\$PID0 PID1=\\\$PID1; \
    wait \\\$PID0; echo box004 remote-gpu0 done; \
    wait \\\$PID1; echo box023 remote-gpu1 done; \
    echo E153 remote complete'\""

echo "Started remote tmux session: ${SESSION}"
echo "Monitor: ssh ${REMOTE} \"tmux capture-pane -t ${SESSION} -p | tail -80\""
