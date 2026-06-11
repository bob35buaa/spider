#!/usr/bin/env bash
# E155 remote launcher for remaining release-smoothing variants.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-full}"
REMOTE="${REMOTE:-spider-remote}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/xiayb/pHRI_workspace/spider}"
SESSION="${SESSION:-E155_${STAGE}_$(date +%H%M%S)}"
REMOTE_GPU0_WORK="${REMOTE_GPU0_WORK:-box004_083_p2:ramp10}"
REMOTE_GPU1_WORK="${REMOTE_GPU1_WORK:-box023_person2:decay box023_person2:neutral}"

if [ "$STAGE" != "smoke" ] && [ "$STAGE" != "full" ]; then
  echo "Invalid STAGE=$STAGE (use smoke|full)" >&2
  exit 2
fi

retry() {
  local attempt=1
  local max_attempts="${E155_REMOTE_RETRY_MAX:-4}"
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

echo "Syncing E155 artifacts to ${REMOTE}:${REMOTE_ROOT}"
retry ssh "$REMOTE" "mkdir -p \
  '$REMOTE_ROOT/spider/optimizers' \
  '$REMOTE_ROOT/spider/simulators' \
  '$REMOTE_ROOT/workspace/core4d/scripts/E155' \
  '$REMOTE_ROOT/workspace/core4d/scripts/launch/active' \
  '$REMOTE_ROOT/workspace/core4d/results/E155/cem/${STAGE}' \
  '$REMOTE_ROOT/workspace/core4d/results/E143/contact_masks' \
  '$REMOTE_ROOT/logs/E155/cem/${STAGE}' \
  '$REMOTE_ROOT/example_datasets/processed/core4d/assets/objects' \
  '$REMOTE_ROOT/example_datasets/processed/core4d/unitree_g1/humanoid_object' \
  '$REMOTE_ROOT/examples/config/override'"

retry rsync -az spider/config.py "${REMOTE}:${REMOTE_ROOT}/spider/config.py"
retry rsync -az spider/simulators/mjwp.py "${REMOTE}:${REMOTE_ROOT}/spider/simulators/mjwp.py"
retry rsync -az spider/optimizers/sampling.py "${REMOTE}:${REMOTE_ROOT}/spider/optimizers/sampling.py"
retry rsync -az spider/optimizers/sampling_fast.py "${REMOTE}:${REMOTE_ROOT}/spider/optimizers/sampling_fast.py"
retry rsync -az workspace/core4d/scripts/E155/ "${REMOTE}:${REMOTE_ROOT}/workspace/core4d/scripts/E155/"
retry rsync -az workspace/core4d/scripts/launch/active/run_E155_local.sh "${REMOTE}:${REMOTE_ROOT}/workspace/core4d/scripts/launch/active/run_E155_local.sh"
retry rsync -az examples/config/override/core4d_E152_box004_083_p2_gateA_b1.yaml "${REMOTE}:${REMOTE_ROOT}/examples/config/override/"
retry rsync -az examples/config/override/core4d_E152_box023_person2_gateA_b1.yaml "${REMOTE}:${REMOTE_ROOT}/examples/config/override/"

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
    spider/config.py spider/simulators/mjwp.py spider/optimizers/sampling.py spider/optimizers/sampling_fast.py && \
  bash -n workspace/core4d/scripts/launch/active/run_E155_local.sh && \
  echo preflight-ok"

retry ssh "$REMOTE" "cd '$REMOTE_ROOT' && \
  tmux new-session -d -s '$SESSION' \"bash -lc 'set -euo pipefail; \
    echo E155 remote session $SESSION stage=$STAGE; \
    (CASE_METHODS=\\\"$REMOTE_GPU0_WORK\\\" LOCAL_GPU=0 bash workspace/core4d/scripts/launch/active/run_E155_local.sh $STAGE) & PID0=\\\$!; \
    (CASE_METHODS=\\\"$REMOTE_GPU1_WORK\\\" LOCAL_GPU=1 bash workspace/core4d/scripts/launch/active/run_E155_local.sh $STAGE) & PID1=\\\$!; \
    echo launched PID0=\\\$PID0 PID1=\\\$PID1; \
    wait \\\$PID0; echo remote-gpu0 done; \
    wait \\\$PID1; echo remote-gpu1 done; \
    echo E155 remote complete'\""

echo "Started remote tmux session: ${SESSION}"
echo "Remote GPU0 work: ${REMOTE_GPU0_WORK}"
echo "Remote GPU1 work: ${REMOTE_GPU1_WORK}"
echo "Monitor: ssh ${REMOTE} \"tmux capture-pane -t ${SESSION} -p | tail -80\""
