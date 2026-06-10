#!/usr/bin/env bash
# E152 box021 recovery: rerun the 2 incomplete box021 rows in parallel.
#   Local  GPU0 : E152_box021_029_p2_gateA
#   Remote GPU1 : E152_box021_029_p2_gateA_b1  (spider-remote, RTX 6000 Ada)
# Both box021 rows died in the original local-gpu0 split (only config_act.yaml,
# no trajectory). box004/box023 are already complete and untouched here.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-full}"
REMOTE="${REMOTE:-spider-remote}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/xiayb/pHRI_workspace/spider}"
REMOTE_GPU="${REMOTE_GPU:-1}"
LOCAL_GPU="${LOCAL_GPU:-0}"
LOCAL_VARIANT="${LOCAL_VARIANT:-E152_box021_029_p2_gateA}"
REMOTE_VARIANT="${REMOTE_VARIANT:-E152_box021_029_p2_gateA_b1}"
TRAIN="workspace/core4d/scripts/train/train_E152_axis1_hand_object_physics_gate.sh"
LSESSION="${LSESSION:-E152_box021_local_$(date +%H%M%S)}"
RSESSION="${RSESSION:-E152_box021_remote_$(date +%H%M%S)}"
TASK_DIR="example_datasets/processed/core4d/unitree_g1/humanoid_object/d003_box021_20231018_029_p2_e107_clean"

retry() {
  local attempt=1 max="${E152_REMOTE_RETRY_MAX:-4}"
  while true; do
    "$@" && return 0
    [ "$attempt" -ge "$max" ] && { echo "FAILED after ${attempt}: $*" >&2; return 1; }
    echo "Retry ${attempt}/${max}: $*" >&2; sleep "$((attempt * 3))"; attempt=$((attempt + 1))
  done
}

# --- 0. manifest + preflight (idempotent) ---
.venv/bin/python workspace/core4d/scripts/E152/build_axis1_hand_gate_manifest.py \
  --hand-gate-min-sdf-m -0.010 --hand-gate-max-violation-pct 0.05 >/tmp/e152_box021_manifest.log
.venv/bin/python workspace/core4d/scripts/E152/check_hand_gate_preflight.py >/tmp/e152_box021_preflight.log

# --- 1. sync box021 deps to remote (code increment + task/object/mask/override) ---
echo "Syncing box021 deps to ${REMOTE}:${REMOTE_ROOT}"
retry ssh "$REMOTE" "mkdir -p \
  '$REMOTE_ROOT/spider/optimizers' '$REMOTE_ROOT/spider/simulators' \
  '$REMOTE_ROOT/workspace/core4d/scripts/E152' '$REMOTE_ROOT/workspace/core4d/scripts/train' \
  '$REMOTE_ROOT/workspace/core4d/results/E152/axis1_hand_object_physics_gate/cem/${STAGE}' \
  '$REMOTE_ROOT/workspace/core4d/results/E143/contact_masks/box021_029_p2' \
  '$REMOTE_ROOT/$TASK_DIR' \
  '$REMOTE_ROOT/example_datasets/processed/core4d/assets/objects/box021' \
  '$REMOTE_ROOT/examples/config/override' '$REMOTE_ROOT/logs/E152/cem/${STAGE}'"

retry rsync -az spider/config.py "${REMOTE}:${REMOTE_ROOT}/spider/config.py"
retry rsync -az spider/simulators/mjwp.py "${REMOTE}:${REMOTE_ROOT}/spider/simulators/mjwp.py"
retry rsync -az spider/optimizers/sampling.py "${REMOTE}:${REMOTE_ROOT}/spider/optimizers/sampling.py"
retry rsync -az spider/optimizers/sampling_fast.py "${REMOTE}:${REMOTE_ROOT}/spider/optimizers/sampling_fast.py"
retry rsync -az workspace/core4d/scripts/E152/ "${REMOTE}:${REMOTE_ROOT}/workspace/core4d/scripts/E152/"
retry rsync -az "$TRAIN" "${REMOTE}:${REMOTE_ROOT}/$TRAIN"
retry rsync -az "$TASK_DIR/" "${REMOTE}:${REMOTE_ROOT}/$TASK_DIR/"
retry rsync -az example_datasets/processed/core4d/assets/objects/box021/ \
  "${REMOTE}:${REMOTE_ROOT}/example_datasets/processed/core4d/assets/objects/box021/"
retry rsync -az workspace/core4d/results/E143/contact_masks/box021_029_p2/ \
  "${REMOTE}:${REMOTE_ROOT}/workspace/core4d/results/E143/contact_masks/box021_029_p2/"
retry rsync -az examples/config/override/ "${REMOTE}:${REMOTE_ROOT}/examples/config/override/"

# --- 2. remote syntax check + launch gateA_b1 on remote GPU ---
retry ssh "$REMOTE" "cd '$REMOTE_ROOT' && \
  .venv/bin/python -m py_compile spider/config.py spider/simulators/mjwp.py \
    spider/optimizers/sampling.py spider/optimizers/sampling_fast.py && \
  bash -n $TRAIN"
retry ssh "$REMOTE" "cd '$REMOTE_ROOT' && \
  tmux new-session -d -s '$RSESSION' \"bash -lc 'set -euo pipefail; \
    echo E152 box021 remote $RSESSION variant=$REMOTE_VARIANT gpu=$REMOTE_GPU; \
    E152_ALLOW_MISSING_REUSE=1 bash $TRAIN single $STAGE $REMOTE_GPU $REMOTE_VARIANT; \
    echo E152_BOX021_REMOTE_DONE'\""
echo "Remote tmux: $RSESSION (GPU$REMOTE_GPU, $REMOTE_VARIANT)"

# --- 3. launch gateA on local GPU in tmux ---
tmux new-session -d -s "$LSESSION" "bash -lc 'set -euo pipefail; cd \"$(pwd)\"; \
  echo E152 box021 local $LSESSION variant=$LOCAL_VARIANT gpu=$LOCAL_GPU; \
  bash $TRAIN single $STAGE $LOCAL_GPU $LOCAL_VARIANT; \
  echo E152_BOX021_LOCAL_DONE'"
echo "Local  tmux: $LSESSION (GPU$LOCAL_GPU, $LOCAL_VARIANT)"
echo "Monitor local : tmux capture-pane -t $LSESSION -p | tail -40"
echo "Monitor remote: ssh $REMOTE \"tmux capture-pane -t $RSESSION -p | tail -40\""
