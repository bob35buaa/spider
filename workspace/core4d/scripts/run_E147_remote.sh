#!/usr/bin/env bash
# E147 remote launcher: sync rubber_hull artifacts and run full CEM on remote GPU0/GPU1.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-full}"
REMOTE="${REMOTE:-spider-remote}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/xiayb/pHRI_workspace/spider}"
SESSION="${SESSION:-E147_${STAGE}_rubber_$(date +%H%M%S)}"
VARIANTS_FILE="${VARIANTS_FILE:-workspace/core4d/scripts/E147/variants.tsv}"

if [ "$STAGE" != "smoke" ] && [ "$STAGE" != "full" ]; then
  echo "Invalid STAGE=$STAGE (use smoke|full)" >&2
  exit 2
fi

retry() {
  local attempt=1
  local max_attempts="${E147_REMOTE_RETRY_MAX:-4}"
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

python workspace/core4d/scripts/E147/build_rubber_hand_collision_manifest.py >/tmp/e147_manifest_build.log

echo "Syncing E147 artifacts to ${REMOTE}:${REMOTE_ROOT}"
retry ssh "$REMOTE" "mkdir -p \
  '$REMOTE_ROOT/workspace/core4d/scripts/E147' \
  '$REMOTE_ROOT/workspace/core4d/scripts/train' \
  '$REMOTE_ROOT/workspace/core4d/results/E147/rubber_hand_collision/cem/${STAGE}' \
  '$REMOTE_ROOT/logs/E147/cem/${STAGE}' \
  '$REMOTE_ROOT/example_datasets/processed/core4d/assets/objects' \
  '$REMOTE_ROOT/example_datasets/processed/core4d/unitree_g1/humanoid_object' \
  '$REMOTE_ROOT/examples/config/override'"

retry rsync -az workspace/core4d/scripts/E147/ "${REMOTE}:${REMOTE_ROOT}/workspace/core4d/scripts/E147/"
retry rsync -az workspace/core4d/scripts/train/train_E147_rubber_hand_collision.sh "${REMOTE}:${REMOTE_ROOT}/workspace/core4d/scripts/train/"
retry rsync -az workspace/core4d/results/E147/rubber_hand_collision/ "${REMOTE}:${REMOTE_ROOT}/workspace/core4d/results/E147/rubber_hand_collision/"

mapfile -t overrides < <(awk -F '\t' 'NF && $1 !~ /^#/ && NR > 1 {print $24}' "$VARIANTS_FILE" | sort -u)
for override in "${overrides[@]}"; do
  retry rsync -az "$override" "${REMOTE}:${REMOTE_ROOT}/$override"
done

mapfile -t tasks < <(awk -F '\t' 'NF && $1 !~ /^#/ && NR > 1 {print $13}' "$VARIANTS_FILE" | sort -u)
for task in "${tasks[@]}"; do
  retry rsync -az "example_datasets/processed/core4d/unitree_g1/humanoid_object/${task}/" \
    "${REMOTE}:${REMOTE_ROOT}/example_datasets/processed/core4d/unitree_g1/humanoid_object/${task}/"
done

mapfile -t object_keys < <(awk -F '\t' 'NF && $1 !~ /^#/ && NR > 1 {print $4}' "$VARIANTS_FILE" | sort -u)
for object_key in "${object_keys[@]}"; do
  retry rsync -az "example_datasets/processed/core4d/assets/objects/${object_key}/" \
    "${REMOTE}:${REMOTE_ROOT}/example_datasets/processed/core4d/assets/objects/${object_key}/"
done

retry ssh "$REMOTE" "cd '$REMOTE_ROOT' && \
  bash -n workspace/core4d/scripts/train/train_E147_rubber_hand_collision.sh && \
  bash workspace/core4d/scripts/train/train_E147_rubber_hand_collision.sh list '$STAGE' 0 remote-gpu0 >/tmp/e147_gpu0.list && \
  bash workspace/core4d/scripts/train/train_E147_rubber_hand_collision.sh list '$STAGE' 0 remote-gpu1 >/tmp/e147_gpu1.list && \
  echo remote-gpu0=\$(wc -l </tmp/e147_gpu0.list) remote-gpu1=\$(wc -l </tmp/e147_gpu1.list)"

retry ssh "$REMOTE" "cd '$REMOTE_ROOT' && \
  tmux new-session -d -s '$SESSION' \"bash -lc 'set -euo pipefail; \
    echo E147 remote session $SESSION stage=$STAGE; \
    (bash workspace/core4d/scripts/train/train_E147_rubber_hand_collision.sh remote-gpu0 $STAGE 0) & PID0=\\\$!; \
    (bash workspace/core4d/scripts/train/train_E147_rubber_hand_collision.sh remote-gpu1 $STAGE 1) & PID1=\\\$!; \
    echo launched PID0=\\\$PID0 PID1=\\\$PID1; \
    wait \\\$PID0; echo remote-gpu0 done; \
    wait \\\$PID1; echo remote-gpu1 done; \
    echo E147 remote complete'\""

echo "Started remote tmux session: ${SESSION}"
echo "Monitor: ssh ${REMOTE} \"tmux capture-pane -t ${SESSION} -p | tail -40\""
