#!/usr/bin/env bash
# E145 remote runner: launch remote GPU0/GPU1 high-priority full-CEM splits.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-full}"
REMOTE="${REMOTE:-spider-remote}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/xiayb/pHRI_workspace/spider}"
SESSION="${SESSION:-E145_${STAGE}_high_$(date +%H%M%S)}"
VARIANTS_FILE="${VARIANTS_FILE:-workspace/core4d/scripts/E145/variants_phase2_high.tsv}"

if [ "$STAGE" != "smoke" ] && [ "$STAGE" != "full" ]; then
  echo "Invalid STAGE=$STAGE (use smoke|full)" >&2
  exit 2
fi

retry() {
  local attempt=1
  local max_attempts="${E145_REMOTE_RETRY_MAX:-4}"
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

variant_count="$(awk -F '\t' 'NF && $1 !~ /^#/ && NR > 2 {n++} END {print n+0}' "$VARIANTS_FILE")"
if [ "$variant_count" -ne 9 ]; then
  echo "E145 high-priority manifest must have 9 rows, got ${variant_count}: ${VARIANTS_FILE}" >&2
  exit 2
fi

echo "Syncing E145 high-priority artifacts to ${REMOTE}:${REMOTE_ROOT}"
retry ssh "$REMOTE" "mkdir -p \
  '$REMOTE_ROOT/workspace/core4d/scripts/E145' \
  '$REMOTE_ROOT/workspace/core4d/scripts/train' \
  '$REMOTE_ROOT/workspace/core4d/scripts/eval' \
  '$REMOTE_ROOT/workspace/core4d/results/E145/full_nonbox_to_rl_ready/cem_ready' \
  '$REMOTE_ROOT/workspace/core4d/results/E145/full_nonbox_to_rl_ready/contact_masks' \
  '$REMOTE_ROOT/workspace/core4d/results/E145/full_nonbox_to_rl_ready/cem/${STAGE}' \
  '$REMOTE_ROOT/example_datasets/processed/core4d/assets/objects' \
  '$REMOTE_ROOT/example_datasets/processed/core4d/unitree_g1/humanoid_object' \
  '$REMOTE_ROOT/examples/config/override'"
retry rsync -az workspace/core4d/scripts/E145/ "${REMOTE}:${REMOTE_ROOT}/workspace/core4d/scripts/E145/"
retry rsync -az workspace/core4d/scripts/train/train_E145_full_nonbox_raw_mask_ref_fk.sh "${REMOTE}:${REMOTE_ROOT}/workspace/core4d/scripts/train/"
retry rsync -az workspace/core4d/scripts/eval/eval_E145_full_nonbox_raw_mask_ref_fk.py workspace/core4d/scripts/eval/eval_E145_full_nonbox_raw_mask_ref_fk.sh "${REMOTE}:${REMOTE_ROOT}/workspace/core4d/scripts/eval/"
retry rsync -az workspace/core4d/results/E145/full_nonbox_to_rl_ready/cem_ready/ "${REMOTE}:${REMOTE_ROOT}/workspace/core4d/results/E145/full_nonbox_to_rl_ready/cem_ready/"
retry rsync -az workspace/core4d/results/E145/full_nonbox_to_rl_ready/contact_masks/ "${REMOTE}:${REMOTE_ROOT}/workspace/core4d/results/E145/full_nonbox_to_rl_ready/contact_masks/"

mapfile -t overrides < <(awk -F '\t' 'NF && $1 !~ /^#/ && NR > 2 {print $18}' "$VARIANTS_FILE" | sort -u)
for override in "${overrides[@]}"; do
  retry rsync -az "$override" "${REMOTE}:${REMOTE_ROOT}/$override"
done

mapfile -t tasks < <(awk -F '\t' 'NF && $1 !~ /^#/ && NR > 2 {print $12}' "$VARIANTS_FILE" | sort -u)
for task in "${tasks[@]}"; do
  retry rsync -az "example_datasets/processed/core4d/unitree_g1/humanoid_object/${task}/" \
    "${REMOTE}:${REMOTE_ROOT}/example_datasets/processed/core4d/unitree_g1/humanoid_object/${task}/"
done

mapfile -t object_keys < <(awk -F '\t' 'NF && $1 !~ /^#/ && NR > 2 {print $4}' "$VARIANTS_FILE" | sort -u)
for object_key in "${object_keys[@]}"; do
  retry rsync -az "example_datasets/processed/core4d/assets/objects/${object_key}/" \
    "${REMOTE}:${REMOTE_ROOT}/example_datasets/processed/core4d/assets/objects/${object_key}/"
done

retry ssh "$REMOTE" "cd '$REMOTE_ROOT' && \
  .venv/bin/python -m py_compile workspace/core4d/scripts/eval/eval_E145_full_nonbox_raw_mask_ref_fk.py && \
  bash -n workspace/core4d/scripts/train/train_E145_full_nonbox_raw_mask_ref_fk.sh && \
  bash -n workspace/core4d/scripts/eval/eval_E145_full_nonbox_raw_mask_ref_fk.sh && \
  bash workspace/core4d/scripts/train/train_E145_full_nonbox_raw_mask_ref_fk.sh list '$STAGE' 0 remote-gpu0 >/tmp/e145_gpu0.list && \
  bash workspace/core4d/scripts/train/train_E145_full_nonbox_raw_mask_ref_fk.sh list '$STAGE' 0 remote-gpu1 >/tmp/e145_gpu1.list && \
  echo remote-gpu0=\$(wc -l </tmp/e145_gpu0.list) remote-gpu1=\$(wc -l </tmp/e145_gpu1.list)"

retry ssh "$REMOTE" "cd '$REMOTE_ROOT' && \
  tmux new-session -d -s '$SESSION' \"bash -lc 'set -euo pipefail; \
    echo E145 remote session $SESSION stage=$STAGE; \
    (bash workspace/core4d/scripts/train/train_E145_full_nonbox_raw_mask_ref_fk.sh remote-gpu0 $STAGE 0) & PID0=\\\$!; \
    (bash workspace/core4d/scripts/train/train_E145_full_nonbox_raw_mask_ref_fk.sh remote-gpu1 $STAGE 1) & PID1=\\\$!; \
    echo launched PID0=\\\$PID0 PID1=\\\$PID1; \
    wait \\\$PID0; echo remote-gpu0 done; \
    wait \\\$PID1; echo remote-gpu1 done; \
    echo E145 remote complete'\""

echo "Started remote tmux session: ${SESSION}"
echo "Monitor: ssh ${REMOTE} \"tmux capture-pane -t ${SESSION} -p | tail -40\""
