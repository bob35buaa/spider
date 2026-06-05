#!/usr/bin/env bash
# E144 remote runner: launch remote GPU0/GPU1 full-CEM splits.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-full}"
REMOTE="${REMOTE:-spider-remote}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/xiayb/pHRI_workspace/spider}"
SESSION="${SESSION:-E144_${STAGE}_$(date +%H%M%S)}"

if [ "$STAGE" != "smoke" ] && [ "$STAGE" != "full" ]; then
  echo "Invalid STAGE=$STAGE (use smoke|full)" >&2
  exit 2
fi

retry() {
  local attempt=1
  local max_attempts="${E144_REMOTE_RETRY_MAX:-4}"
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

python workspace/core4d/scripts/E144/build_full_nonbox_raw_mask_ref_fk_cem_ready.py
variant_count="$(awk -F '\t' 'NF && $1 !~ /^#/ && NR > 2 {n++} END {print n+0}' workspace/core4d/scripts/E144/variants.tsv)"
if [ "$variant_count" -eq 0 ]; then
  echo "E144 has no CEM-ready variants. Phase 2 remote full CEM is skipped."
  exit 0
fi

echo "Syncing E144 artifacts to ${REMOTE}:${REMOTE_ROOT}"
retry ssh "$REMOTE" "mkdir -p \
  '$REMOTE_ROOT/workspace/core4d/scripts/E144' \
  '$REMOTE_ROOT/workspace/core4d/scripts/train' \
  '$REMOTE_ROOT/workspace/core4d/scripts/eval' \
  '$REMOTE_ROOT/workspace/core4d/results/E144/cem_ready' \
  '$REMOTE_ROOT/workspace/core4d/results/E144/contact_masks' \
  '$REMOTE_ROOT/example_datasets/processed/core4d/assets/objects' \
  '$REMOTE_ROOT/example_datasets/processed/core4d/unitree_g1/humanoid_object' \
  '$REMOTE_ROOT/examples/config/override'"
retry rsync -az workspace/core4d/scripts/E144/ "${REMOTE}:${REMOTE_ROOT}/workspace/core4d/scripts/E144/"
retry rsync -az workspace/core4d/scripts/train/train_E144_raw_mask_ref_fk_full_cem.sh "${REMOTE}:${REMOTE_ROOT}/workspace/core4d/scripts/train/"
retry rsync -az workspace/core4d/scripts/eval/eval_E144_raw_mask_ref_fk_full_cem.py workspace/core4d/scripts/eval/eval_E144_raw_mask_ref_fk_full_cem.sh "${REMOTE}:${REMOTE_ROOT}/workspace/core4d/scripts/eval/"
retry rsync -az workspace/core4d/results/E144/cem_ready/ "${REMOTE}:${REMOTE_ROOT}/workspace/core4d/results/E144/cem_ready/"
retry rsync -az workspace/core4d/results/E144/contact_masks/ "${REMOTE}:${REMOTE_ROOT}/workspace/core4d/results/E144/contact_masks/"
retry rsync -az --ignore-missing-args examples/config/override/core4d_E144_*_raw_mask_ref_fk.yaml "${REMOTE}:${REMOTE_ROOT}/examples/config/override/"

mapfile -t tasks < <(awk -F '\t' 'NF && $1 !~ /^#/ && NR > 2 {print $12}' workspace/core4d/scripts/E144/variants.tsv | sort -u)
for task in "${tasks[@]}"; do
  retry rsync -az "example_datasets/processed/core4d/unitree_g1/humanoid_object/${task}/" \
    "${REMOTE}:${REMOTE_ROOT}/example_datasets/processed/core4d/unitree_g1/humanoid_object/${task}/"
done

mapfile -t object_keys < <(awk -F '\t' 'NF && $1 !~ /^#/ && NR > 2 {print $4}' workspace/core4d/scripts/E144/variants.tsv | sort -u)
for object_key in "${object_keys[@]}"; do
  retry rsync -az "example_datasets/processed/core4d/assets/objects/${object_key}/" \
    "${REMOTE}:${REMOTE_ROOT}/example_datasets/processed/core4d/assets/objects/${object_key}/"
done

retry ssh "$REMOTE" "cd '$REMOTE_ROOT' && \
  .venv/bin/python -m py_compile workspace/core4d/scripts/E144/build_full_nonbox_raw_mask_ref_fk_cem_ready.py workspace/core4d/scripts/eval/eval_E144_raw_mask_ref_fk_full_cem.py && \
  bash -n workspace/core4d/scripts/train/train_E144_raw_mask_ref_fk_full_cem.sh && \
  bash -n workspace/core4d/scripts/eval/eval_E144_raw_mask_ref_fk_full_cem.sh && \
  bash workspace/core4d/scripts/train/train_E144_raw_mask_ref_fk_full_cem.sh list '$STAGE' 0 remote-gpu0 >/tmp/e144_gpu0.list && \
  bash workspace/core4d/scripts/train/train_E144_raw_mask_ref_fk_full_cem.sh list '$STAGE' 0 remote-gpu1 >/tmp/e144_gpu1.list && \
  echo remote-gpu0=\$(wc -l </tmp/e144_gpu0.list) remote-gpu1=\$(wc -l </tmp/e144_gpu1.list)"

retry ssh "$REMOTE" "cd '$REMOTE_ROOT' && \
  tmux new-session -d -s '$SESSION' \"bash -lc 'set -euo pipefail; \
    echo E144 remote session $SESSION stage=$STAGE; \
    (bash workspace/core4d/scripts/train/train_E144_raw_mask_ref_fk_full_cem.sh remote-gpu0 $STAGE 0) & PID0=\\\$!; \
    (bash workspace/core4d/scripts/train/train_E144_raw_mask_ref_fk_full_cem.sh remote-gpu1 $STAGE 1) & PID1=\\\$!; \
    echo launched PID0=\\\$PID0 PID1=\\\$PID1; \
    wait \\\$PID0; echo remote-gpu0 done; \
    wait \\\$PID1; echo remote-gpu1 done; \
    echo E144 remote complete'\""

echo "Started remote tmux session: ${SESSION}"
echo "Monitor: ssh ${REMOTE} \"tmux capture-pane -t ${SESSION} -p | tail -40\""
