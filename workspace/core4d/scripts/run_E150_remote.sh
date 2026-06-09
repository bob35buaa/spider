#!/usr/bin/env bash
# E150 remote launcher: run eef_offset 0.08/0.11 sweep on remote GPU0/GPU1.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-full}"
REMOTE="${REMOTE:-spider-remote}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/xiayb/pHRI_workspace/spider}"
SESSION="${SESSION:-E150_${STAGE}_$(date +%H%M%S)}"
VARIANTS_FILE="${VARIANTS_FILE:-workspace/core4d/scripts/E150/variants.tsv}"

if [ "$STAGE" != "smoke" ] && [ "$STAGE" != "full" ]; then
  echo "Invalid STAGE=$STAGE (use smoke|full)" >&2
  exit 2
fi

retry() {
  local attempt=1
  local max_attempts="${E150_REMOTE_RETRY_MAX:-4}"
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

.venv/bin/python workspace/core4d/scripts/E150/build_eef_offset_sweep_manifest.py >/tmp/e150_manifest_build.log

echo "Syncing E150 artifacts to ${REMOTE}:${REMOTE_ROOT}"
retry ssh "$REMOTE" "mkdir -p \
  '$REMOTE_ROOT/workspace/core4d/scripts/E150' \
  '$REMOTE_ROOT/workspace/core4d/scripts/train' \
  '$REMOTE_ROOT/workspace/core4d/results/E150/contact_anchor_eef_offset_sweep/cem/${STAGE}' \
  '$REMOTE_ROOT/workspace/core4d/results/E143/contact_masks' \
  '$REMOTE_ROOT/logs/E150/cem/${STAGE}' \
  '$REMOTE_ROOT/example_datasets/processed/core4d/assets/objects' \
  '$REMOTE_ROOT/example_datasets/processed/core4d/unitree_g1/humanoid_object' \
  '$REMOTE_ROOT/examples/config/override'"

retry rsync -az workspace/core4d/scripts/E150/ "${REMOTE}:${REMOTE_ROOT}/workspace/core4d/scripts/E150/"
retry rsync -az workspace/core4d/scripts/train/train_E150_eef_offset_sweep.sh "${REMOTE}:${REMOTE_ROOT}/workspace/core4d/scripts/train/"
retry rsync -az workspace/core4d/results/E150/contact_anchor_eef_offset_sweep/ "${REMOTE}:${REMOTE_ROOT}/workspace/core4d/results/E150/contact_anchor_eef_offset_sweep/"
retry rsync -az workspace/core4d/results/E143/contact_masks/ "${REMOTE}:${REMOTE_ROOT}/workspace/core4d/results/E143/contact_masks/"
retry rsync -az examples/config/override/core4d_E143_*_raw_mask_ref_fk.yaml "${REMOTE}:${REMOTE_ROOT}/examples/config/override/"
retry rsync -az examples/config/override/core4d_E147_*_rubber_hull.yaml "${REMOTE}:${REMOTE_ROOT}/examples/config/override/"
retry rsync -az examples/config/override/core4d_E148_*_rubber_hull.yaml "${REMOTE}:${REMOTE_ROOT}/examples/config/override/"

mapfile -t task_dirs < <(awk -F '\t' 'NF && NR > 1 && $11 == "to_run" {print "example_datasets/processed/core4d/unitree_g1/humanoid_object/" $18}' "$VARIANTS_FILE" | sort -u)
mapfile -t rubber_dirs < <(awk -F '\t' 'NF && NR > 1 && $11 == "to_run" {print $22}' "$VARIANTS_FILE" | xargs -n1 dirname | sort -u)
for dir in "${task_dirs[@]}" "${rubber_dirs[@]}"; do
  [ -d "$dir" ] || continue
  retry ssh "$REMOTE" "mkdir -p '$REMOTE_ROOT/$dir'"
  retry rsync -az "$dir/" "${REMOTE}:${REMOTE_ROOT}/${dir}/"
done

mapfile -t object_keys < <(awk -F '\t' 'NF && NR > 1 && $11 == "to_run" {print $7}' "$VARIANTS_FILE" | sort -u)
for object_key in "${object_keys[@]}"; do
  retry rsync -az "example_datasets/processed/core4d/assets/objects/${object_key}/" \
    "${REMOTE}:${REMOTE_ROOT}/example_datasets/processed/core4d/assets/objects/${object_key}/"
done

retry ssh "$REMOTE" "cd '$REMOTE_ROOT' && \
  .venv/bin/python -m py_compile workspace/core4d/scripts/E150/build_eef_offset_sweep_manifest.py && \
  bash -n workspace/core4d/scripts/train/train_E150_eef_offset_sweep.sh && \
  bash workspace/core4d/scripts/train/train_E150_eef_offset_sweep.sh list '$STAGE' 0 remote-gpu0 >/tmp/e150_gpu0.list && \
  bash workspace/core4d/scripts/train/train_E150_eef_offset_sweep.sh list '$STAGE' 0 remote-gpu1 >/tmp/e150_gpu1.list && \
  echo remote-gpu0=\$(wc -l </tmp/e150_gpu0.list) remote-gpu1=\$(wc -l </tmp/e150_gpu1.list)"

retry ssh "$REMOTE" "cd '$REMOTE_ROOT' && \
  tmux new-session -d -s '$SESSION' \"bash -lc 'set -euo pipefail; \
    echo E150 remote session $SESSION stage=$STAGE; \
    (bash workspace/core4d/scripts/train/train_E150_eef_offset_sweep.sh remote-gpu0 $STAGE 0) & PID0=\\\$!; \
    (bash workspace/core4d/scripts/train/train_E150_eef_offset_sweep.sh remote-gpu1 $STAGE 1) & PID1=\\\$!; \
    echo launched PID0=\\\$PID0 PID1=\\\$PID1; \
    wait \\\$PID0; echo remote-gpu0 done; \
    wait \\\$PID1; echo remote-gpu1 done; \
    echo E150 remote complete'\""

echo "Started remote tmux session: ${SESSION}"
echo "Monitor: ssh ${REMOTE} \"tmux capture-pane -t ${SESSION} -p | tail -80\""
