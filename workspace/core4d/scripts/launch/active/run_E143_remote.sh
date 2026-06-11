#!/usr/bin/env bash
# E143 remote runner: remote GPU0/GPU1 split, local artifacts synced by rsync because E143 files are untracked.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-full}"
REMOTE="${REMOTE:-spider-remote}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/xiayb/pHRI_workspace/spider}"
SESSION="${SESSION:-E143_${STAGE}_$(date +%H%M%S)}"

if [ "$STAGE" != "smoke" ] && [ "$STAGE" != "full" ]; then
  echo "Invalid STAGE=$STAGE (use smoke|full)" >&2
  exit 2
fi

python workspace/core4d/scripts/E143/build_raw_mask_ref_fk_24case_manifest.py

echo "Syncing E143 untracked artifacts to ${REMOTE}:${REMOTE_ROOT}"
ssh "$REMOTE" "mkdir -p \
  '$REMOTE_ROOT/workspace/core4d/scripts/E143' \
  '$REMOTE_ROOT/workspace/core4d/scripts/train' \
  '$REMOTE_ROOT/workspace/core4d/scripts/eval' \
  '$REMOTE_ROOT/workspace/core4d/results/E143/contact_masks' \
  '$REMOTE_ROOT/workspace/core4d/results/E143/preflight' \
  '$REMOTE_ROOT/examples/config/override'"
rsync -az \
  workspace/core4d/scripts/E143/ \
  "${REMOTE}:${REMOTE_ROOT}/workspace/core4d/scripts/E143/"
rsync -az \
  workspace/core4d/scripts/train/train_E143_raw_mask_ref_fk_24case.sh \
  "${REMOTE}:${REMOTE_ROOT}/workspace/core4d/scripts/train/train_E143_raw_mask_ref_fk_24case.sh"
rsync -az \
  workspace/core4d/scripts/eval/eval_E143_raw_mask_ref_fk_24case.py \
  workspace/core4d/scripts/eval/eval_E143_raw_mask_ref_fk_24case.sh \
  "${REMOTE}:${REMOTE_ROOT}/workspace/core4d/scripts/eval/"
rsync -az \
  workspace/core4d/results/E143/contact_masks/ \
  "${REMOTE}:${REMOTE_ROOT}/workspace/core4d/results/E143/contact_masks/"
rsync -az \
  workspace/core4d/results/E143/preflight/ \
  "${REMOTE}:${REMOTE_ROOT}/workspace/core4d/results/E143/preflight/"
rsync -az examples/config/override/core4d_E143_*_raw_mask_ref_fk.yaml \
  "${REMOTE}:${REMOTE_ROOT}/examples/config/override/"

ssh "$REMOTE" "cd '$REMOTE_ROOT' && \
  .venv/bin/python -m py_compile workspace/core4d/scripts/E143/build_raw_mask_ref_fk_24case_manifest.py workspace/core4d/scripts/eval/eval_E143_raw_mask_ref_fk_24case.py && \
  bash -n workspace/core4d/scripts/train/train_E143_raw_mask_ref_fk_24case.sh && \
  bash -n workspace/core4d/scripts/eval/eval_E143_raw_mask_ref_fk_24case.sh && \
  bash workspace/core4d/scripts/train/train_E143_raw_mask_ref_fk_24case.sh list '$STAGE' 0 remote-gpu0 >/tmp/e143_gpu0.list && \
  bash workspace/core4d/scripts/train/train_E143_raw_mask_ref_fk_24case.sh list '$STAGE' 0 remote-gpu1 >/tmp/e143_gpu1.list && \
  echo remote-gpu0=\$(wc -l </tmp/e143_gpu0.list) remote-gpu1=\$(wc -l </tmp/e143_gpu1.list)"

ssh "$REMOTE" "cd '$REMOTE_ROOT' && \
  tmux new-session -d -s '$SESSION' \"bash -lc 'set -euo pipefail; \
    echo E143 remote session $SESSION stage=$STAGE; \
    (bash workspace/core4d/scripts/train/train_E143_raw_mask_ref_fk_24case.sh remote-gpu0 $STAGE 0) & PID0=\\\$!; \
    (bash workspace/core4d/scripts/train/train_E143_raw_mask_ref_fk_24case.sh remote-gpu1 $STAGE 1) & PID1=\\\$!; \
    echo launched PID0=\\\$PID0 PID1=\\\$PID1; \
    wait \\\$PID0; echo remote-gpu0 done; \
    wait \\\$PID1; echo remote-gpu1 done; \
    echo E143 remote complete'\""

echo "Started remote tmux session: ${SESSION}"
echo "Monitor: ssh ${REMOTE} \"tmux capture-pane -t ${SESSION} -p | tail -40\""
