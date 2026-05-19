#!/usr/bin/env bash
# Internal remote entrypoint for E016.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
export RUN_STALL_TIMEOUT_SECONDS="${RUN_STALL_TIMEOUT_SECONDS:-600}"
export E016_QUICK_NUM_SAMPLES="${E016_QUICK_NUM_SAMPLES:-128}"
export E016_QUICK_MAX_ITERS="${E016_QUICK_MAX_ITERS:-4}"

bash workspace/core4d_collab_retarget/scripts/run_E016_preprocess.sh --force

(
  bash workspace/core4d_collab_retarget/scripts/train/train_E016.sh remote_gpu0 "$GPU0"
) &
pid0=$!
(
  bash workspace/core4d_collab_retarget/scripts/train/train_E016.sh remote_gpu1 "$GPU1"
) &
pid1=$!

wait "$pid0"
wait "$pid1"
