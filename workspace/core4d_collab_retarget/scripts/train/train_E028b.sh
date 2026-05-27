#!/usr/bin/env bash
# E028b: local candidate-only dynamic retarget with contact-centroid anchors.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-local}"

case "$MODE" in
  eval)
    shift || true
    .venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E028b.py "$@"
    ;;
  *)
    EXP_ID=E028b \
    VARIANTS_FILE=workspace/core4d_collab_retarget/results/E028b_anchor_refit/manifest.tsv \
    RESULTS=workspace/core4d_collab_retarget/results/E028b_anchor_refit \
    LOGS=logs/core4d_collab_retarget/E028b_anchor_refit \
      bash workspace/core4d_collab_retarget/scripts/train/train_E028.sh "$@"
    ;;
esac
