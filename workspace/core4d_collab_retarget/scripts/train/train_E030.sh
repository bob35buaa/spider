#!/usr/bin/env bash
# E030: direct full CEM for the three E029 D6-locked sanity-pass cases.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-local}"
GPU="${2:-0}"
VARIANT="${3:-}"

export VARIANTS_FILE="${VARIANTS_FILE:-workspace/core4d_collab_retarget/results/E029/d6/manifest.tsv}"
export RESULTS="${RESULTS:-workspace/core4d_collab_retarget/results/E030/d6_locked_cem}"
export LOGS="${LOGS:-logs/core4d_collab_retarget/E030}"

LOCAL_VARIANT="E029_d003_box021_20231018_029_p2_d6_locked"
REMOTE_GPU0_VARIANT="E029_d003_box021_20231011_035_p2_d6_locked"
REMOTE_GPU1_VARIANT="E029_d003_box021_20231020_019_p1_d6_locked"

mkdir -p "$RESULTS" "$LOGS"

run_one_e030() {
  local gpu=$1
  local variant=$2
  bash workspace/core4d_collab_retarget/scripts/train/train_E029.sh one "$gpu" "$variant"
}

case "$MODE" in
  local)
    run_one_e030 "$GPU" "$LOCAL_VARIANT"
    ;;
  remote-gpu0)
    run_one_e030 "$GPU" "$REMOTE_GPU0_VARIANT"
    ;;
  remote-gpu1)
    run_one_e030 "$GPU" "$REMOTE_GPU1_VARIANT"
    ;;
  one)
    if [ -z "$VARIANT" ]; then
      echo "Usage: $0 one <gpu> <variant>" >&2
      exit 2
    fi
    run_one_e030 "$GPU" "$VARIANT"
    ;;
  list)
    printf '%s\n' "$LOCAL_VARIANT" "$REMOTE_GPU0_VARIANT" "$REMOTE_GPU1_VARIANT"
    ;;
  *)
    echo "Usage: $0 local 0 | remote-gpu0 0 | remote-gpu1 1 | one <gpu> <variant> | list" >&2
    exit 2
    ;;
esac

