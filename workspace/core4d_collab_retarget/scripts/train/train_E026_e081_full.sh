#!/usr/bin/env bash
# E026: run E081-style leg/object baseline variants without overwriting E081.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-local}"
GPU="${2:-0}"
VARIANTS_FILE="${VARIANTS_FILE:-workspace/core4d_collab_retarget/scripts/E026/e081_full_variants.tsv}"
RESULTS="${RESULTS:-workspace/core4d_collab_retarget/results/E026_E081_full}"
LOGS="${LOGS:-logs/core4d_collab_retarget/E026_E081_full}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-7200}"
mkdir -p "$RESULTS" "$LOGS"

variants_for_split() {
  local split_name=$1
  awk -F '\t' -v want_split="$split_name" 'NF && $1 !~ /^#/ && $7 == want_split {print $1}' "$VARIANTS_FILE"
}

run_eval() {
  VARIANTS_FILE="$VARIANTS_FILE" RESULTS="$RESULTS" \
    .venv/bin/python workspace/core4d/scripts/eval/eval_E081.py "$@" \
    | tee "$LOGS/eval_${MODE}.log"
}

case "$MODE" in
  local|remote_gpu0|remote_gpu1)
    mapfile -t variants < <(variants_for_split "$MODE")
    if [ "${#variants[@]}" -eq 0 ]; then
      echo "No E026 E081 variants for split=${MODE}" >&2
      exit 2
    fi
    for variant in "${variants[@]}"; do
      VARIANTS_FILE="$VARIANTS_FILE" RESULTS="$RESULTS" LOGS="$LOGS" \
        timeout "$RUN_TIMEOUT_SECONDS" bash workspace/core4d/scripts/train/train_E081.sh single "$GPU" "$variant"
    done
    run_eval "${variants[@]}"
    ;;
  single)
    variant="${3:-}"
    if [ -z "$variant" ]; then
      echo "Usage: $0 single <gpu> <variant>" >&2
      exit 2
    fi
    VARIANTS_FILE="$VARIANTS_FILE" RESULTS="$RESULTS" LOGS="$LOGS" \
      timeout "$RUN_TIMEOUT_SECONDS" bash workspace/core4d/scripts/train/train_E081.sh single "$GPU" "$variant"
    run_eval "$variant"
    ;;
  eval)
    shift || true
    run_eval "$@"
    ;;
  *)
    echo "Usage:"
    echo "  $0 local 0"
    echo "  $0 remote_gpu0 0"
    echo "  $0 remote_gpu1 1"
    echo "  $0 single 0 E026_E081_box025_p2_legobj"
    echo "  $0 eval [variant ...]"
    exit 2
    ;;
esac

echo "=== E026 E081 ${MODE} done ==="
