#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
OUT_DIR="${OUT_DIR:-$REPO/workspace/core4d/results/E109/fair_eval}"
SCRIPT_DIR="$REPO/workspace/core4d/scripts/eval_omni_vs_spider"

mkdir -p "$OUT_DIR"

python3 "$SCRIPT_DIR/build_case_bank.py" \
  --out-dir "$OUT_DIR" \
  --existing-cases "$REPO/workspace/core4d/data_construction_v3/existing_cases.tsv" \
  --e026-metrics "$REPO/workspace/core4d_collab_retarget/results/E026_full_eval/method_case_metrics.csv" \
  --cem-summary "$REPO/workspace/core4d/results/E105/cem/full/full_eval_summary.csv" \
  --cem-summary "$REPO/workspace/core4d/results/E106/cem/full/full_eval_summary.csv" \
  --cem-summary "$REPO/workspace/core4d/results/E107/cem/full/full_eval_summary.csv" \
  --cem-summary "$REPO/workspace/core4d/results/E108/s6_downstream/cem/full/E108_cem_eval_summary.csv"

python3 "$SCRIPT_DIR/compute_proxy_metrics.py" \
  --case-bank "$OUT_DIR/case_bank.tsv" \
  --out-dir "$OUT_DIR" \
  --thresholds-m "${THRESHOLDS_M:-0.03,0.05,0.08}"

python3 "$SCRIPT_DIR/summarize_fair_eval.py" \
  --out-dir "$OUT_DIR"

echo "[run_E109_fair_eval] outputs written to $OUT_DIR"
