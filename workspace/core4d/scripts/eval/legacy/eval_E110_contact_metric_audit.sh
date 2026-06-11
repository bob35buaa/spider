#!/usr/bin/env bash
# E110 contact metric audit.
# CPU-only evaluation; does not launch CEM/RL or use GPU.
set -euo pipefail

cd "$(dirname "$0")/../../../.."

python3 workspace/core4d/scripts/eval_omni_vs_spider/contact_metric_audit.py \
  --method-metrics-tsv workspace/core4d/results/E109/expanded_24_work_cases/expanded_24_method_metrics.tsv \
  --case-comparison-tsv workspace/core4d/results/E109/expanded_24_work_cases/expanded_24_case_comparison.tsv \
  --out-dir workspace/core4d/results/E110/contact_metric_audit
