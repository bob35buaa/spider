#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."

OUT_DIR="${OUT_DIR:-workspace/core4d/results/E111/contact_alignment_smoke}"

python3 workspace/core4d/scripts/data_construction_v3/stages/s6_downstream/evaluate_contact_alignment.py \
  --make-smoke-fixture \
  --out-dir "$OUT_DIR" \
  --repo-root "$(pwd)"
