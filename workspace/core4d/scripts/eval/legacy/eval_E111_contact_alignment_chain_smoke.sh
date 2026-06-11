#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."

CHAIN_MANIFEST="${CHAIN_MANIFEST:-workspace/core4d/results/E111/contact_chain_smoke/E111_contact_chain_smoke/s5_handoff/handoff_manifest.tsv}"
OUT_DIR="${OUT_DIR:-workspace/core4d/results/E111/contact_alignment_chain_smoke}"

python3 workspace/core4d/scripts/data_construction_v3/stages/s6_downstream/evaluate_contact_alignment.py \
  --manifest-tsv "$CHAIN_MANIFEST" \
  --out-dir "$OUT_DIR" \
  --repo-root "$(pwd)" \
  --method-name "chain_smoke"
