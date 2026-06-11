#!/usr/bin/env bash
# Fixed entry for E113 contact-aware expanded workset evaluation.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-full}"
shift || true

.venv/bin/python workspace/core4d/scripts/eval/eval_E113_contact_aware_expanded.py \
  --stage "$STAGE" "$@"
