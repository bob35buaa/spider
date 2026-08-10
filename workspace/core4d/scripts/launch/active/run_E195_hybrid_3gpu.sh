#!/usr/bin/env bash
# Start E195 on local GPU0 plus remote Ada GPU0/1; all three queues are append-only.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
AUDITOR="workspace/core4d/scripts/experiments/E195/audit_config.py"
"$PYTHON_BIN" "$AUDITOR" --require-all

PYTHON_BIN="$PYTHON_BIN" bash workspace/core4d/scripts/launch/active/run_E195_local_gpu0.sh
PYTHON_BIN="$PYTHON_BIN" bash workspace/core4d/scripts/launch/active/run_E195_remote_Ada6000.sh
echo "E195 three-GPU append-only launch complete: local=7 Ada0=4 Ada1=4"

