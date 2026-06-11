#!/usr/bin/env bash
# Run E091 template preflight for the data_construction_v2 Stage2b backlog.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" "$SCRIPT_DIR/template_preflight.py" "$@"
