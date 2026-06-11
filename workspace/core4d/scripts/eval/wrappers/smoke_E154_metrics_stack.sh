#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
cd "${REPO_ROOT}"

python3 - <<'PY'
import sys
from pathlib import Path

repo = Path.cwd()
sys.path.insert(0, str(repo / "workspace/core4d/scripts"))
sys.path.insert(0, str(repo / "workspace/core4d/scripts/eval"))

import eval.core.core_metrics as core
import lib.core_metrics as compat

assert compat.evaluate_sequence is core.evaluate_sequence
assert compat.EVAL_METRIC_STANDARD_ID == core.EVAL_METRIC_STANDARD_ID
PY

python3 workspace/core4d/scripts/E154/eval_omniretarget_reference.py
python3 workspace/core4d/scripts/eval/eval_E152_axis1_hand_object_physics_gate.py full --skip-visual
python3 workspace/core4d/scripts/eval/eval_E153_gate_threshold_sweep.py full
python3 workspace/core4d/scripts/E154/gen_compare_table.py
