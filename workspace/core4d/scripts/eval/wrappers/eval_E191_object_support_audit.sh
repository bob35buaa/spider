#!/usr/bin/env bash
# E191: offline object-support audit. Re-scores already-completed E172/E173/
# E174/E189 rollouts with the additive object-support columns, then emits the
# pre-registered H1..H6 verdicts and the config-provenance table.
#
# Runs no physics simulation and writes nothing outside results/E191/audit/.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
WORKERS="${WORKERS:-32}"

# The CEM/eval scripts on this host must not force MUJOCO_GL; forcing egl or
# osmesa makes `import mujoco` fail with a raw AttributeError (see E189 launcher).
unset MUJOCO_GL || true

"${PYTHON_BIN}" workspace/core4d/scripts/eval/runners/eval_E191_object_support_audit.py \
  --workers "${WORKERS}" "$@"

"${PYTHON_BIN}" workspace/core4d/scripts/eval/reports/gen_E191_object_support_report.py

echo "E191 audit complete -> workspace/core4d/results/E191/audit/"
