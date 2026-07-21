#!/usr/bin/env bash
# E171 evaluation wrapper: unified evaluator over the frozen CEM full manifest.
# Imports eval.core.core_metrics directly (no dynamic loading of other exps).
#   STAGE=full  -> --require-all ; STAGE=available -> --allow-missing
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
STAGE="${STAGE:-${1:-full}}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
E=workspace/core4d/results/E171
RUNNER=workspace/core4d/scripts/eval/runners/eval_E171_box022_box026.py
ARGS=(--manifest "$E/s6_downstream/manifests/cem_full_manifest.tsv" --out-dir "$E/s6_downstream/eval/full")
if [ "$STAGE" = "full" ]; then ARGS+=(--require-all); else ARGS+=(--allow-missing); fi
exec .venv/bin/python "$RUNNER" "${ARGS[@]}"
