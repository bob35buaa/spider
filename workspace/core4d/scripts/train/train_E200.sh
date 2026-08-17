#!/usr/bin/env bash
# E200 data prep: reuse the E199 full-scale translation augmentation and wire up
# two additional downstream arms (noprg / prg_g1a2). This does NOT launch CEM
# (GPU); run the per-arm launch scripts afterwards:
#   workspace/core4d/scripts/launch/active/run_E200_prg_g1a2_8gpu.sh   (machine A)
#   workspace/core4d/scripts/launch/active/run_E200_noprg_8gpu.sh      (machine B)
#
# Steps (CPU only):
#   1) build_arm_scenes.py  -> prg_g1a2: gravcomp (G1) sidecars (single-var diff),
#                              noprg: verify rubber-hull (no-PRG) scenes; snapshot both.
#   2) build_manifest.py    -> per-arm 8-GPU priority manifests (arm-tagged outputs).
#
# Usage:
#   bash workspace/core4d/scripts/train/train_E200.sh                 # both arms
#   ARMS=prg_g1a2 bash workspace/core4d/scripts/train/train_E200.sh   # one arm
# Env overrides: ARMS(noprg,prg_g1a2), OVERWRITE(0|1)
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

PY=".venv/bin/python"
ARMS="${ARMS:-noprg,prg_g1a2}"
E200="workspace/core4d/scripts/experiments/E200"
scene_args=(--arms "$ARMS")
[[ "${OVERWRITE:-0}" == "1" ]] && scene_args+=(--overwrite)

echo "[train_E200] step 1/2: build/verify per-arm scenes (arms=$ARMS)"
"$PY" "$E200/build_arm_scenes.py" "${scene_args[@]}"

echo "[train_E200] step 2/2: build per-arm priority manifests"
"$PY" "$E200/build_manifest.py" --arms "$ARMS"

echo "[train_E200] done. launch CEM per arm with the run_E200_*_8gpu.sh scripts."
