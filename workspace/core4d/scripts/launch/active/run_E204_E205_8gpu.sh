#!/usr/bin/env bash
# E204 (noPRG) + E205 (G1A2) arm ablation on E178's 27 bucket cases — 54 full CEM.
#
# SELF-CONTAINED launcher for an 8-GPU machine. Reuses E178's omnirt_v1 ref_fk
# reference trajectories + 3cm contact masks (core4d v1; 1/27 cases is omnirt_v2)
# VERBATIM — only reruns CEM under two new reward arms on the SAME contact-aligned
# 5-segment object proxy. No upstream retarget, no trajectory rebuild.
#
# Runs three steps in order (resume-safe, coexists with other GPU jobs):
#   0. build_arm_scenes.py  -> regenerate E178 scene + derive E204 noPRG / E205 gravcomp
#                              scenes into each task dir, snapshot (rule 10b).
#   1. build_overrides.py   -> write + compose-audit 27x2 override YAMLs.
#   2. run_e204e205_cem.py  -> 54 full CEM (1024x32 seed0) across the GPU pool,
#                              skip-already-done, per-arm output_dir (no collision).
#
# PREREQ on the run machine:
#   * git pull  (sync this branch)
#   * .venv present (shared SPIDER venv; do NOT run `uv sync`)
#   * example_datasets/processed/core4d/.../dcv3_omnirt_v{1,2}_ref_fk_<case>/ present
#     with scene_act_E174_rubberHull_PRG.xml + 0/trajectory_kinematic.npz + 3cm mask.
#
# Usage:
#   bash workspace/core4d/scripts/launch/active/run_E204_E205_8gpu.sh
#   GPUS=0,1,2,3 bash .../run_E204_E205_8gpu.sh                # subset of GPUs
#   ARMS=noprg_e204 bash .../run_E204_E205_8gpu.sh             # one arm only
#   DRY_RUN=1 bash .../run_E204_E205_8gpu.sh                   # print CEM commands
#   STAGE=smoke LIMIT=1 NUM_SAMPLES=64 MAX_ITERS=4 bash .../run_E204_E205_8gpu.sh   # canary smoke
#   REUSE_E178=1 bash .../run_E204_E205_8gpu.sh                # skip E178 scene rebuild
#   E204E205_MUJOCO_GL=egl bash .../run_E204_E205_8gpu.sh      # force a GL backend (default 'disable' is headless-safe)
set -uo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
export PATH="$PWD/.venv/bin:$PATH"
PY=.venv/bin/python
ED=workspace/core4d/scripts/experiments/E204_E205

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
ARMS="${ARMS:-noprg_e204,g1a2_e205}"
NUM_SAMPLES="${NUM_SAMPLES:-1024}"
MAX_ITERS="${MAX_ITERS:-32}"
SEED="${SEED:-0}"
export E204E205_MUJOCO_GL="${E204E205_MUJOCO_GL:-disable}"  # CEM is headless (save_video=false); 'disable' skips the GL backend import entirely -> works on boxes with no libGL/libOSMesa/EGL. Override to egl/osmesa only if you need on-the-fly rendering.
export E204E205_TORCH_COMPILE="${E204E205_TORCH_COMPILE:-0}"  # 0 = no torch.compile (freeze); also sets TORCHDYNAMO_DISABLE=1 so triton JIT (needs python3.12-dev) is never invoked. CEM passes use_torch_compile=false regardless.
[ "${FORCE:-0}" = "1" ] && export E204E205_FORCE=1

SCENE_ARGS=""; [ "${REUSE_E178:-0}" = "1" ] && SCENE_ARGS="--reuse-existing-e178-scene"
[ -n "${LIMIT:-}" ] && SCENE_ARGS="$SCENE_ARGS --limit $LIMIT"
STAGE="${STAGE:-full}"   # set STAGE=smoke for canary runs so they don't shadow full/
CEM_ARGS="--arms $ARMS --gpus $GPUS --num-samples $NUM_SAMPLES --max-iterations $MAX_ITERS --seed $SEED --stage $STAGE"
[ "${DRY_RUN:-0}" = "1" ] && CEM_ARGS="$CEM_ARGS --dry-run"
[ -n "${LIMIT:-}" ] && CEM_ARGS="$CEM_ARGS --limit $LIMIT"

echo "===== [E204/E205] STEP 0: build arm scenes (+ snapshot, rule 10b) ====="
# scene build compiles MuJoCo; unset egl/osmesa hint here can crash import on some
# boxes -- build_arm_scenes leaves MUJOCO_GL to the env, so honor E204E205_MUJOCO_GL.
MUJOCO_GL="$E204E205_MUJOCO_GL" $PY $ED/build_arm_scenes.py $SCENE_ARGS

echo "===== [E204/E205] STEP 1: build + audit 27x2 override YAMLs ====="
$PY $ED/build_overrides.py --audit

echo "===== [E204/E205] STEP 2: 54 full CEM (8-GPU, skip-already-done) ====="
$PY $ED/run_e204e205_cem.py $CEM_ARGS
RC=$?

echo "===== [E204/E205] DONE (rc=$RC). Summary: ====="
cat "workspace/core4d/results/E204/s6_downstream/cem/$STAGE/_driver/e204e205_cem_summary.json" 2>/dev/null \
  | $PY -c "import sys,json,collections; d=json.load(sys.stdin); print(collections.Counter(r['status'] for r in d))" 2>/dev/null || true
exit $RC
