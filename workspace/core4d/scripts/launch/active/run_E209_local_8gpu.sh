#!/usr/bin/env bash
# E209 desk/chair G1 (E206 PRG + object gravcomp, A0 hand-gate) -- CEM on local 8 GPUs.
#
# Reuses the E199 manifest-driven priority queue (all run params live in the
# manifest rows). Coexists with other GPU jobs: only dispatches when free mem
# >= PER_GPU_MEM_MIB; never kills/preempts foreign processes. Resume-safe.
#
# Must be the E199 queue, NOT E200's: e200_common.TIER_RANK is {"P1": 1} and the
# E209 manifest is tier P0, which would KeyError on dispatch. e199_common:140
# carries P0/P1/P2.
#
# STAGE=smoke runs the 1-case 64x4 manifest into cem/smoke/ so it can never
# shadow the 1024x32 full rollouts via skip-already-done.
#
# Usage:
#   bash workspace/core4d/scripts/launch/active/run_E209_local_8gpu.sh
#   DRY_RUN=1 bash workspace/core4d/scripts/launch/active/run_E209_local_8gpu.sh
#   STAGE=smoke GPUS=0 bash workspace/core4d/scripts/launch/active/run_E209_local_8gpu.sh
# Env overrides: STAGE, GPUS, PER_GPU_MEM_MIB, MAX_PER_GPU, POLL_INTERVAL, DRY_RUN, SKIP_SNAPSHOT
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

STAGE="${STAGE:-full}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
PER_GPU_MEM_MIB="${PER_GPU_MEM_MIB:-5000}"
MAX_PER_GPU="${MAX_PER_GPU:-1}"
POLL_INTERVAL="${POLL_INTERVAL:-15}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_SNAPSHOT="${SKIP_SNAPSHOT:-0}"
PY=".venv/bin/python"
Q="workspace/core4d/scripts/experiments/E199/run_local_priority_queue.py"
MANIFEST_DIR="workspace/core4d/results/E209/s6_downstream/manifests"

case "$STAGE" in
  full)  MANIFEST="$MANIFEST_DIR/e209_g1_full_manifest.tsv" ;;
  smoke) MANIFEST="$MANIFEST_DIR/e209_g1_smoke_manifest.tsv" ;;
  *) echo "unknown STAGE=$STAGE (want full|smoke)" >&2; exit 2 ;;
esac
[[ -f "$MANIFEST" ]] || { echo "missing manifest: $MANIFEST (run build_manifest.py)" >&2; exit 2; }

# rules/experiment.md §7 Safeguard 2: freeze the scenes before spending GPU.
if [[ "$SKIP_SNAPSHOT" != "1" && "$DRY_RUN" != "1" ]]; then
  bash workspace/core4d/scripts/experiments/E209/snapshot_E209_scenes.sh
fi

# triton JIT needs python3.12-dev, which is not guaranteed on this host.
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
# CEM is headless; osmesa is only needed by the renderer.
export MUJOCO_GL="${MUJOCO_GL:-disable}"

EXTRA=""
[[ "$DRY_RUN" == "1" ]] && EXTRA="--dry-run"

echo "[run_E209] STAGE=$STAGE GPUS=$GPUS PER_GPU_MEM_MIB=$PER_GPU_MEM_MIB MAX_PER_GPU=$MAX_PER_GPU"
echo "[run_E209] manifest=$MANIFEST $EXTRA"
exec "$PY" "$Q" --manifest "$MANIFEST" --gpus "$GPUS" \
  --per-gpu-mem-mib "$PER_GPU_MEM_MIB" --max-per-gpu "$MAX_PER_GPU" \
  --poll-interval "$POLL_INTERVAL" $EXTRA
