#!/usr/bin/env bash
# E207 bucket G1only (E178 PRG + object gravcomp, A0 hand-gate) -- full CEM local 8-GPU.
# Reuses the E199 manifest-driven priority queue (all run params live in the manifest
# rows). Coexists with other GPU jobs: only dispatches when free mem >= PER_GPU_MEM_MIB;
# never kills/preempts foreign processes. Resume-safe.
#
# STAGE=smoke runs the 1-case 64x4 manifest into cem/smoke/ so it can never shadow
# the 1024x32 full rollouts via skip-already-done.
#
# Usage:
#   bash workspace/core4d/scripts/launch/active/run_E207_local_8gpu.sh
#   DRY_RUN=1 bash workspace/core4d/scripts/launch/active/run_E207_local_8gpu.sh
#   STAGE=smoke GPUS=0 bash workspace/core4d/scripts/launch/active/run_E207_local_8gpu.sh
# Env overrides: STAGE, GPUS, PER_GPU_MEM_MIB, MAX_PER_GPU, POLL_INTERVAL, DRY_RUN
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

STAGE="${STAGE:-full}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
PER_GPU_MEM_MIB="${PER_GPU_MEM_MIB:-5000}"
MAX_PER_GPU="${MAX_PER_GPU:-1}"
POLL_INTERVAL="${POLL_INTERVAL:-15}"
DRY_RUN="${DRY_RUN:-0}"
PY=".venv/bin/python"
Q="workspace/core4d/scripts/experiments/E199/run_local_priority_queue.py"
MANIFEST_DIR="workspace/core4d/results/E207/s6_downstream/manifests"

case "$STAGE" in
  full)  MANIFEST="$MANIFEST_DIR/g1only_full_manifest.tsv" ;;
  smoke) MANIFEST="$MANIFEST_DIR/g1only_smoke_manifest.tsv" ;;
  *) echo "unknown STAGE=$STAGE (want full|smoke)" >&2; exit 2 ;;
esac
[[ -f "$MANIFEST" ]] || { echo "missing manifest: $MANIFEST (run build_manifest.py)" >&2; exit 2; }

# triton JIT needs python3.12-dev, which is not guaranteed on this host.
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"

EXTRA=""
[[ "$DRY_RUN" == "1" ]] && EXTRA="--dry-run"

echo "[run_E207] STAGE=$STAGE GPUS=$GPUS PER_GPU_MEM_MIB=$PER_GPU_MEM_MIB MAX_PER_GPU=$MAX_PER_GPU"
echo "[run_E207] manifest=$MANIFEST $EXTRA"
exec "$PY" "$Q" --manifest "$MANIFEST" --gpus "$GPUS" \
  --per-gpu-mem-mib "$PER_GPU_MEM_MIB" --max-per-gpu "$MAX_PER_GPU" \
  --poll-interval "$POLL_INTERVAL" $EXTRA
