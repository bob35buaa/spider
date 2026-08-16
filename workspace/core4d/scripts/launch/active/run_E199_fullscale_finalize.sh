#!/usr/bin/env bash
# E199 full-scale (plan229) autonomous finalizer: chain the remaining stages once
# the sharded data build (run_E199_fullscale_build.sh) has produced the manifest.
#   1. wait for the full-scale priority manifest + all 6 build shards to finish
#   2. run the 8-GPU CEM priority queue (resume-safe; pilot-done trans auto-skip)
#   3. score aug-vs-orig (public core, per-case pairing) once CEM completes
#   4. render QC keyframes for a per-object sample
# All stages log under logs/E199/fullscale/. Does NOT write log287 (needs manual
# analysis) -- it just gets the artifacts ready.
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

LOG_DIR="logs/E199/fullscale"
mkdir -p "$LOG_DIR"
MANIFEST="workspace/core4d/results/E199/s6_downstream/manifests/e199_fullscale_priority_manifest.tsv"

echo "[finalize] $(date -Iseconds) waiting for data build (manifest + shards done)"
while :; do
  shards_alive=$(pgrep -fc "build_augmented_tasks.py" || true)
  if [[ -f "$MANIFEST" && "${shards_alive:-0}" -eq 0 ]]; then
    break
  fi
  sleep 60
done
rows=$(($(wc -l < "$MANIFEST") - 1))
echo "[finalize] $(date -Iseconds) data build done; manifest rows=$rows"

echo "[finalize] $(date -Iseconds) stage: CEM 8-GPU priority queue"
SCOPE=box_fullscale GPUS="${GPUS:-0,1,2,3,4,5,6,7}" \
  bash workspace/core4d/scripts/launch/active/run_E199_local_8gpu.sh \
  > "$LOG_DIR/cem_queue.log" 2>&1 || echo "[finalize] CEM queue returned nonzero (some rows may need attention)"

echo "[finalize] $(date -Iseconds) stage: eval (aug vs reused-A0 orig)"
bash workspace/core4d/scripts/eval/wrappers/eval_E199_fullscale_augmentation.sh \
  > "$LOG_DIR/eval.log" 2>&1 || echo "[finalize] eval returned nonzero"

echo "[finalize] $(date -Iseconds) stage: render QC (per-object sample)"
MUJOCO_GL=egl .venv/bin/python workspace/core4d/scripts/experiments/E199/render_qc.py \
  --objects box001,box004,box021,box023,box024 \
  --manifest "$MANIFEST" --max-per-object 6 \
  --out workspace/core4d/results/E199/s6_downstream/render/fullscale_qc \
  > "$LOG_DIR/render.log" 2>&1 || echo "[finalize] render returned nonzero"

echo "[finalize] $(date -Iseconds) DONE. Artifacts ready for log287 + tracker."
