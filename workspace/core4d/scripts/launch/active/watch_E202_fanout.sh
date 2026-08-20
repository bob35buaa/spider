#!/usr/bin/env bash
# E202 fan-out watcher: wait for the full 27-case data build to finish, then build
# the full priority manifest and launch the 8-GPU CEM queue (resume-safe). Fully
# unattended. Avoids manifest/queue races by only building the manifest once the
# data build has exited and no shakeout queue is still writing it.
set -uo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
PY=".venv/bin/python"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

log "waiting for full data build (build_augmented_tasks.py) to finish..."
while pgrep -f "experiments/E202/build_augmented_tasks.py" >/dev/null; do sleep 60; done
log "data build finished."

log "waiting for any live shakeout CEM queue to exit..."
while pgrep -f "run_local_priority_queue.py" >/dev/null; do sleep 30; done
log "no queue running."

log "building full priority manifest..."
"$PY" workspace/core4d/scripts/experiments/E202/build_aug_manifest.py || { log "manifest build FAILED"; exit 1; }
ROWS=$(($(wc -l < workspace/core4d/results/E202/s6_downstream/manifests/e202_bucket_priority_manifest.tsv) - 1))
log "manifest ready: $ROWS queue rows."

log "snapshotting scene sidecars (rule 10b)..."
SNAP_DIR="workspace/core4d/results/E202/scene_snapshot"; mkdir -p "$SNAP_DIR"
{ echo "# E202 scene snapshot manifest"; echo "git_head: $(git rev-parse HEAD)"; echo "created_at: $(date -Iseconds)"; echo;
  find "$SNAP_DIR/cem_sidecars" -name '*.xml' 2>/dev/null | sort | while read -r f; do echo "$(sha256sum "$f" | cut -d' ' -f1)  ${f#workspace/}"; done; } > "$SNAP_DIR/manifest.txt"

log "launching 8-GPU CEM queue (resume-safe)..."
GPUS=0,1,2,3,4,5,6,7 bash workspace/core4d/scripts/launch/active/run_E202_local_8gpu.sh
log "queue exited."
