#!/usr/bin/env bash
# E202 data construction entry: bucket object augmentation (omnirt_v2, translation
# only) -> SPIDER tasks with the E178 collision body + E174 PRG arm -> CEM overrides
# + priority manifest. Does NOT launch CEM (GPU); run
# scripts/launch/active/run_E202_local_8gpu.sh afterwards.
#
# Scene reproducibility (rule 10b): build_augmented_tasks.py snapshots every E202
# scene sidecar it builds into results/E202/scene_snapshot/cem_sidecars/<task>/.
# The orig E178 baselines are reused (already snapshotted under results/E178).
#
# Usage:
#   bash workspace/core4d/scripts/train/train_E202.sh                 # all 27 bucket cases
#   CASES=bucket004 bash workspace/core4d/scripts/train/train_E202.sh # subset (object or case_id)
# Env overrides: CASES, MAX_WORKERS, FORCE, SKIP_UPSTREAM
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

PY=".venv/bin/python"
CASES="${CASES:-}"
MAX_WORKERS="${MAX_WORKERS:-1}"
DRIVER="workspace/core4d/scripts/experiments/E202/build_augmented_tasks.py"
MANIFEST="workspace/core4d/scripts/experiments/E202/build_aug_manifest.py"
SNAP_DIR="workspace/core4d/results/E202/scene_snapshot"

driver_args=(--max-workers "$MAX_WORKERS")
[[ -n "$CASES" ]] && driver_args+=(--cases "$CASES")
[[ "${FORCE:-0}" == "1" ]] && driver_args+=(--force)
[[ "${SKIP_UPSTREAM:-0}" == "1" ]] && driver_args+=(--skip-upstream)

echo "[train_E202] step 1/3: build augmented bucket SPIDER tasks (${CASES:-all 27}) with E178 collision"
"$PY" "$DRIVER" "${driver_args[@]}"

echo "[train_E202] step 2/3: build CEM overrides + priority manifest (P1 trans; orig reused from E178)"
"$PY" "$MANIFEST"

echo "[train_E202] step 3/3: snapshot manifest (git HEAD + sha256)"
mkdir -p "$SNAP_DIR"
{
  echo "# E202 scene snapshot manifest"
  echo "git_head: $(git rev-parse HEAD)"
  echo "created_at: $(date -Iseconds)"
  echo ""
  find "$SNAP_DIR/cem_sidecars" -name '*.xml' 2>/dev/null | sort | while read -r f; do
    echo "$(sha256sum "$f" | cut -d' ' -f1)  ${f#workspace/}"
  done
} > "$SNAP_DIR/manifest.txt"
echo "[train_E202] done. snapshot manifest -> $SNAP_DIR/manifest.txt"
echo "[train_E202] launch CEM with: bash workspace/core4d/scripts/launch/active/run_E202_local_8gpu.sh"
