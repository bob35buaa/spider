#!/usr/bin/env bash
# E199 data construction entry: upstream object augmentation -> SPIDER tasks ->
# CEM overrides + priority manifest. This does NOT launch CEM (GPU); run
# scripts/launch/active/run_E199_local_8gpu.sh afterwards.
#
# Scene reproducibility: build_augmented_tasks.py snapshots every scene sidecar
# it builds into workspace/core4d/results/E199/scene_snapshot/cem_sidecars/<task>/.
# A manifest.txt (git HEAD + sha256) is written here as the second safeguard.
#
# Usage:
#   bash workspace/core4d/scripts/train/train_E199.sh                       # pilot: all 8 cases
#   SCOPE=box_fullscale bash workspace/core4d/scripts/train/train_E199.sh   # plan229: all s6 box cases, translation-only
#   CASES=box024 bash workspace/core4d/scripts/train/train_E199.sh          # subset
# Env overrides: SCOPE(pilot|box_fullscale), CASES, MAX_WORKERS, FORCE, SKIP_UPSTREAM
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

PY=".venv/bin/python"
SCOPE="${SCOPE:-pilot}"
CASES="${CASES:-}"
MAX_WORKERS="${MAX_WORKERS:-1}"
DRIVER="workspace/core4d/scripts/experiments/E199/build_augmented_tasks.py"
MANIFEST="workspace/core4d/scripts/experiments/E199/build_aug_manifest.py"
SNAP_DIR="workspace/core4d/results/E199/scene_snapshot"

driver_args=(--scope "$SCOPE" --max-workers "$MAX_WORKERS")
[[ -n "$CASES" ]] && driver_args+=(--cases "$CASES")
[[ "${FORCE:-0}" == "1" ]] && driver_args+=(--force)
[[ "${SKIP_UPSTREAM:-0}" == "1" ]] && driver_args+=(--skip-upstream)

echo "[train_E199] scope=$SCOPE step 1/3: build augmented SPIDER tasks (${CASES:-all})"
"$PY" "$DRIVER" "${driver_args[@]}"

echo "[train_E199] step 2/3: build CEM overrides + priority manifest"
"$PY" "$MANIFEST" --scope "$SCOPE"

echo "[train_E199] step 3/3: snapshot manifest (git HEAD + sha256)"
mkdir -p "$SNAP_DIR"
{
  echo "# E199 scene snapshot manifest"
  echo "git_head: $(git rev-parse HEAD)"
  echo "created_at: $(date -Iseconds)"
  echo ""
  find "$SNAP_DIR/cem_sidecars" -name '*.xml' 2>/dev/null | sort | while read -r f; do
    echo "$(sha256sum "$f" | cut -d' ' -f1)  ${f#workspace/}"
  done
} > "$SNAP_DIR/manifest.txt"
echo "[train_E199] done. snapshot manifest -> $SNAP_DIR/manifest.txt"
echo "[train_E199] launch CEM with: bash workspace/core4d/scripts/launch/active/run_E199_local_8gpu.sh"
