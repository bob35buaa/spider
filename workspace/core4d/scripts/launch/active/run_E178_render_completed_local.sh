#!/usr/bin/env bash
# E178: render a launch-time snapshot of locally complete Full CEM rows.
#
# Completeness requires the primary NPZ, rollout NPZ, config, scene, and source
# trajectory. Rows still running are excluded even if their output directory
# already exists. Rendering is intentionally single-process so it can coexist
# with the RTX 5090 Hybrid CEM worker with limited interference.
#
# Usage:
#   DRY_RUN=1 bash workspace/core4d/scripts/launch/active/run_E178_render_completed_local.sh
#   bash workspace/core4d/scripts/launch/active/run_E178_render_completed_local.sh
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
MUJOCO_GL="${MUJOCO_GL:-egl}"
DRY_RUN="${DRY_RUN:-0}"
MAX_FRAMES="${MAX_FRAMES:-0}"

MANIFEST="workspace/core4d/results/E178/s6_downstream/manifests/semantic_bucket_full_manifest.tsv"
OUT_DIR="workspace/core4d/results/E178/s6_downstream/render/full"
RUNNER="workspace/core4d/scripts/experiments/E168/render_a100_cem_videos.py"
LOG_DIR="logs/E178/render/full"

test -f "$MANIFEST"
test -f "$RUNNER"
mkdir -p "$OUT_DIR" "$LOG_DIR"

exec 9>"$OUT_DIR/.render_completed.lock"
if ! flock -n 9; then
  echo "another E178 completed-row renderer is already running" >&2
  exit 3
fi

mapfile -t COMPLETE_CASES < <(
  "$PYTHON_BIN" - "$MANIFEST" <<'PY'
import csv
import sys
from pathlib import Path

repo = Path.cwd()
manifest = Path(sys.argv[1])
with manifest.open("r", encoding="utf-8", newline="") as stream:
    rows = list(csv.DictReader(stream, delimiter="\t"))

def local_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else repo / path

for row in rows:
    required = (
        row["result_npz"],
        row["outdir_npz"],
        row["config_act"],
        row["scene_act"],
        row["trajectory"],
    )
    if all(local_path(value).is_file() for value in required):
        print(row["case_id"])
PY
)

if [ "${#COMPLETE_CASES[@]}" -eq 0 ]; then
  echo "no complete E178 Full rows are available locally" >&2
  exit 4
fi

run_stamp="$(date '+%Y%m%d_%H%M%S')"
selection_file="$OUT_DIR/render_selection_${run_stamp}.txt"
printf '%s\n' "${COMPLETE_CASES[@]}" > "$selection_file"

args=(
  "$PYTHON_BIN" "$RUNNER"
  --manifest "$MANIFEST"
  --pool all
  --output-dir "$OUT_DIR"
  --cases "${COMPLETE_CASES[@]}"
  --max-frames "$MAX_FRAMES"
)
if [ "$DRY_RUN" = "1" ]; then
  args+=(--dry-run)
fi

echo "selected=${#COMPLETE_CASES[@]} gl=$MUJOCO_GL dry_run=$DRY_RUN"
echo "selection=$selection_file"
MUJOCO_GL="$MUJOCO_GL" "${args[@]}" \
  2>&1 | tee "$LOG_DIR/render_completed_${run_stamp}.log"

