#!/usr/bin/env bash
# E213 -- rules/experiment.md §7 Safeguard 2.
#
# Snapshots every aug *task* dir E213 touches (the 100 non-PRG (case, variant)
# rows), capturing scene.xml + the E206-PRG base scene + the newly written
# selected-arm gravcomp sidecar, so the exact CEM scene state is recoverable even
# if the working copies are later overwritten.
#
# Only shard A's machine runs this; all four machines share /mnt and see the
# result.  Run AFTER build_source_arm_scenes.py (the sidecars must exist).
#
# Usage:
#   bash workspace/core4d/scripts/experiments/E213/snapshot_E213_scenes.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

mapfile -t TASKS < <(
  MUJOCO_GL=disable .venv/bin/python - <<'PY'
import sys
sys.path.insert(0, "workspace/core4d/scripts/experiments/E213")
import e213_common as C
cases = {c["case_id"]: c for c in C.load_cases()}
aug = C.aug_rows_by_case()
seen = []
for cid in sorted(cases):
    if C.is_prg_case(cases[cid]):
        continue
    for row in aug.get(cid, []):
        seen.append(row["target_task"])
for t in seen:
    print(t)
PY
)

if [ "${#TASKS[@]}" -eq 0 ]; then
  echo "ERROR: no aug task dirs enumerated" >&2
  exit 1
fi

echo "[E213] snapshotting ${#TASKS[@]} aug task dirs -> workspace/core4d/results/E213/scene_snapshot/"
bash workspace/core4d/scripts/convert/snapshot_scenes.sh E213 "${TASKS[@]}" > /dev/null
echo "[E213] snapshot manifest: workspace/core4d/results/E213/scene_snapshot/manifest.txt"
