#!/usr/bin/env bash
# E216 -- rules/experiment.md §7 Safeguard 2.
#
# Snapshots every E216 aug *task* dir (the built (case, variant) rows), capturing
# scene.xml + scene_act.xml + the arm scene (PRG / rubberHull / gravcomp sidecar),
# so the exact 0.1m CEM scene state is recoverable even if working copies are
# later overwritten.  Run AFTER build_augmented_tasks.py, BEFORE the CEM queue.
#
# Usage:
#   bash workspace/core4d/scripts/experiments/E216/snapshot_E216_scenes.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

mapfile -t TASKS < <(
  MUJOCO_GL=disable .venv/bin/python - <<'PY'
import sys
sys.path.insert(0, "workspace/core4d/scripts/experiments/E216")
import e216_common as C
arts = C.read_tsv(C.ARTIFACTS_TSV) if C.ARTIFACTS_TSV.is_file() else []
seen = []
for r in arts:
    if r.get("status", "").startswith("built"):
        seen.append(r["target_task"])
for t in sorted(set(seen)):
    print(t)
PY
)

if [ "${#TASKS[@]}" -eq 0 ]; then
  echo "ERROR: no built E216 aug task dirs enumerated (run build_augmented_tasks.py first)" >&2
  exit 1
fi

echo "[E216] snapshotting ${#TASKS[@]} aug task dirs -> workspace/core4d/results/E216/scene_snapshot/"
bash workspace/core4d/scripts/convert/snapshot_scenes.sh E216 "${TASKS[@]}" > /dev/null
echo "[E216] snapshot manifest: workspace/core4d/results/E216/scene_snapshot/manifest.txt"
