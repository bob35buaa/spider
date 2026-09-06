#!/usr/bin/env bash
# E209 P4 -- rules/experiment.md §7 Safeguard 2.
#
# Snapshots the dcv3 *task* dirs (NOT the `<obj>_person<N>/` source templates).
# This matters: E206's own snapshot captured the source templates and therefore
# does NOT contain `scene_act_E206_lowgeom_PRG.xml`, the scene its CEM actually
# ran.  Snapshotting the task dirs the way E207 did captures BOTH arms in one
# pass -- the E206 PRG baseline and the E209 gravcomp sidecar -- so the whole
# paired comparison stays recoverable even if the working copies are overwritten.
#
# Usage:
#   bash workspace/core4d/scripts/experiments/E209/snapshot_E209_scenes.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

mapfile -t TASKS < <(
  MUJOCO_GL=disable .venv/bin/python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "workspace/core4d/scripts/experiments/E209")
import e209_common as C
for row in C.sources():
    print(C.target_task(row))
PY
)

if [ "${#TASKS[@]}" -ne 22 ]; then
  echo "ERROR: expected 22 task dirs, got ${#TASKS[@]}" >&2
  exit 1
fi

bash workspace/core4d/scripts/convert/snapshot_scenes.sh E209 "${TASKS[@]}" > /dev/null

MANIFEST="workspace/core4d/results/E209/scene_snapshot/manifest.txt"

# Exit check: every task dir must carry BOTH arms' scenes, and their hashes must
# match the CEM manifest -- otherwise the snapshot documents a different run.
MUJOCO_GL=disable .venv/bin/python - "$MANIFEST" <<'PY'
import csv
import sys
from pathlib import Path

sys.path.insert(0, "workspace/core4d/scripts/experiments/E209")
import e209_common as C

manifest = Path(sys.argv[1])
snap: dict[tuple[str, str], str] = {}
for line in manifest.read_text(encoding="utf-8").splitlines():
    if line.startswith("#") or not line.strip() or line.split()[0] == "case":
        continue
    case, fname, _size, sha = line.split()
    snap[(case, fname)] = sha

with (C.MANIFEST).open(encoding="utf-8", newline="") as stream:
    cem = {r["case_id"]: r for r in csv.DictReader(stream, delimiter="\t")}

problems = []
for row in C.sources():
    case_id, task = row["case_id"], C.target_task(row)
    for fname, want in (
        (f"{C.BASE_SCENE}.xml", cem[case_id]["baseline_scene_sha256"]),
        (f"{C.SCENE}.xml", cem[case_id]["effective_scene_sha256"]),
    ):
        got = snap.get((task, fname))
        if got is None:
            problems.append(f"{case_id}: {fname} absent from snapshot")
        elif got != want:
            problems.append(f"{case_id}: {fname} sha {got[:12]} != manifest {want[:12]}")

if problems:
    print("\n".join(f"  FAIL {p}" for p in problems))
    raise SystemExit(f"P4 snapshot check failed ({len(problems)} problems)")

tasks = {c for c, _ in snap}
print(f"P4 PASS: {len(tasks)} task dirs snapshotted, both arms' scenes sha-matched to the CEM manifest")
print(f"  {manifest}")
PY
