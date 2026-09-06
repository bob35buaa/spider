#!/usr/bin/env bash
# E212 P5 -- rules/experiment.md §7 Safeguard 2.
#
# Snapshots the 4 desk023 dcv3 *task* dirs (NOT the `<obj>_person<N>/` source
# templates -- E206's own snapshot made that mistake and therefore does not
# contain the scene its CEM actually ran).  One pass captures all five points of
# the g-curve: the E206 PRG base (g=0), the E209 gravcomp sidecar (g=1) and the
# three E212 partial sidecars, so the whole comparison stays recoverable even if
# the working copies are later overwritten.
#
# Only shard A's machine runs this.  Both machines share /mnt, and two concurrent
# snapshots would race on the same output dir; run_E212_local_8gpu.sh therefore
# takes SKIP_SNAPSHOT=1 on the remote side and asserts this snapshot already
# exists at the current git HEAD before dispatching.
#
# Usage:
#   bash workspace/core4d/scripts/experiments/E212/snapshot_E212_scenes.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

mapfile -t TASKS < <(
  MUJOCO_GL=disable .venv/bin/python - <<'PY'
import sys
sys.path.insert(0, "workspace/core4d/scripts/experiments/E212")
import e212_common as C
for row in C.sources():
    print(C.target_task(row))
PY
)

if [ "${#TASKS[@]}" -ne 4 ]; then
  echo "ERROR: expected 4 desk023 task dirs, got ${#TASKS[@]}" >&2
  exit 1
fi

bash workspace/core4d/scripts/convert/snapshot_scenes.sh E212 "${TASKS[@]}" > /dev/null

MANIFEST="workspace/core4d/results/E212/scene_snapshot/manifest.txt"

# Exit check: every task dir must carry all five arms' scenes, and the three
# E212 hashes must match the CEM manifest -- otherwise the snapshot documents a
# different run than the one about to be dispatched.
MUJOCO_GL=disable .venv/bin/python - "$MANIFEST" <<'PY'
import csv
import sys
from pathlib import Path

sys.path.insert(0, "workspace/core4d/scripts/experiments/E212")
import e212_common as C

manifest = Path(sys.argv[1])
snap: dict[tuple[str, str], str] = {}
for line in manifest.read_text(encoding="utf-8").splitlines():
    if line.startswith("#") or not line.strip() or line.split()[0] == "case":
        continue
    case, fname, _size, sha = line.split()
    snap[(case, fname)] = sha

with C.manifest_path("full").open(encoding="utf-8", newline="") as stream:
    cem = {(r["case_id"], r["arm"]): r for r in csv.DictReader(stream, delimiter="\t")}

problems = []
for row in C.sources():
    case_id, task = row["case_id"], C.target_task(row)
    want = {f"{C.BASE_SCENE}.xml": cem[(case_id, C.ARM_ORDER[0])]["baseline_scene_sha256"]}
    for arm in C.ARM_ORDER:
        want[f"{C.SCENE_BY_ARM[arm]}.xml"] = cem[(case_id, arm)]["effective_scene_sha256"]
    # g=1 endpoint: present in the snapshot, but its hash authority is E209's
    # manifest, not ours -- assert presence only.
    for fname, expect in want.items():
        got = snap.get((task, fname))
        if got is None:
            problems.append(f"{case_id}: {fname} absent from snapshot")
        elif got != expect:
            problems.append(f"{case_id}: {fname} sha {got[:12]} != manifest {expect[:12]}")
    if (task, f"{C.G1_SCENE}.xml") not in snap:
        problems.append(f"{case_id}: {C.G1_SCENE}.xml (g=1 endpoint) absent from snapshot")

if problems:
    print("\n".join(f"  FAIL {p}" for p in problems))
    raise SystemExit(f"P5 snapshot check failed ({len(problems)} problems)")

tasks = {c for c, _ in snap}
print(f"P5 PASS: {len(tasks)} task dirs snapshotted; all 5 g-curve scenes present, "
      f"E212 hashes matched to the CEM manifest")
print(f"  {manifest}")
PY
