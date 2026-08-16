#!/usr/bin/env python3
"""Pack a self-contained bundle for running the E199 full-scale machineB CEM
queue on a SECOND machine (separate filesystem).

The bundle contains every input the machineB rows need that is NOT part of a
standard SPIDER git checkout: the augmented task dirs (scene/trajectory/scene_act),
the E199 CEM override yamls (+ their base task yamls), the reused contact masks,
the object collision meshes (example_datasets/ is gitignored), the standalone
remote queue, the machineB manifest, a remote launch wrapper, and a README.

Assumes the remote already has, at the SAME repo-relative layout:
  - a working SPIDER repo + venv (.venv/bin/python with mujoco_warp/hydra),
  - examples/run_mjwp.py + examples/config base,
  - spider/assets/robots/unitree_g1 robot MJCF + meshes (tracked in git).

Extract the tarball at the remote repo root, then launch (see README).
"""

from __future__ import annotations

import csv
import io
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
MANIFEST = REPO / "workspace/core4d/results/E199/s6_downstream/manifests/e199_fullscale_machineB_manifest.tsv"
OBJECTS = ("box001", "box004", "box021", "box023", "box024")
OUT_DIR = REPO / "workspace/core4d/results/E199/s6_downstream/remote_bundle"

REMOTE_LAUNCH = """#!/usr/bin/env bash
# E199 machineB remote CEM launch. Run from the remote SPIDER repo ROOT after
# extracting the bundle here. Coexists with other GPU jobs; resume-safe.
# Usage: GPUS=0,1,2,3,4,5,6,7 bash run_E199_machineB_remote.sh
set -euo pipefail
PY="${PYTHON_BIN:-.venv/bin/python}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
MANIFEST="workspace/core4d/results/E199/s6_downstream/manifests/e199_fullscale_machineB_manifest.tsv"
Q="workspace/core4d/scripts/experiments/E199/run_E199_machineB_remote.py"
mkdir -p logs/E199/fullscale
echo "[machineB] launching CEM queue on GPUS=$GPUS"
exec "$PY" "$Q" --manifest "$MANIFEST" --gpus "$GPUS" \\
  --per-gpu-mem-mib "${PER_GPU_MEM_MIB:-5000}" --max-per-gpu "${MAX_PER_GPU:-1}" \\
  --poll-interval "${POLL_INTERVAL:-15}"
"""

README = """# E199 full-scale machineB remote bundle

Runs half of the E199 box translation-augmentation full-CEM (plan229) on a second
8-GPU machine, in parallel with the local machine (which runs machineA).

## Assumptions (remote must already have)
- A working SPIDER repo + venv at the repo root: `.venv/bin/python` with
  mujoco_warp / torch / hydra, plus `examples/run_mjwp.py` and `examples/config`.
- `spider/assets/robots/unitree_g1` robot MJCF + meshes (tracked in git).
- The SAME repo-relative directory layout as the local machine.

Everything else the CEM runs need is IN this bundle (aug task scenes/trajectories,
E199 overrides + base task yamls, contact masks, object meshes, the standalone
queue, and the machineB manifest).

## Steps
1. Copy the tarball to the remote repo ROOT and extract:
     tar xzf e199_machineB_bundle.tar.gz -C /path/to/remote/spider
2. Launch the queue (resume-safe, coexists with other jobs):
     cd /path/to/remote/spider
     GPUS=0,1,2,3,4,5,6,7 bash run_E199_machineB_remote.sh > logs/E199/fullscale/cem_machineB.log 2>&1 &
   (or: nohup ... &)
3. It writes CEM outputs under workspace/core4d/results/E199/s6_downstream/cem/full/
   and status back into the machineB manifest.

## Pull results back to the local machine
After the remote queue finishes (or periodically), copy the remote's CEM outputs
back so the local machine can run the combined eval:
     rsync -a REMOTE:/path/to/remote/spider/workspace/core4d/results/E199/s6_downstream/cem/full/ \\
       /LOCAL/spider/workspace/core4d/results/E199/s6_downstream/cem/full/
Only the machineB variant_id NPZs are new; filenames are unique per case+variant
so they never collide with machineA outputs.
"""


def main() -> int:
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8"), delimiter="\t"))
    if not rows:
        raise SystemExit(f"empty manifest: {MANIFEST}")

    members: set[Path] = set()

    def add(p: Path) -> None:
        if p.is_file():
            members.add(p)
        elif p.is_dir():
            for f in p.rglob("*"):
                if f.is_file():
                    members.add(f)

    # per-row inputs
    for r in rows:
        add(REPO / f"example_datasets/processed/core4d/unitree_g1/humanoid_object/{r['target_task']}")
        add(REPO / r["override_path"])
        add(REPO / f"examples/config/override/core4d_{r['target_task']}.yaml")
        add(REPO / r["contact_mask"])
    # object collision meshes (example_datasets is gitignored -> not on a fresh clone)
    for obj in OBJECTS:
        add(REPO / f"example_datasets/processed/core4d/assets/objects/{obj}")
    # standalone remote queue + this manifest
    add(REPO / "workspace/core4d/scripts/experiments/E199/run_E199_machineB_remote.py")
    add(MANIFEST)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tar_path = OUT_DIR / "e199_machineB_bundle.tar.gz"
    filelist = OUT_DIR / "e199_machineB_filelist.txt"
    total_bytes = 0
    with tarfile.open(tar_path, "w:gz") as tar:
        for p in sorted(members):
            arc = p.relative_to(REPO).as_posix()
            tar.add(p, arcname=arc)
            total_bytes += p.stat().st_size
        # inject launch wrapper + README at repo root
        for name, text in (("run_E199_machineB_remote.sh", REMOTE_LAUNCH),
                           ("E199_machineB_README.md", README)):
            data = text.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o755 if name.endswith(".sh") else 0o644
            tar.addfile(info, io.BytesIO(data))

    filelist.write_text("\n".join(sorted(m.relative_to(REPO).as_posix() for m in members)) + "\n",
                        encoding="utf-8")
    print(f"[pack] rows={len(rows)} files={len(members)} "
          f"raw={total_bytes/1e6:.1f}MB tar={tar_path.stat().st_size/1e6:.1f}MB")
    print(f"[pack] tarball  -> {tar_path.relative_to(REPO).as_posix()}")
    print(f"[pack] filelist -> {filelist.relative_to(REPO).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
