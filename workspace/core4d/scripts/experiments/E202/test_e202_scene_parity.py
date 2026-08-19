#!/usr/bin/env python3
"""Geometry-parity self-test for the E202 bucket collision builder (no GPU/conda).

Reproduces the E178 collision scene for one orig case from its standard
`scene_act.xml`, then asserts that the object_collision geoms + robot-object
pairs match E178's snapshotted `scene_act_E178_contactAlignedTop.xml` (modulo the
cosmetic pair-name prefix). Proves that reusing the E175/E177/E178 geometry code
on an (aug or orig) task reproduces the E178 collision body by construction.

Run: .venv/bin/python workspace/core4d/scripts/experiments/E202/test_e202_scene_parity.py
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e202_common as C  # noqa: E402

CASE_ID = "bucket003_20231018_001_p1"
OBJECT_KEY = "bucket003"
TASK = "dcv3_omnirt_v1_ref_fk_bucket003_20231018_001_p1"


def object_geoms(root: ET.Element) -> list[tuple]:
    body = C.GEOM.find_object_body(root)
    out = []
    for geom in C.GEOM.object_collision_elements(body):
        out.append((
            geom.get("name"), geom.get("type", "box"),
            geom.get("size", ""), geom.get("pos", ""), geom.get("quat", ""),
        ))
    return sorted(out)


def robot_object_pairs(root: ET.Element) -> list[tuple]:
    contact = root.find("contact")
    pairs = []
    for pair in (contact.findall("pair") if contact is not None else []):
        if C.GEOM.is_robot_object_pair(pair):
            # ignore the name (E178_ vs E202_ prefix); physics = geoms+condim+solref+friction
            pairs.append((
                pair.get("geom1"), pair.get("geom2"), pair.get("condim"),
                pair.get("solref"), pair.get("friction", ""),
                pair.get("margin", ""), pair.get("gap", ""),
            ))
    return sorted(pairs)


def main() -> int:
    task_dir = C.TASK_ROOT / TASK
    base_scene_act = task_dir / "scene_act.xml"
    trajectory = task_dir / "0/trajectory_kinematic.npz"
    e178_snap = (
        C.REPO / "workspace/core4d/results/E178/scene_snapshot/semantic_bucket_proxy"
        / CASE_ID / "scene_act_E178_contactAlignedTop.xml"
    )
    for p in (base_scene_act, trajectory, e178_snap):
        if not p.is_file():
            raise SystemExit(f"missing input: {p}")

    print(f"[build] E202 collision scene for {CASE_ID} from {base_scene_act.name}")
    info = C.build_prg_scene(CASE_ID, base_scene_act, trajectory,
                             overwrite=True, object_key=OBJECT_KEY)
    produced = task_dir / f"{C.SCENE_NAME}.xml"
    print(f"  geom_count={info['object_geom_count']} pair_count={info['compiled_robot_object_pair_count']} "
          f"mesh->proxy_p90={info['mesh_to_proxy_p90_m']:.4f} "
          f"ref_first5_min={info['reference_first5_min_lowerbody_object_distance_m']:.4f}")

    e202_root = ET.parse(produced).getroot()
    e178_root = ET.parse(e178_snap).getroot()

    g202, g178 = object_geoms(e202_root), object_geoms(e178_root)
    p202, p178 = robot_object_pairs(e202_root), robot_object_pairs(e178_root)

    ok = True
    if g202 != g178:
        ok = False
        print(f"  [FAIL] object geoms differ: E202={len(g202)} E178={len(g178)}")
        for a, b in zip(g202, g178):
            if a != b:
                print(f"    E202 {a}\n    E178 {b}")
    else:
        print(f"  [PASS] object_collision geoms identical ({len(g202)} geoms)")

    if p202 != p178:
        ok = False
        print(f"  [FAIL] robot-object pairs differ: E202={len(p202)} E178={len(p178)}")
        only202 = [x for x in p202 if x not in p178][:5]
        only178 = [x for x in p178 if x not in p202][:5]
        print(f"    only in E202: {only202}")
        print(f"    only in E178: {only178}")
    else:
        print(f"  [PASS] robot-object pairs identical ({len(p202)} pairs)")

    # cleanup the E202 sidecars written into the historical orig task dir
    for name in (f"{C.SCENE_NAME}.xml", f"{C.RUBBER_INTERMEDIATE}.xml"):
        (task_dir / name).unlink(missing_ok=True)
    print("  [cleanup] removed E202 test sidecars from orig task dir")

    print("PARITY:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
