#!/usr/bin/env python3
"""Create E087 mass-scaled task clones for Box021."""

from __future__ import annotations

import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
TASK_ROOT = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
SOURCE_TASK = "d003_box021_20231018_029_p2_upperobj_e083"
TARGETS = {
    "d003_box021_20231018_029_p2_upperobj_e083_m5_e087": 5.0,
    "d003_box021_20231018_029_p2_upperobj_e083_m10_e087": 10.0,
}


def _patch_scene(scene: Path, new_mass: float) -> dict[str, object]:
    tree = ET.parse(scene)
    root = tree.getroot()
    obj = None
    for body in root.iter("body"):
        if body.get("name") == "object":
            obj = body
            break
    if obj is None:
        raise ValueError(f"object body not found in {scene}")
    inertial = obj.find("inertial")
    if inertial is None:
        raise ValueError(f"object inertial not found in {scene}")
    old_mass = float(inertial.get("mass", "0"))
    if old_mass <= 0:
        raise ValueError(f"invalid old mass {old_mass} in {scene}")
    old_inertia = [float(x) for x in inertial.get("diaginertia", "").split()]
    if len(old_inertia) != 3:
        raise ValueError(f"invalid old inertia in {scene}: {old_inertia}")
    scale = new_mass / old_mass
    new_inertia = [x * scale for x in old_inertia]
    inertial.set("mass", f"{new_mass:.6g}")
    inertial.set("diaginertia", " ".join(f"{x:.6g}" for x in new_inertia))
    tree.write(scene, encoding="utf-8", xml_declaration=False)
    return {
        "scene": str(scene.relative_to(REPO)),
        "old_mass_kg": old_mass,
        "new_mass_kg": new_mass,
        "old_diaginertia": old_inertia,
        "new_diaginertia": new_inertia,
        "scale": scale,
    }


def main() -> None:
    source = TASK_ROOT / SOURCE_TASK
    if not source.is_dir():
        raise FileNotFoundError(source)
    created: list[dict[str, object]] = []
    for task, mass in TARGETS.items():
        dst = TASK_ROOT / task
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(source, dst)
        patched = []
        for name in ("scene_act.xml", "scene.xml"):
            scene = dst / name
            if scene.is_file():
                patched.append(_patch_scene(scene, mass))
        meta = {
            "source_task": SOURCE_TASK,
            "target_task": task,
            "new_mass_kg": mass,
            "patched": patched,
        }
        (dst / "e087_mass_meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
        )
        created.append(meta)
        print(f"Created {dst.relative_to(REPO)} mass={mass:g}kg")

    out = REPO / "workspace/core4d/results/E087/mass_audit"
    out.mkdir(parents=True, exist_ok=True)
    (out / "mass_variant_meta.json").write_text(
        json.dumps(created, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Wrote {(out / 'mass_variant_meta.json').relative_to(REPO)}")


if __name__ == "__main__":
    main()

