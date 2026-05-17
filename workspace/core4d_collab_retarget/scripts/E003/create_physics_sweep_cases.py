#!/usr/bin/env python3
"""Create E003 true-freejoint physics-sweep CORE4D tasks.

Each E003 task is copied from an E002 freejoint+leg-object derived task, then
only the copied scene.xml is patched. Source tasks are left untouched.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable


REPO = Path(__file__).resolve().parents[4]
BASE = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
VARIANTS = REPO / "workspace/core4d_collab_retarget/scripts/E003/variants.tsv"

FIELDNAMES = [
    "variant",
    "source_task",
    "derived_task",
    "mask_source_exp",
    "mask_slug",
    "person_idx",
    "remote_group",
    "role",
    "mass_kg",
    "inertia_scale",
    "hand_object_friction",
    "object_floor_friction",
]


def read_variants(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(
            (line for line in f if line.strip() and not line.startswith("#")),
            delimiter="\t",
            fieldnames=FIELDNAMES,
        )
        rows.extend(reader)
    return rows


def copy_case(source_task: str, derived_task: str, *, force: bool) -> Path:
    src = BASE / source_task
    dst = BASE / derived_task
    if not src.is_dir():
        raise FileNotFoundError(src)
    if dst.exists():
        if not force:
            print(f"[SKIP] derived task exists: {dst.relative_to(REPO)}")
            return dst
        shutil.rmtree(dst)

    shutil.copytree(src, dst)
    return dst


def _find_parent(root: ET.Element, child: ET.Element) -> ET.Element | None:
    for parent in root.iter():
        if child in list(parent):
            return parent
    return None


def _parse_floats(value: str) -> list[float]:
    return [float(part) for part in value.split()]


def _fmt(value: float) -> str:
    return f"{value:.9g}"


def _fmt_vec(values: Iterable[float]) -> str:
    return " ".join(_fmt(value) for value in values)


def _pair_by_name(root: ET.Element, name: str) -> ET.Element:
    for pair in root.iter("pair"):
        if pair.get("name") == name:
            return pair
    raise ValueError(f"Missing contact pair {name!r}")


def _object_geom(root: ET.Element) -> ET.Element:
    for geom in root.iter("geom"):
        if geom.get("name") == "object_collision":
            return geom
    raise ValueError("Missing geom name='object_collision'")


def patch_scene(scene_xml: Path, row: dict[str, str]) -> dict[str, object]:
    tree = ET.parse(scene_xml)
    root = tree.getroot()

    object_geom = _object_geom(root)
    object_body = _find_parent(root, object_geom)
    if object_body is None or object_body.tag != "body":
        raise ValueError(f"Could not locate object body in {scene_xml}")
    inertial = object_body.find("inertial")
    if inertial is None:
        raise ValueError(f"Object body has no inertial in {scene_xml}")

    old_mass = float(inertial.get("mass", "nan"))
    old_inertia = _parse_floats(inertial.get("diaginertia", ""))
    new_mass = float(row["mass_kg"])
    inertia_scale = float(row["inertia_scale"])
    new_inertia = [value * inertia_scale for value in old_inertia]
    inertial.set("mass", _fmt(new_mass))
    inertial.set("diaginertia", _fmt_vec(new_inertia))

    hand_friction = float(row["hand_object_friction"])
    floor_friction = float(row["object_floor_friction"])
    for pair_name in ("left_hand_object", "right_hand_object"):
        pair = _pair_by_name(root, pair_name)
        pair.set("friction", f"{_fmt(hand_friction)} 1")
    floor_pair = _pair_by_name(root, "object_floor")
    floor_pair.set("friction", f"{_fmt(floor_friction)} 1")

    # Keep geom default consistent with the explicit object-floor pair.
    geom_friction = _parse_floats(object_geom.get("friction", "1 0.005 0.0001"))
    if geom_friction:
        geom_friction[0] = floor_friction
        object_geom.set("friction", _fmt_vec(geom_friction))

    ET.indent(tree, space="  ")
    tree.write(scene_xml, encoding="utf-8", xml_declaration=False)

    meta: dict[str, object] = {
        "variant": row["variant"],
        "source_task": row["source_task"],
        "derived_task": row["derived_task"],
        "patched_file": str(scene_xml.relative_to(REPO)),
        "scene_mode": "freejoint_scene_xml",
        "object_geom": "object_collision",
        "old_mass_kg": old_mass,
        "new_mass_kg": new_mass,
        "old_diaginertia": old_inertia,
        "new_diaginertia": new_inertia,
        "inertia_scale": inertia_scale,
        "hand_object_friction": hand_friction,
        "object_floor_friction": floor_friction,
        "note": "E003 true-freejoint physics sweep; source E002 derived task is not modified.",
    }
    (scene_xml.parent / "physics_sweep_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
    )
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", type=Path, default=VARIANTS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for row in read_variants(args.variants):
        dst = copy_case(row["source_task"], row["derived_task"], force=args.force)
        meta = patch_scene(dst / "scene.xml", row)
        print(
            f"{row['source_task']} -> {row['derived_task']}: "
            f"mass={meta['new_mass_kg']} inertia_scale={meta['inertia_scale']} "
            f"hand_friction={meta['hand_object_friction']} "
            f"floor_friction={meta['object_floor_friction']} path={dst.relative_to(REPO)}"
        )


if __name__ == "__main__":
    main()
