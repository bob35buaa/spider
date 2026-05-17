#!/usr/bin/env python3
"""Create E002 freejoint CORE4D tasks with leg/foot-object contact pairs.

The source tasks are left untouched. Each derived task copies only the
freejoint scene/data needed by `contact_guidance=false` and patches the derived
`scene.xml`.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
BASE = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
VARIANTS = REPO / "workspace/core4d_collab_retarget/scripts/E002/variants.tsv"

LEG_FOOT_GEOMS = [
    "left_hip_collision",
    "right_hip_collision",
    "left_thigh_collision",
    "right_thigh_collision",
    "left_shin_collision",
    "right_shin_collision",
    "left_linkage_brace_collision",
    "right_linkage_brace_collision",
    "lf0",
    "lf1",
    "lf2",
    "lf3",
    "rf0",
    "rf1",
    "rf2",
    "rf3",
]


def read_variants(path: Path) -> list[dict[str, str]]:
    fieldnames = [
        "variant",
        "source_task",
        "derived_task",
        "mask_source_exp",
        "mask_slug",
        "person_idx",
        "split",
        "role",
    ]
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(
            (line for line in f if line.strip() and not line.startswith("#")),
            delimiter="\t",
            fieldnames=fieldnames,
        )
        rows.extend(reader)
    return rows


def copy_freejoint_case(source_task: str, derived_task: str, *, force: bool) -> Path:
    src = BASE / source_task
    dst = BASE / derived_task
    if not src.is_dir():
        raise FileNotFoundError(src)
    if dst.exists():
        if not force:
            print(f"[SKIP] derived task exists: {dst.relative_to(REPO)}")
            return dst
        shutil.rmtree(dst)

    (dst / "0").mkdir(parents=True, exist_ok=True)
    for rel in ["scene.xml", "task_info.json", "0/trajectory_kinematic.npz"]:
        src_file = src / rel
        if not src_file.is_file():
            raise FileNotFoundError(src_file)
        dst_file = dst / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)
    return dst


def patch_scene(scene_xml: Path, source_task: str, derived_task: str) -> list[str]:
    text = scene_xml.read_text(encoding="utf-8")
    marker = "  </contact>"
    if marker not in text:
        raise ValueError(f"Missing {marker!r} in {scene_xml}")
    if "scene_act" in scene_xml.name:
        raise ValueError(f"E002 must patch freejoint scene.xml, got {scene_xml}")

    added: list[str] = []
    lines: list[str] = []
    for geom in LEG_FOOT_GEOMS:
        pair_name = f"{geom}_object"
        if f'name="{pair_name}"' in text:
            continue
        lines.append(
            f'    <pair name="{pair_name}" geom1="{geom}" geom2="object_collision" '
            'solref="0.008 1" friction="1 1" condim="3" />'
        )
        added.append(pair_name)
    if lines:
        text = text.replace(marker, "\n".join(lines) + "\n" + marker, 1)
        scene_xml.write_text(text, encoding="utf-8")

    meta = {
        "source_task": source_task,
        "derived_task": derived_task,
        "patched_file": str(scene_xml.relative_to(REPO)),
        "scene_mode": "freejoint_scene_xml",
        "object_geom": "object_collision",
        "added_pairs": added,
        "note": "E002 derived freejoint scene; source task and scene_act.xml are not modified.",
    }
    (scene_xml.parent / "freejoint_leg_object_collision_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
    )
    return added


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", type=Path, default=VARIANTS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for row in read_variants(args.variants):
        dst = copy_freejoint_case(row["source_task"], row["derived_task"], force=args.force)
        added = patch_scene(dst / "scene.xml", row["source_task"], row["derived_task"])
        print(
            f"{row['source_task']} -> {row['derived_task']}: "
            f"added_pairs={len(added)} path={dst.relative_to(REPO)}"
        )


if __name__ == "__main__":
    main()
