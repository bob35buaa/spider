#!/usr/bin/env python3
"""Build E103 source-template preflight table from live scenes and audits."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SCENE_ROOT = Path("example_datasets/processed/core4d/unitree_g1/humanoid_object")
DEFAULT_TASKS = [
    "box021_person1",
    "box021_person2",
    "box022_person1",
    "box022_person2",
    "box026_person1",
    "box026_person2",
]


def read_tsv(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        return {row["task"]: row for row in csv.DictReader(f, delimiter="\t")}


def write_tsv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def source_policy(task: str) -> tuple[str, str]:
    p = "person1" if task.endswith("person1") else "person2"
    obj = task.split("_")[0]
    return obj, p


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inertial", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, default=SCENE_ROOT)
    parser.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    args = parser.parse_args()

    inertial = read_tsv(args.inertial)
    geometry = read_tsv(args.geometry)
    registry = read_tsv(args.registry)
    rows = []
    for task in args.tasks:
        scene = args.scene_root / task / "scene.xml"
        task_info = args.scene_root / task / "task_info.json"
        obj, person = source_policy(task)
        ir = inertial.get(task, {})
        gr = geometry.get(task, {})
        rr = registry.get(task, {})
        ok = (
            scene.is_file()
            and task_info.is_file()
            and ir.get("status") == "clean"
            and gr.get("status") == "clean"
            and rr.get("action") == "keep_clean"
        )
        rows.append(
            {
                "source_scene_task": task,
                "object_key": obj,
                "person": person,
                "scene_xml": str(scene),
                "scene_xml_exists": str(scene.is_file()),
                "task_info": str(task_info),
                "task_info_exists": str(task_info.is_file()),
                "registry_action": rr.get("action", ""),
                "inertial_status": ir.get("status", ""),
                "geometry_status": gr.get("status", ""),
                "robot_inertial_unique_pairs": ir.get("robot_inertial_unique_pairs", ""),
                "pelvis_mass": ir.get("pelvis_mass", ""),
                "object_mass": ir.get("object_mass") or gr.get("object_mass", ""),
                "collision_max_rel_error": gr.get("collision_max_rel_error", ""),
                "preflight": "PASS" if ok else "FAIL",
                "notes": "clean canonical source template" if ok else "requires review before target regeneration",
            }
        )

    fields = [
        "source_scene_task",
        "object_key",
        "person",
        "scene_xml",
        "scene_xml_exists",
        "task_info",
        "task_info_exists",
        "registry_action",
        "inertial_status",
        "geometry_status",
        "robot_inertial_unique_pairs",
        "pelvis_mass",
        "object_mass",
        "collision_max_rel_error",
        "preflight",
        "notes",
    ]
    write_tsv(args.out, rows, fields)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["preflight"]] = counts.get(row["preflight"], 0) + 1
    print(f"wrote {args.out} rows={len(rows)} counts={counts}")


if __name__ == "__main__":
    main()
