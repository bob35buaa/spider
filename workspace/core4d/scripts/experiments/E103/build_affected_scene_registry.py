#!/usr/bin/env python3
"""Build E103 affected scene registry and summary from audit TSVs."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def classify_task(task: str) -> str:
    canonical = {
        "box021_person1",
        "box021_person2",
        "box022_person1",
        "box022_person2",
        "box026_person1",
        "box026_person2",
        "box004_person1",
        "box004_person2",
        "box023_person1",
    }
    if task in canonical:
        return "canonical_source_template"
    if task.startswith("d003_box021_") or task.startswith("e091_box026_"):
        return "target_or_experiment_derived"
    if "_e0" in task or "_upperobj" in task or "_legobj" in task or "_freejoint" in task:
        return "experiment_derived"
    return "other_scene"


def validity_action(in_status: str, geom_status: str, task_class: str) -> str:
    if "polluted_robot_inertial" in in_status:
        if task_class == "canonical_source_template":
            return "rebuild_source_template"
        return "quarantine_invalidated_by_scene_inertial_bug"
    if geom_status != "clean" or "object_mass_policy_review" in in_status:
        return "review_before_use"
    return "keep_clean"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inertial", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    inertial = {row["task"]: row for row in read_tsv(args.inertial)}
    geometry = {row["task"]: row for row in read_tsv(args.geometry)}
    rows: list[dict[str, str]] = []

    expected_canonical = {
        "box021_person1",
        "box021_person2",
        "box022_person1",
        "box022_person2",
        "box026_person1",
        "box026_person2",
        "box004_person1",
        "box004_person2",
        "box023_person1",
    }

    for task in sorted(set(inertial) | set(geometry) | expected_canonical):
        ir = inertial.get(task, {})
        gr = geometry.get(task, {})
        task_class = classify_task(task)
        if task_class == "canonical_source_template" and task not in inertial and task not in geometry:
            action = "rebuild_missing_source_template"
        else:
            action = validity_action(
                ir.get("status", "missing_inertial_audit"),
                gr.get("status", "missing_geometry_audit"),
                task_class,
            )
        rows.append(
            {
                "task": task,
                "task_class": task_class,
                "action": action,
                "inertial_status": ir.get("status", ""),
                "geometry_status": gr.get("status", ""),
                "scene_xml": ir.get("scene_xml") or gr.get("scene_xml", ""),
                "robot_inertial_unique_pairs": ir.get("robot_inertial_unique_pairs", ""),
                "pelvis_mass": ir.get("pelvis_mass", ""),
                "object_mass": ir.get("object_mass") or gr.get("object_mass", ""),
                "object_mesh_path": gr.get("object_mesh_path", ""),
                "object_collision_half_extents_m": gr.get("object_collision_half_extents_m", ""),
                "expected_collision_half_extents_m": gr.get("expected_collision_half_extents_m", ""),
                "collision_max_rel_error": gr.get("collision_max_rel_error", ""),
                "notes": "",
            }
        )

    fields = [
        "task",
        "task_class",
        "action",
        "inertial_status",
        "geometry_status",
        "scene_xml",
        "robot_inertial_unique_pairs",
        "pelvis_mass",
        "object_mass",
        "object_mesh_path",
        "object_collision_half_extents_m",
        "expected_collision_half_extents_m",
        "collision_max_rel_error",
        "notes",
    ]
    write_tsv(args.out, rows, fields)

    existing_scene_count = len(set(inertial) | set(geometry))
    missing_canonical_count = len(
        [task for task in expected_canonical if task not in inertial and task not in geometry]
    )
    action_counts = Counter(row["action"] for row in rows)
    prefix_counts = Counter(row["task"].split("_")[0] for row in rows if row["action"] != "keep_clean")
    class_counts = Counter(row["task_class"] for row in rows)
    canonical_rebuild = [
        row["task"]
        for row in rows
        if row["action"] in {"rebuild_source_template", "rebuild_missing_source_template"}
    ]

    lines = [
        "# E103 Scene Audit Summary",
        "",
        f"- Existing scene audit rows: `{existing_scene_count}`",
        f"- Expected missing canonical templates: `{missing_canonical_count}`",
        f"- Total registry rows: `{len(rows)}`",
        "",
        "Action counts:",
        "",
        "| action | count |",
        "|---|---:|",
    ]
    for action, count in action_counts.most_common():
        lines.append(f"| `{action}` | {count} |")
    lines.extend(["", "Task class counts:", "", "| class | count |", "|---|---:|"])
    for cls, count in class_counts.most_common():
        lines.append(f"| `{cls}` | {count} |")
    lines.extend(["", "Non-clean prefix counts:", "", "| prefix | count |", "|---|---:|"])
    for prefix, count in prefix_counts.most_common():
        lines.append(f"| `{prefix}` | {count} |")
    lines.extend(
        [
            "",
            "Canonical source templates requiring rebuild:",
            "",
        ]
    )
    if canonical_rebuild:
        lines.extend(f"- `{task}`" for task in canonical_rebuild)
    else:
        lines.append("- None in the current registry.")
    lines.extend(
        [
            "",
            "Decision:",
            "",
            "- `quarantine_invalidated_by_scene_inertial_bug` rows must not be used as dynamics labels.",
            "- Canonical source templates must be clean before any Box021/Box022/Box026 target regeneration.",
            "- Clean rows can remain as positive/negative guards only if their target-specific evidence is otherwise valid.",
        ]
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"wrote {args.summary}")


if __name__ == "__main__":
    main()
