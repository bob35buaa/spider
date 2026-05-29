#!/usr/bin/env python3
"""Build the E093 contact-geometry audit manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[4]
PROC = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
OUT_ROOT = REPO / "workspace/core4d/results/E093/contact_geometry"
V3_ROOT = Path("/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2")


CASES: list[dict[str, Any]] = [
    {
        "case_id": "box023_p2",
        "object_group": "box023",
        "task": "box023_person2",
        "person_idx": 1,
        "role": "known_guard",
        "mask_path": REPO / "workspace/core4d/results/E079/contact_masks/box023_person2/raw_contact_mask_3cm.npz",
    },
    {
        "case_id": "box025_p2",
        "object_group": "box025",
        "task": "box025_person2",
        "person_idx": 1,
        "role": "large_box_partial_guard",
        "mask_path": REPO / "workspace/core4d/results/E080/contact_masks/box025_person2/raw_contact_mask_3cm.npz",
    },
    {
        "case_id": "box004_083_p2",
        "object_group": "box004",
        "task": "e091_box004_20231003_2_083_p2",
        "person_idx": 1,
        "role": "E092_C1_WORK",
        "mask_path": V3_ROOT
        / "results/stage2b_medium/results/contact_masks/e091_box004_20231003_2_083_p2/raw_contact_mask_3cm.npz",
    },
    {
        "case_id": "box021_person1",
        "object_group": "box021",
        "task": "box021_person1",
        "person_idx": 0,
        "role": "old_box021_control",
        "mask_path": REPO / "workspace/core4d/results/E079/contact_masks/box021_person1/raw_contact_mask_3cm.npz",
    },
    {
        "case_id": "box021_d003_029_p2",
        "object_group": "box021",
        "task": "d003_box021_20231018_029_p2",
        "person_idx": 1,
        "role": "known_D003_fail",
        "mask_path": REPO
        / "workspace/core4d/results/E084/contact_masks/d003_box021_20231018_029_p2/raw_contact_mask_3cm.npz",
    },
    {
        "case_id": "box026_039_p2",
        "object_group": "box026",
        "task": "e091_box026_20231018_039_p2",
        "person_idx": 1,
        "role": "E092_C2_FAIL_low_support",
        "mask_path": V3_ROOT
        / "results/stage2b_medium/results/contact_masks/e091_box026_20231018_039_p2/raw_contact_mask_3cm.npz",
    },
    {
        "case_id": "box026_135_p2",
        "object_group": "box026",
        "task": "e091_box026_20231020_135_p2",
        "person_idx": 1,
        "role": "E092_C3_FAIL_inside_risk",
        "mask_path": V3_ROOT
        / "results/stage2b_medium/results/contact_masks/e091_box026_20231020_135_p2/raw_contact_mask_3cm.npz",
    },
]


FIELDS = [
    "case_id",
    "object_group",
    "task",
    "person_idx",
    "role",
    "task_dir",
    "scene_xml",
    "trajectory_npz",
    "mask_path",
    "audit_summary",
    "scene_exists",
    "trajectory_exists",
    "mask_exists",
    "audit_exists",
    "ready",
]


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in CASES:
        task_dir = PROC / spec["task"]
        scene = task_dir / "scene.xml"
        traj = task_dir / "0/trajectory_kinematic.npz"
        mask = Path(spec["mask_path"])
        audit = mask.parent / "audit_summary_3cm.json"
        row = {
            **spec,
            "task_dir": str(task_dir),
            "scene_xml": str(scene),
            "trajectory_npz": str(traj),
            "mask_path": str(mask),
            "audit_summary": str(audit),
            "scene_exists": scene.is_file(),
            "trajectory_exists": traj.is_file(),
            "mask_exists": mask.is_file(),
            "audit_exists": audit.is_file(),
        }
        row["ready"] = bool(row["scene_exists"] and row["trajectory_exists"] and row["mask_exists"] and row["audit_exists"])
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--force", action="store_true", help="Overwrite existing manifest")
    args = parser.parse_args()

    manifest = args.out_root / "case_manifest.tsv"
    if manifest.exists() and not args.force:
        raise FileExistsError(f"{manifest} exists; pass --force")

    rows = build_rows()
    write_tsv(manifest, rows, FIELDS)
    write_json(args.out_root / "case_manifest.json", {"rows": rows})
    ready = sum(1 for row in rows if row["ready"])
    print(f"Wrote {manifest}")
    print(f"Ready rows: {ready}/{len(rows)}")
    for row in rows:
        status = "READY" if row["ready"] else "MISSING"
        print(f"{status}\t{row['case_id']}\t{row['task']}")


if __name__ == "__main__":
    main()
