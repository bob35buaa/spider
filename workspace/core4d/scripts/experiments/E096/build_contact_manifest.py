#!/usr/bin/env python3
"""Build E096 contact-semantics manifest for the three E095 box004 cases."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[4]
TASK_ROOT = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
MASK_ROOT = Path(
    "/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2/results/stage2b_medium/results/contact_masks"
)
OUT_ROOT = REPO / "workspace/core4d/results/E096/contact_semantics"

CASES = [
    {
        "case_id": "box004_083_p1",
        "object_group": "box004",
        "task": "e091_box004_20231003_2_083_p1",
        "person_idx": "0",
        "role": "E096_P1_first_batch_ready",
        "split": "local",
    },
    {
        "case_id": "box004_082_p1",
        "object_group": "box004",
        "task": "e091_box004_20231003_2_082_p1",
        "person_idx": "0",
        "role": "E096_P2_first_batch_ready",
        "split": "remote-gpu0",
    },
    {
        "case_id": "box004_082_p2",
        "object_group": "box004",
        "task": "e091_box004_20231003_2_082_p2",
        "person_idx": "1",
        "role": "E096_P3_preprocess_infeasible",
        "split": "remote-gpu1",
    },
]

FIELDS = [
    "case_id",
    "object_group",
    "task",
    "person_idx",
    "role",
    "split",
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
    "blocker",
]


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def build_rows(mask_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in CASES:
        task = case["task"]
        task_dir = TASK_ROOT / task
        scene = task_dir / "scene.xml"
        trajectory = task_dir / "0/trajectory_kinematic.npz"
        mask = mask_root / task / "raw_contact_mask_3cm.npz"
        audit = mask_root / task / "audit_summary_3cm.json"
        checks = {
            "scene_exists": scene.is_file(),
            "trajectory_exists": trajectory.is_file(),
            "mask_exists": mask.is_file(),
            "audit_exists": audit.is_file(),
        }
        missing = [name for name, ok in checks.items() if not ok]
        row = {
            **case,
            "task_dir": str(task_dir),
            "scene_xml": str(scene),
            "trajectory_npz": str(trajectory),
            "mask_path": str(mask),
            "audit_summary": str(audit),
            **checks,
            "ready": not missing,
            "blocker": "none" if not missing else "missing " + ",".join(missing),
        }
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-root", type=Path, default=MASK_ROOT)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    args = parser.parse_args()

    rows = build_rows(args.mask_root)
    write_tsv(args.out_root / "case_manifest.tsv", rows)
    write_json(args.out_root / "case_manifest.json", rows)
    write_json(
        args.out_root / "readiness_summary.json",
        {
            "num_cases": len(rows),
            "num_ready": sum(bool(row["ready"]) for row in rows),
            "ready_cases": [row["case_id"] for row in rows if row["ready"]],
            "blocked_cases": {row["case_id"]: row["blocker"] for row in rows if not row["ready"]},
        },
    )
    for row in rows:
        state = "READY" if row["ready"] else f"BLOCKED {row['blocker']}"
        print(f"{row['case_id']} {row['task']} {state}")
    print(args.out_root / "case_manifest.tsv")


if __name__ == "__main__":
    main()
