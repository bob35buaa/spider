#!/usr/bin/env python3
"""Build threshold-specific medium-box manifests from E104 D002 summaries."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


PERSON_SHORT = {"person1": "p1", "person2": "p2"}
PRIMARY_OBJECTS = {"box026"}
SECONDARY_OBJECTS = {"box004"}
STRESS_OBJECTS = {"box022"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def as_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def planned_target_task(row: dict[str, Any]) -> str:
    person_short = PERSON_SHORT.get(str(row["person"]), str(row["person"]))
    return f"e091_{row['object_key']}_{row['date']}_{row['seq']}_{person_short}"


def source_scene_task(object_key: str, person: str, task_root: Path) -> tuple[str, bool]:
    person_task = f"{object_key}_{person}"
    if (task_root / person_task / "scene.xml").is_file():
        return person_task, True
    fallback = f"{object_key}_person1"
    if (task_root / fallback / "scene.xml").is_file():
        return fallback, True
    return person_task, False


def raw_proxy_path(row: dict[str, Any], d002_root: Path) -> str:
    slug = str(row["sequence"]).replace("/", "_")
    path = d002_root / "per_sequence" / f"{slug}_{row['object_key']}" / "raw_contact_proxy.npz"
    return str(path) if path.is_file() else ""


def classify_case(row: dict[str, Any]) -> tuple[str, str, int]:
    object_key = str(row["object_key"])
    decision = str(row.get("stage1_decision", ""))
    label = str(row.get("contact_threshold_label", ""))
    if decision == "raw_contact_pass":
        if object_key in PRIMARY_OBJECTS:
            return f"primary_stage2b_{label}", f"{object_key} D002 {label} pass; threshold candidate", 1
        if object_key in SECONDARY_OBJECTS:
            return f"secondary_stage2b_{label}", f"{object_key} D002 {label} pass; threshold candidate", 2
        if object_key in STRESS_OBJECTS:
            return f"box022_stage2b_{label}", f"{object_key} D002 {label} pass; stress-test candidate", 3
    if decision == "raw_contact_review":
        return f"review_raw_contact_{label}", f"D002 {label} review; requires visual/fingertip review", 4
    if decision == "raw_contact_fail":
        return f"hold_raw_contact_failed_{label}", f"D002 {label} raw-contact failed", 8
    return f"hold_raw_contact_error_{label}", f"D002 {label} decision={decision}", 9


def build_manifest(rows: list[dict[str, str]], d002_root: Path, task_root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        object_key = str(row.get("object_key", ""))
        if object_key not in PRIMARY_OBJECTS | SECONDARY_OBJECTS | STRESS_OBJECTS:
            continue
        role, notes, priority = classify_case(row)
        source_task, source_exists = source_scene_task(object_key, str(row["person"]), task_root)
        target_task = planned_target_task(row)
        out.append(
            {
                "priority": priority,
                "case_role": role,
                "contact_threshold_label": row.get("contact_threshold_label", ""),
                "contact_threshold_m": row.get("contact_threshold_m", ""),
                "sequence": row["sequence"],
                "date": row["date"],
                "seq": row["seq"],
                "person": row["person"],
                "object_name": row["object_name"],
                "object_key": object_key,
                "object_category": row.get("object_category", ""),
                "action": row.get("action", ""),
                "size_band": row.get("size_band", ""),
                "aabb_volume_m3": row.get("aabb_volume_m3", ""),
                "extent_x_m": row.get("extent_x_m", ""),
                "extent_y_m": row.get("extent_y_m", ""),
                "extent_z_m": row.get("extent_z_m", ""),
                "size_vs_box023_volume_ratio": row.get("size_vs_box023_volume_ratio", ""),
                "size_vs_box025_volume_ratio": row.get("size_vs_box025_volume_ratio", ""),
                "stage0_v2_decision": row.get("stage0_v2_decision", ""),
                "stage0_decision_group": row.get("stage0_decision_group", ""),
                "stage0_review_flags": row.get("stage0_review_flags", ""),
                "stage0_route_tags": row.get("stage0_route_tags", ""),
                "stage1_decision": row.get("stage1_decision", ""),
                "stage1_decision_group": row.get("stage1_decision_group", ""),
                "stage1_raw_contact_score": row.get("stage1_raw_contact_score", ""),
                "target_both_active_frac": row.get("target_both_active_frac", ""),
                "target_left_active_frac": row.get("target_left_active_frac", ""),
                "target_right_active_frac": row.get("target_right_active_frac", ""),
                "target_both_longest_run_active_frac": row.get("target_both_longest_run_active_frac", ""),
                "partner_any_active_frac": row.get("partner_any_active_frac", ""),
                "partner_both_active_frac": row.get("partner_both_active_frac", ""),
                "target_both_active_frac_3cm": row.get("target_both_active_frac_3cm", ""),
                "target_both_active_frac_5cm": row.get("target_both_active_frac_5cm", ""),
                "target_both_longest_run_active_frac_3cm": row.get("target_both_longest_run_active_frac_3cm", ""),
                "target_both_longest_run_active_frac_5cm": row.get("target_both_longest_run_active_frac_5cm", ""),
                "stage2_route": row.get("stage2_route", ""),
                "has_template": row.get("has_template", ""),
                "has_existing_wbt_partner_npz": row.get("has_existing_wbt_partner_npz", ""),
                "object_model_rel": row.get("object_mesh_rel", ""),
                "source_scene_task": source_task,
                "source_scene_exists": str(source_exists),
                "planned_target_task": target_task,
                "mask_slug": target_task,
                "raw_contact_proxy_path": raw_proxy_path(row, d002_root),
                "notes": notes,
            }
        )
    out.sort(
        key=lambda r: (
            int(r["priority"]),
            r["object_key"],
            -as_float(r["stage1_raw_contact_score"], -1.0),
            r["sequence"],
            r["person"],
        )
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d002-root", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, default=Path("example_datasets/processed/core4d/unitree_g1/humanoid_object"))
    parser.add_argument("--out-dir", type=Path, default=Path("workspace/core4d/results/E104"))
    parser.add_argument("--threshold-labels", default="3cm,5cm")
    args = parser.parse_args()

    summaries: dict[str, Any] = {}
    for label in [x.strip() for x in args.threshold_labels.split(",") if x.strip()]:
        rows = read_tsv(args.d002_root / f"raw_contact_summary_v2_{label}.tsv")
        manifest = build_manifest(rows, args.d002_root, args.task_root)
        write_tsv(args.out_dir / f"medium_box_manifest_{label}.tsv", manifest)
        write_json(args.out_dir / f"medium_box_manifest_{label}.json", manifest)
        summaries[label] = {
            "rows": len(manifest),
            "decision_counts": dict(Counter(row["stage1_decision"] for row in manifest)),
            "case_role_counts": dict(Counter(row["case_role"] for row in manifest)),
            "object_counts": dict(Counter(row["object_key"] for row in manifest)),
        }
    write_json(args.out_dir / "medium_box_manifest_threshold_summary.json", summaries)
    print(json.dumps(summaries, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
