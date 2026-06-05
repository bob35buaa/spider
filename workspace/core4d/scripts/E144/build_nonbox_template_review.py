#!/usr/bin/env python3
"""Build the E144 non-box template review TSV from auditable local evidence."""

from __future__ import annotations

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[4]
DEFAULT_SOURCE_RUN_ROOT = REPO / "workspace/core4d/results/E144/E144_full_nonbox_raw_contact"
REVIEW_FIELDS = [
    "source_scene_task",
    "object_key",
    "person",
    "object_category",
    "proxy_scene_xml",
    "review_decision",
    "reviewer",
    "review_notes",
    "approved_collision_policy",
    "approved_mass_policy",
    "evidence_video",
    "evidence_sheet",
]


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        lines = [line for line in f if line.strip() and not line.startswith("#")]
        if not lines:
            return []
        return list(csv.DictReader(lines, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def bucket_wall_proxy_ok(scene_xml: str) -> tuple[bool, str]:
    scene = Path(scene_xml)
    if not scene.is_file():
        return False, "scene_xml_missing"
    try:
        root = ET.parse(scene).getroot()
    except Exception as exc:  # noqa: BLE001
        return False, f"xml_parse_error:{type(exc).__name__}"
    object_body = next((body for body in root.iter("body") if body.get("name") == "object"), None)
    if object_body is None:
        return False, "object_body_missing"
    collision_geoms = [
        geom
        for geom in object_body.iter("geom")
        if geom.get("name", "").startswith("object_collision") and geom.get("contype", "1") != "0"
    ]
    names = {geom.get("name", "") for geom in collision_geoms}
    required = {
        "object_collision",
        "object_collision_bucket_xneg",
        "object_collision_bucket_xpos",
        "object_collision_bucket_yneg",
        "object_collision_bucket_ypos",
    }
    missing = sorted(required - names)
    if missing:
        return False, "bucket_wall_proxy_missing_geoms:" + ",".join(missing)
    bad_types = sorted(geom.get("name", "") for geom in collision_geoms if geom.get("type", "") != "box")
    if bad_types:
        return False, "bucket_wall_proxy_non_box_geoms:" + ",".join(bad_types)
    return True, "bucket_wall_proxy_aabb_bottom_plus_four_walls"


def review_row(
    row: dict[str, str],
    visual: dict[str, str],
    reviewer: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    task = row["source_scene_task"]
    category = row.get("object_category", "")
    scene_xml = row.get("scene_xml", "")
    render_status = visual.get("render_status", "missing_visual_row")
    video = visual.get("video_path", "")
    sheet = visual.get("sheet_path", "")
    wall_ok, wall_note = bucket_wall_proxy_ok(scene_xml) if category == "bucket" else (False, "")
    checks = {
        "source_scene_task": task,
        "object_category": category,
        "render_status": render_status,
        "mujoco_load_ok": row.get("mujoco_load_ok", ""),
        "inertial_status": row.get("inertial_status", ""),
        "robot_polluted_mass_29_632": row.get("robot_polluted_mass_29_632", ""),
        "bucket_wall_proxy_ok": str(wall_ok),
        "bucket_wall_proxy_note": wall_note,
        "template_adapter": row.get("template_adapter", ""),
        "backlog_collision_policy": row.get("collision_policy", ""),
        "build_status": row.get("build_status", ""),
        "required_by_count": row.get("required_by_count", ""),
        "required_by_cases": row.get("required_by_cases", ""),
    }
    approved = (
        category == "bucket"
        and render_status == "pass"
        and row.get("mujoco_load_ok") == "True"
        and row.get("inertial_status") == "clean"
        and row.get("robot_polluted_mass_29_632") == "False"
        and wall_ok
    )
    if approved:
        decision = "approve_clean"
        notes = (
            "APPROVE_CLEAN: local Codex review verified MuJoCo load, clean inertials, "
            "rendered template sheet, and XML bottom+four-wall bucket proxy collision. "
            f"required_by_count={row.get('required_by_count', '')}."
        )
        collision_policy = "bucket_wall_proxy_aabb"
        mass = row.get("object_mass", "")
        mass_policy = f"current_scene_object_mass_{mass}kg_reviewed_proxy" if mass else "current_scene_object_mass_reviewed_proxy"
    else:
        decision = "needs_manual_edit"
        reasons: list[str] = []
        if category in {"desk", "chair"}:
            reasons.append("complex_shape_not_auto_reviewed")
        if category == "bucket" and not wall_ok:
            reasons.append(wall_note)
        if render_status != "pass":
            reasons.append(f"render_status={render_status}")
        if row.get("mujoco_load_ok") != "True":
            reasons.append(f"mujoco_load_ok={row.get('mujoco_load_ok', '')}")
        if row.get("inertial_status") != "clean":
            reasons.append(f"inertial_status={row.get('inertial_status', '')}")
        if row.get("robot_polluted_mass_29_632") == "True":
            reasons.append("robot_polluted_mass_29_632")
        if not reasons:
            reasons.append("not_covered_by_local_bucket_proxy_review")
        notes = "NEEDS_MANUAL_EDIT: " + ";".join(reasons)
        collision_policy = ""
        mass_policy = ""
    review = {
        "source_scene_task": task,
        "object_key": row.get("object_key", ""),
        "person": row.get("person", ""),
        "object_category": category,
        "proxy_scene_xml": scene_xml,
        "review_decision": decision,
        "reviewer": reviewer,
        "review_notes": notes,
        "approved_collision_policy": collision_policy,
        "approved_mass_policy": mass_policy,
        "evidence_video": video,
        "evidence_sheet": sheet,
    }
    return review, checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-root", type=Path, default=DEFAULT_SOURCE_RUN_ROOT)
    parser.add_argument("--out-tsv", type=Path, default=None)
    parser.add_argument("--reviewer", default="codex_local_review_20260605")
    args = parser.parse_args()

    source_root = args.source_run_root if args.source_run_root.is_absolute() else REPO / args.source_run_root
    template_dir = source_root / "s2_templates"
    backlog_path = template_dir / "template_backlog.tsv"
    visual_path = template_dir / "template_visual_review/template_visual_manifest.tsv"
    out_tsv = args.out_tsv or template_dir / "nonbox_template_review.tsv"
    if not backlog_path.is_file():
        raise SystemExit(f"missing template backlog: {backlog_path}")
    if not visual_path.is_file():
        raise SystemExit(f"missing template visual manifest: {visual_path}")

    visual_index = {row["source_scene_task"]: row for row in read_tsv(visual_path)}
    reviews: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    for row in read_tsv(backlog_path):
        review, check = review_row(row, visual_index.get(row["source_scene_task"], {}), args.reviewer)
        reviews.append(review)
        checks.append(check)
    reviews.sort(key=lambda item: (item["review_decision"] != "approve_clean", item["object_category"], item["object_key"], item["person"]))
    checks.sort(key=lambda item: item["source_scene_task"])

    write_tsv(out_tsv, reviews, REVIEW_FIELDS)
    check_fields = sorted({key for row in checks for key in row})
    write_tsv(template_dir / "nonbox_template_review_checks.tsv", checks, check_fields)
    summary = {
        "source_run_root": str(source_root),
        "review_tsv": str(out_tsv),
        "rows": len(reviews),
        "decision_counts": dict(Counter(row["review_decision"] for row in reviews)),
        "approved_templates": [row["source_scene_task"] for row in reviews if row["review_decision"] == "approve_clean"],
        "blocked_templates": [row["source_scene_task"] for row in reviews if row["review_decision"] != "approve_clean"],
    }
    write_json(template_dir / "nonbox_template_review_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
