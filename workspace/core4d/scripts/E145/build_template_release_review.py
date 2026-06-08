#!/usr/bin/env python3
"""Build E145 explicit non-box template release TSV from E144 review evidence."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[4]
DEFAULT_E144_ROOT = REPO / "workspace/core4d/results/E144/E144_full_nonbox_raw_contact"
DEFAULT_E145_ROOT = REPO / "workspace/core4d/results/E145/full_nonbox_to_rl_ready"
REVIEWER = "codex_local_review_20260605"
REVIEW_SOURCE = "E144_logs_181_183_mesh_collision_overlay_review"

FIELDS = [
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
    "mesh_collision_evidence_video",
    "mesh_collision_evidence_sheet",
    "orbit_evidence_video",
    "orbit_evidence_sheet",
    "review_source",
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


def existing_path(path_text: str) -> str:
    if not path_text:
        return ""
    path = Path(path_text)
    if not path.is_absolute():
        path = REPO / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def policy_for(row: dict[str, str], mesh_row: dict[str, str]) -> str:
    return row.get("approved_collision_policy") or row.get("draft_policy") or mesh_row.get("proxy_policy", "")


def mass_policy_for(row: dict[str, str]) -> str:
    if row.get("approved_mass_policy"):
        return row["approved_mass_policy"]
    return "current_scene_object_mass_reviewed_proxy"


def note_for(task: str, category: str, policy: str) -> str:
    if category in {"desk", "chair"}:
        return (
            "APPROVE_CLEAN: E144 local review accepted tight surface-voxel multi-box proxy; "
            "semantic desk/chair proxy was rejected; overlay/object-only sheets reviewed; "
            f"policy={policy}; source={REVIEW_SOURCE}; task={task}."
        )
    if category == "bucket":
        return (
            "APPROVE_CLEAN: E144 local review accepted bucket wall proxy or repaired draft; "
            f"mesh/collision overlay reviewed; policy={policy}; source={REVIEW_SOURCE}; task={task}."
        )
    return f"APPROVE_CLEAN: E144 local review accepted non-box proxy; policy={policy}; task={task}."


def build_rows(e144_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    old_review = {row["source_scene_task"]: row for row in read_tsv(e144_root / "s2_templates/nonbox_template_review.tsv")}
    draft = {row["source_scene_task"]: row for row in read_tsv(e144_root / "s2_templates/draft_proxy/draft_proxy_review_queue.tsv")}
    mesh_rows = read_tsv(e144_root / "s2_templates_mesh_collision_review/mesh_collision_review_manifest.tsv")
    if not mesh_rows:
        raise FileNotFoundError(e144_root / "s2_templates_mesh_collision_review/mesh_collision_review_manifest.tsv")

    rows: list[dict[str, Any]] = []
    missing_orbit: list[str] = []
    for mesh in mesh_rows:
        task = mesh["source_scene_task"]
        old = old_review.get(task, {})
        draft_row = draft.get(task, {})
        source = {**old, **draft_row}
        category = mesh.get("object_category") or source.get("object_category", "")
        policy = policy_for(source, mesh)
        if mesh.get("render_status") != "pass":
            raise ValueError(f"mesh/collision render is not pass for {task}: {mesh.get('render_status')}")
        orbit_video = source.get("orbit_evidence_video") or old.get("evidence_video", "")
        orbit_sheet = source.get("orbit_evidence_sheet") or old.get("evidence_sheet", "")
        if not orbit_video or not orbit_sheet:
            missing_orbit.append(task)
            continue
        scene_xml = source.get("scene_xml") or old.get("proxy_scene_xml") or mesh.get("scene_xml", "")
        rows.append(
            {
                "source_scene_task": task,
                "object_key": mesh.get("object_key") or source.get("object_key", ""),
                "person": mesh.get("person") or source.get("person", ""),
                "object_category": category,
                "proxy_scene_xml": existing_path(scene_xml),
                "review_decision": "approve_clean",
                "reviewer": REVIEWER,
                "review_notes": note_for(task, category, policy),
                "approved_collision_policy": policy,
                "approved_mass_policy": mass_policy_for(source),
                "evidence_video": existing_path(orbit_video),
                "evidence_sheet": existing_path(orbit_sheet),
                "mesh_collision_evidence_video": existing_path(mesh.get("video_path", "")),
                "mesh_collision_evidence_sheet": existing_path(mesh.get("sheet_path", "")),
                "orbit_evidence_video": existing_path(orbit_video),
                "orbit_evidence_sheet": existing_path(orbit_sheet),
                "review_source": REVIEW_SOURCE,
            }
        )
    if missing_orbit:
        raise FileNotFoundError("missing orbit evidence for: " + ",".join(sorted(missing_orbit)))
    rows.sort(key=lambda row: (row["object_category"], row["object_key"], row["person"]))
    return rows, sorted(set(mesh_rows[0]) if mesh_rows else [])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e144-root", type=Path, default=DEFAULT_E144_ROOT)
    parser.add_argument("--e145-root", type=Path, default=DEFAULT_E145_ROOT)
    args = parser.parse_args()

    e144_root = args.e144_root if args.e144_root.is_absolute() else REPO / args.e144_root
    e145_root = args.e145_root if args.e145_root.is_absolute() else REPO / args.e145_root
    rows, _ = build_rows(e144_root)
    out_tsv = e145_root / "s2_templates/nonbox_template_review.tsv"
    write_tsv(out_tsv, rows, FIELDS)
    summary = {
        "source_e144_root": str(e144_root),
        "e145_root": str(e145_root),
        "review_tsv": str(out_tsv),
        "rows": len(rows),
        "decision_counts": dict(Counter(row["review_decision"] for row in rows)),
        "category_counts": dict(Counter(row["object_category"] for row in rows)),
        "policy_counts": dict(Counter(row["approved_collision_policy"] for row in rows)),
        "reviewer": REVIEWER,
        "review_source": REVIEW_SOURCE,
    }
    write_json(e145_root / "s2_templates/nonbox_template_review_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
