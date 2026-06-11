#!/usr/bin/env python3
"""Build the E091 medium-box manifest for Holosoma data_construction_v2."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_V2_ROOT = Path("/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2")
DEFAULT_OLD_ROOT = Path("/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction")
DEFAULT_SPIDER_TASK_ROOT = Path(
    "/home/ubuntu/Workspace/spider/example_datasets/processed/core4d/unitree_g1/humanoid_object"
)

PERSON_SHORT = {"person1": "p1", "person2": "p2"}
PRIMARY_OBJECTS = {"box026"}
SECONDARY_OBJECTS = {"box004"}
STRESS_OBJECTS = {"box022"}
SELECTED_OBJECTS = PRIMARY_OBJECTS | SECONDARY_OBJECTS | STRESS_OBJECTS
STRESS_REVIEW_SEQUENCES = {"20231023/125", "20231023/126"}
BOX026_FIRST_BATCH = {
    "e091_box026_20231018_039_p2",
    "e091_box026_20231018_040_p2",
    "e091_box026_20231020_135_p2",
}
BOX026_SMOKE_FIRST = {"e091_box026_20231018_039_p2"}
BOX026_135_SINGLE = {"e091_box026_20231020_135_p2"}
BOX004_CONTROL_BATCH = {
    "e091_box004_20231003_2_083_p2",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_bool_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return "true"
    if text in {"false", "0", "no"}:
        return "false"
    return str(value)


def sequence_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["sequence"]), str(row["person"])


def source_scene_task(object_key: str, person: str, task_root: Path) -> tuple[str, bool]:
    person_task = f"{object_key}_{person}"
    if (task_root / person_task / "scene.xml").is_file():
        return person_task, True
    fallback = f"{object_key}_person1"
    if (task_root / fallback / "scene.xml").is_file():
        return fallback, True
    return person_task, False


def planned_target_task(row: dict[str, Any]) -> str:
    person_short = PERSON_SHORT.get(str(row["person"]), str(row["person"]))
    return f"e091_{row['object_key']}_{row['date']}_{row['seq']}_{person_short}"


def raw_proxy_path(row: dict[str, Any], old_root: Path) -> str:
    slug = f"{row['date']}_{row['seq']}_{row['object_key']}"
    path = old_root / "results/d002_stage1_raw_contact_v2/per_sequence" / slug / "raw_contact_proxy.npz"
    return str(path) if path.is_file() else ""


def classify_case(row0: dict[str, Any], row2: dict[str, Any] | None) -> tuple[str, str, int]:
    object_key = str(row0["object_key"])
    d001_group = str(row0.get("decision_group", ""))
    stage1_decision = str(row2.get("stage1_decision", "")) if row2 else ""
    if object_key in PRIMARY_OBJECTS and stage1_decision == "raw_contact_pass":
        return "primary_stage2b_template_backlog", "Box026 D002 pass; first C-route production target", 1
    if object_key in PRIMARY_OBJECTS and stage1_decision == "raw_contact_fail":
        return "hold_raw_contact_failed", "Box026 clean but D002 raw-contact failed; keep for evidence only", 0
    if object_key in SECONDARY_OBJECTS and stage1_decision == "raw_contact_pass":
        return "secondary_stage2b_template_backlog", "box004 D002 pass; lower-risk fill-in target", 2
    if object_key in STRESS_OBJECTS and d001_group == "边界/review":
        if str(row0["sequence"]) in STRESS_REVIEW_SEQUENCES:
            return "review_stress_test_selected", "Box022 review stress-test seed; not primary", 3
        return "review_stress_test_reserve", "Box022 review-only reserve; not primary", 4
    return "not_selected_for_e091", "Selected object but not in E091 entry queue", 9


def build_rows(old_root: Path, task_root: Path) -> list[dict[str, Any]]:
    d001 = read_json(old_root / "results/d001_stage0_inventory_v2/candidate_inventory_v2.json")
    d002 = read_json(old_root / "results/d002_stage1_raw_contact_v2/raw_contact_summary_v2.json")
    d002_idx = {sequence_key(row): row for row in d002}
    rows: list[dict[str, Any]] = []

    for row0 in d001:
        object_key = str(row0.get("object_key", ""))
        if object_key not in SELECTED_OBJECTS:
            continue
        row2 = d002_idx.get(sequence_key(row0))
        role, notes, priority = classify_case(row0, row2)
        source_task, source_exists = source_scene_task(object_key, str(row0["person"]), task_root)
        merged = row2 if row2 is not None else row0
        target_task = planned_target_task(row0)
        rows.append(
            {
                "priority": priority,
                "case_role": role,
                "sequence": row0["sequence"],
                "date": row0["date"],
                "seq": row0["seq"],
                "person": row0["person"],
                "object_name": row0["object_name"],
                "object_key": object_key,
                "object_category": row0.get("object_category", ""),
                "action": row0.get("action", ""),
                "size_band": row0.get("size_band", ""),
                "aabb_volume_m3": row0.get("aabb_volume_m3", ""),
                "extent_x_m": row0.get("extent_x_m", ""),
                "extent_y_m": row0.get("extent_y_m", ""),
                "extent_z_m": row0.get("extent_z_m", ""),
                "size_vs_box023_volume_ratio": row0.get("size_vs_box023_volume_ratio", ""),
                "size_vs_box025_volume_ratio": row0.get("size_vs_box025_volume_ratio", ""),
                "stage0_v2_decision": row0.get("stage0_v2_decision", ""),
                "stage0_decision_group": row0.get("decision_group", ""),
                "stage0_review_flags": row0.get("review_flags", ""),
                "stage0_route_tags": row0.get("route_tags", ""),
                "stage1_decision": row2.get("stage1_decision", "") if row2 else "not_run",
                "stage1_decision_group": row2.get("stage1_decision_group", "") if row2 else "",
                "stage1_raw_contact_score": row2.get("stage1_raw_contact_score", "") if row2 else "",
                "target_both_active_frac_3cm": row2.get("target_both_active_frac_3cm", "") if row2 else "",
                "target_left_active_frac_3cm": row2.get("target_left_active_frac_3cm", "") if row2 else "",
                "target_right_active_frac_3cm": row2.get("target_right_active_frac_3cm", "") if row2 else "",
                "target_both_longest_run_active_frac_3cm": row2.get(
                    "target_both_longest_run_active_frac_3cm", ""
                )
                if row2
                else "",
                "partner_any_active_frac_3cm": row2.get("partner_any_active_frac_3cm", "") if row2 else "",
                "partner_both_active_frac_3cm": row2.get("partner_both_active_frac_3cm", "") if row2 else "",
                "stage2_route": row2.get("stage2_route", "") if row2 else "review_not_in_d002",
                "has_template": as_bool_text(merged.get("has_template", "")),
                "has_existing_wbt_partner_npz": as_bool_text(merged.get("has_existing_wbt_partner_npz", "")),
                "object_model_rel": row0.get("object_mesh_rel", ""),
                "source_scene_task": source_task,
                "source_scene_exists": source_exists,
                "planned_target_task": target_task,
                "mask_slug": target_task,
                "raw_contact_proxy_path": raw_proxy_path(row0, old_root) if row2 else "",
                "notes": notes,
            }
        )

    rows.sort(
        key=lambda r: (
            int(r["priority"]),
            r["object_key"],
            -as_float(r["stage1_raw_contact_score"], -1.0),
            r["sequence"],
            r["person"],
        )
    )
    return rows


def pipeline_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if row["case_role"] not in {
            "primary_stage2b_template_backlog",
            "secondary_stage2b_template_backlog",
            "review_stress_test_selected",
        }:
            continue
        enabled = 1 if row["source_scene_exists"] else 0
        out.append(
            {
                "# enabled": enabled,
                "date": row["date"],
                "seq": row["seq"],
                "person": row["person"],
                "object_name": row["object_name"],
                "object_model_rel": row["object_model_rel"],
                "source_scene_task": row["source_scene_task"],
                "target_task": row["planned_target_task"],
                "trim_start": "auto",
                "trim_frames": "auto",
                "data_id": "0",
                "mask_slug": row["mask_slug"],
                "case_role": row["case_role"],
                "source_scene_exists": row["source_scene_exists"],
            }
        )
    return out


def write_pipeline_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "# enabled",
        "date",
        "seq",
        "person",
        "object_name",
        "object_model_rel",
        "source_scene_task",
        "target_task",
        "trim_start",
        "trim_frames",
        "data_id",
        "mask_slug",
        "case_role",
        "source_scene_exists",
    ]
    write_tsv(path, rows, fields)


def write_pipeline_case_file(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "# enabled",
        "date",
        "seq",
        "person",
        "object_name",
        "object_model_rel",
        "source_scene_task",
        "target_task",
        "trim_start",
        "trim_frames",
        "data_id",
        "mask_slug",
    ]
    slim_rows = [{field: row[field] for field in fields} for row in rows]
    write_tsv(path, slim_rows, fields)


def plot_inventory(rows: list[dict[str, Any]], out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    by_obj = Counter(row["object_key"] for row in rows)
    role_counts = Counter(row["case_role"] for row in rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    axes[0].bar(list(by_obj.keys()), list(by_obj.values()), color=["#4477aa", "#66ccee", "#cc6677"])
    axes[0].set_ylabel("case-person count")
    axes[0].set_title("E091 selected medium objects")
    axes[1].barh(list(role_counts.keys()), list(role_counts.values()), color="#228833")
    axes[1].set_xlabel("count")
    axes[1].set_title("E091 entry roles")
    path = out_dir / "medium_manifest_counts.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    x = [as_float(row["size_vs_box023_volume_ratio"]) for row in rows]
    y = [as_float(row["stage1_raw_contact_score"]) if row["stage1_raw_contact_score"] != "" else -5 for row in rows]
    colors = ["#4477aa" if row["object_key"] == "box026" else "#66ccee" if row["object_key"] == "box004" else "#cc6677" for row in rows]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.scatter(x, y, c=colors, alpha=0.8, edgecolors="black", linewidths=0.3)
    ax.axhline(0, color="0.8", linewidth=1)
    ax.set_xlabel("volume ratio vs Box023")
    ax.set_ylabel("D002 raw-contact score (-5 = not run)")
    ax.set_title("Size vs raw-contact readiness")
    path = out_dir / "medium_size_vs_raw_contact.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))
    return paths


def write_summary(root: Path, rows: list[dict[str, Any]], case_rows: list[dict[str, Any]], plots: list[str]) -> None:
    summary = {
        "stage": "E091 medium manifest",
        "v2_root": str(root),
        "total_manifest_rows": len(rows),
        "object_counts": dict(Counter(row["object_key"] for row in rows)),
        "case_role_counts": dict(Counter(row["case_role"] for row in rows)),
        "stage2b_case_rows": len(case_rows),
        "stage2b_enabled_now": sum(int(row["# enabled"]) for row in case_rows),
        "stage2b_template_backlog": sum(1 for row in case_rows if not bool(row["source_scene_exists"])),
        "plots": plots,
    }
    write_json(root / "results/medium_manifest/summary.json", summary)

    lines = [
        "# E091 Medium Box Manifest Summary",
        "",
        "## Counts",
        "",
        f"- Manifest rows: `{len(rows)}`",
        f"- Stage2b candidate rows: `{len(case_rows)}`",
        f"- Stage2b enabled now: `{summary['stage2b_enabled_now']}`",
        f"- Template backlog: `{summary['stage2b_template_backlog']}`",
        "",
        "Object counts:",
        "",
        "| object | count |",
        "|---|---:|",
    ]
    for key, count in Counter(row["object_key"] for row in rows).most_common():
        lines.append(f"| `{key}` | {count} |")
    lines.extend(["", "Case roles:", "", "| role | count |", "|---|---:|"])
    for role, count in Counter(row["case_role"] for row in rows).most_common():
        lines.append(f"| `{role}` | {count} |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- Box026 D002-pass rows are the primary C-route backlog.",
            "- box004 D002-pass rows are the secondary fill-in backlog.",
            "- Box022 remains review-only; selected rows are stress-test seeds, not primary data.",
            "- All Stage2b rows are disabled until source scene templates exist.",
            "",
            "## Visualizations",
            "",
        ]
    )
    lines.extend(f"- `{path}`" for path in plots)
    (root / "results/medium_manifest/summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-root", type=Path, default=DEFAULT_V2_ROOT)
    parser.add_argument("--old-root", type=Path, default=DEFAULT_OLD_ROOT)
    parser.add_argument("--spider-task-root", type=Path, default=DEFAULT_SPIDER_TASK_ROOT)
    args = parser.parse_args()

    root = args.v2_root
    for rel in [
        "inputs",
        "results/medium_manifest",
        "visualizations/dashboard",
        "visualizations/raw_contact",
        "reports",
        "logs",
        "scripts",
    ]:
        (root / rel).mkdir(parents=True, exist_ok=True)

    rows = build_rows(args.old_root, args.spider_task_root)
    fields = list(rows[0].keys()) if rows else []
    write_tsv(root / "inputs/medium_box_manifest.tsv", rows, fields)
    write_json(root / "inputs/medium_box_manifest.json", rows)
    write_tsv(root / "results/medium_manifest/medium_box_manifest.tsv", rows, fields)
    write_json(root / "results/medium_manifest/medium_box_manifest.json", rows)

    cases = pipeline_rows(rows)
    write_pipeline_tsv(root / "inputs/cases_stage2b_medium_backlog.tsv", cases)
    write_pipeline_case_file(root / "inputs/cases_stage2b_medium_pipeline.tsv", cases)
    write_pipeline_case_file(
        root / "inputs/cases_stage2b_box026_top3_pipeline.tsv",
        [row for row in cases if row["target_task"] in BOX026_FIRST_BATCH],
    )
    write_pipeline_case_file(
        root / "inputs/cases_stage2b_box026_first_pipeline.tsv",
        [row for row in cases if row["target_task"] in BOX026_SMOKE_FIRST],
    )
    write_pipeline_case_file(
        root / "inputs/cases_stage2b_box026_135_p2_pipeline.tsv",
        [row for row in cases if row["target_task"] in BOX026_135_SINGLE],
    )
    write_pipeline_case_file(
        root / "inputs/cases_stage2b_box004_control_pipeline.tsv",
        [row for row in cases if row["target_task"] in BOX004_CONTROL_BATCH],
    )
    write_json(root / "inputs/cases_stage2b_medium_backlog.json", cases)

    plots = plot_inventory(rows, root / "visualizations/dashboard")
    write_summary(root, rows, cases, plots)
    print(f"Wrote {len(rows)} manifest rows to {root / 'inputs/medium_box_manifest.tsv'}")
    print(f"Wrote {len(cases)} Stage2b backlog rows to {root / 'inputs/cases_stage2b_medium_backlog.tsv'}")
    print(f"Stage2b enabled now: {sum(int(row['# enabled']) for row in cases)}")


if __name__ == "__main__":
    main()
