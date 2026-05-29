#!/usr/bin/env python3
"""Mine worklike medium-box candidates after E092/E094 failure analysis."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_OLD_ROOT = Path("/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction")
DEFAULT_V2_ROOT = Path("/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2")
DEFAULT_SPIDER_TASK_ROOT = Path(
    "/home/ubuntu/Workspace/spider/example_datasets/processed/core4d/unitree_g1/humanoid_object"
)
DEFAULT_OUT_ROOT = Path("workspace/core4d/results/E095/worklike_candidate_mining")

PERSON_SHORT = {"person1": "p1", "person2": "p2"}
KNOWN_WORK = {"e091_box004_20231003_2_083_p2"}
BOX004_PRIORITY = {
    "e091_box004_20231003_2_083_p1",
    "e091_box004_20231003_2_082_p1",
    "e091_box004_20231003_2_082_p2",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def planned_target(row: dict[str, Any]) -> str:
    return f"e091_{row['object_key']}_{row['date']}_{row['seq']}_{PERSON_SHORT.get(row['person'], row['person'])}"


def source_scene_task(object_key: str, person: str, task_root: Path) -> tuple[str, bool]:
    task = f"{object_key}_{person}"
    return task, (task_root / task / "scene.xml").is_file()


def raw_proxy_path(row: dict[str, Any], old_root: Path) -> str:
    slug = f"{row['date']}_{row['seq']}_{row['object_key']}"
    path = old_root / "results/d002_stage1_raw_contact_v2/per_sequence" / slug / "raw_contact_proxy.npz"
    return str(path) if path.is_file() else ""


def risk_label(row: dict[str, Any], d2: dict[str, Any] | None) -> tuple[str, str]:
    obj = str(row["object_key"])
    action = str(row.get("action", ""))
    if obj == "box004":
        return "positive_pattern", "closest to E092/E094 WORK pattern"
    if obj == "box021":
        return "review_after_target_gate", "smaller than Box026 but prior D003/Box021 CEM failures require target/posture gate"
    if obj == "box026":
        return "deprioritize_box026", "recent E092/E094 full CEM failures despite raw-contact pass; large reach/wrong-face risk"
    if obj == "box022":
        return "needs_raw_contact_and_reach_review", "long dimension is larger than Box026; no D002 raw-contact evidence yet"
    if action.startswith("pass") or action == "rot":
        return "reject_action_family", "non-lift/move action family"
    if d2 is None:
        return "needs_raw_contact", "not in D002 raw-contact queue"
    return "generic_review", "not matched to known positive pattern"


def score_row(row: dict[str, Any], d2: dict[str, Any] | None, task_root: Path) -> tuple[float, str, str, str]:
    target = planned_target(row)
    obj = str(row["object_key"])
    volume_ratio = as_float(row.get("size_vs_box023_volume_ratio"), 999.0)
    max_extent = max(
        as_float(row.get("extent_x_m")),
        as_float(row.get("extent_y_m")),
        as_float(row.get("extent_z_m")),
    )
    min_extent = min(
        as_float(row.get("extent_x_m")),
        as_float(row.get("extent_y_m")),
        as_float(row.get("extent_z_m")),
    )
    aspect = max_extent / max(min_extent, 1e-6)
    raw_score = as_float(d2.get("stage1_raw_contact_score") if d2 else "", -20.0)
    both = as_float(d2.get("target_both_active_frac_3cm") if d2 else "", 0.0)
    longest = as_float(d2.get("target_both_longest_run_active_frac_3cm") if d2 else "", 0.0)
    source_task, source_ready = source_scene_task(obj, str(row["person"]), task_root)

    # box004/box023-like: close to box004 volume ratio 1.138, not just "between box023 and box025".
    size_score = 25.0 * clamp(1.0 - abs(volume_ratio - 1.138) / 1.15)
    raw_component = 0.35 * clamp(raw_score, 0.0, 100.0)
    contact_component = 18.0 * clamp(both) + 10.0 * clamp(longest)
    source_component = 4.0 if source_ready else 0.0
    long_extent_penalty = max(0.0, max_extent - 0.50) * 90.0
    aspect_penalty = max(0.0, aspect - 1.8) * 12.0

    score = size_score + raw_component + contact_component + source_component
    score -= long_extent_penalty + aspect_penalty

    if obj == "box004":
        score += 18.0
    elif obj == "box021":
        score -= 8.0
    elif obj == "box026":
        score -= 42.0
    elif obj == "box022":
        score -= 28.0

    if target in KNOWN_WORK:
        tier = "tier0_known_work"
    elif target in BOX004_PRIORITY:
        tier = "tier1_box004_priority"
    elif obj == "box004":
        tier = "tier1_box004_extra"
    elif obj == "box021" and d2 and d2.get("stage1_decision") == "raw_contact_pass":
        tier = "tier2_box021_review_after_target_gate"
    elif obj == "box022":
        tier = "tier3_box022_needs_raw_contact"
    elif obj == "box026":
        tier = "tier4_box026_deprioritized"
    else:
        tier = "tier5_other_review"

    risk, note = risk_label(row, d2)
    return round(score, 3), tier, risk, note


def build_rows(old_root: Path, task_root: Path) -> list[dict[str, Any]]:
    d001 = read_json(old_root / "results/d001_stage0_inventory_v2/candidate_inventory_v2.json")
    d002 = read_json(old_root / "results/d002_stage1_raw_contact_v2/raw_contact_summary_v2.json")
    d2_idx = {(row["sequence"], row["person"]): row for row in d002}
    rows: list[dict[str, Any]] = []

    for row in d001:
        obj = str(row.get("object_key", ""))
        d2 = d2_idx.get((row["sequence"], row["person"]))
        include = False
        if d2 and d2.get("stage1_decision") == "raw_contact_pass" and obj in {"box004", "box021", "box026"}:
            include = True
        if obj == "box022" and row.get("stage0_v2_decision") == "stage0_boundary_review_to_stage1":
            include = True
        if not include:
            continue

        target = planned_target(row)
        source_task, source_ready = source_scene_task(obj, str(row["person"]), task_root)
        score, tier, risk, note = score_row(row, d2, task_root)
        rows.append(
            {
                "rank": 0,
                "target_task": target,
                "tier": tier,
                "score": score,
                "risk_label": risk,
                "object_key": obj,
                "object_name": row.get("object_name", ""),
                "sequence": row["sequence"],
                "date": row["date"],
                "seq": row["seq"],
                "person": row["person"],
                "action": row.get("action", ""),
                "aabb_volume_m3": row.get("aabb_volume_m3", ""),
                "size_vs_box023_volume_ratio": row.get("size_vs_box023_volume_ratio", ""),
                "extent_x_m": row.get("extent_x_m", ""),
                "extent_y_m": row.get("extent_y_m", ""),
                "extent_z_m": row.get("extent_z_m", ""),
                "stage0_v2_decision": row.get("stage0_v2_decision", ""),
                "stage1_decision": d2.get("stage1_decision", "not_run") if d2 else "not_run",
                "stage1_raw_contact_score": d2.get("stage1_raw_contact_score", "") if d2 else "",
                "target_both_active_frac_3cm": d2.get("target_both_active_frac_3cm", "") if d2 else "",
                "target_both_longest_run_active_frac_3cm": d2.get("target_both_longest_run_active_frac_3cm", "")
                if d2
                else "",
                "object_model_rel": row.get("object_mesh_rel", ""),
                "source_scene_task": source_task,
                "source_scene_exists": source_ready,
                "raw_contact_proxy_path": raw_proxy_path(row, old_root) if d2 else "",
                "learned_note": note,
            }
        )

    tier_order = {
        "tier0_known_work": 0,
        "tier1_box004_priority": 1,
        "tier1_box004_extra": 2,
        "tier2_box021_review_after_target_gate": 3,
        "tier3_box022_needs_raw_contact": 4,
        "tier4_box026_deprioritized": 5,
        "tier5_other_review": 6,
    }
    rows.sort(key=lambda r: (tier_order.get(r["tier"], 99), -float(r["score"]), r["target_task"]))
    for i, row in enumerate(rows, start=1):
        row["rank"] = i
    return rows


def pipeline_rows(rows: list[dict[str, Any]], targets: set[str], enabled_default: int = 1) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if row["target_task"] not in targets:
            continue
        out.append(
            {
                "# enabled": enabled_default,
                "date": row["date"],
                "seq": row["seq"],
                "person": row["person"],
                "object_name": row["object_name"],
                "object_model_rel": row["object_model_rel"],
                "source_scene_task": row["source_scene_task"],
                "target_task": row["target_task"],
                "trim_start": "auto",
                "trim_frames": "auto",
                "data_id": "0",
                "mask_slug": row["target_task"],
            }
        )
    return out


def write_summary(path: Path, rows: list[dict[str, Any]], box004_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# E095 Worklike Candidate Mining",
        "",
        "## Summary",
        "",
        f"- Candidate rows: `{len(rows)}`",
        f"- Box004 priority Stage2b rows: `{len(box004_rows)}`",
        "",
        "Tier counts:",
        "",
        "| tier | count |",
        "|---|---:|",
    ]
    for tier, count in Counter(row["tier"] for row in rows).most_common():
        lines.append(f"| `{tier}` | {count} |")
    lines.extend(
        [
            "",
            "## First Batch",
            "",
            "| rank | target | score | raw score | source scene | note |",
            "|---:|---|---:|---:|---|---|",
        ]
    )
    for row in box004_rows:
        lines.append(
            f"| {row['rank']} | `{row['target_task']}` | {row['score']} | "
            f"{row['stage1_raw_contact_score']} | `{row['source_scene_task']}` | {row['learned_note']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `tier1_box004_priority` is the only batch to run immediately.",
            "- `tier2_box021_review_after_target_gate` is held for a later target/posture-gated route because D003/Box021 has repeated CEM failures.",
            "- `tier4_box026_deprioritized` remains in the bank for traceability but is not a first-batch data source after E092/E094 failures.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-root", type=Path, default=DEFAULT_OLD_ROOT)
    parser.add_argument("--v2-root", type=Path, default=DEFAULT_V2_ROOT)
    parser.add_argument("--spider-task-root", type=Path, default=DEFAULT_SPIDER_TASK_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    args = parser.parse_args()

    rows = build_rows(args.old_root, args.spider_task_root)
    fields = list(rows[0].keys()) if rows else []
    out_root = args.out_root
    v2_out = args.v2_root / "results/e095_worklike_candidates"
    for root in (out_root, v2_out):
        write_tsv(root / "worklike_candidate_bank.tsv", rows, fields)
        write_json(root / "worklike_candidate_bank.json", rows)

    box004 = [row for row in rows if row["target_task"] in BOX004_PRIORITY]
    case_fields = [
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
    box004_cases = pipeline_rows(rows, BOX004_PRIORITY)
    for path in (
        out_root / "cases_e095_box004_priority_pipeline.tsv",
        args.v2_root / "inputs/cases_e095_box004_priority_pipeline.tsv",
    ):
        write_tsv(path, box004_cases, case_fields)

    box021_targets = {row["target_task"] for row in rows if row["tier"] == "tier2_box021_review_after_target_gate"}
    write_tsv(out_root / "cases_e095_box021_review_disabled.tsv", pipeline_rows(rows, box021_targets, 0), case_fields)

    write_summary(out_root / "summary.md", rows, box004)
    write_summary(v2_out / "summary.md", rows, box004)
    write_json(
        out_root / "summary.json",
        {
            "rows": len(rows),
            "tier_counts": dict(Counter(row["tier"] for row in rows)),
            "box004_priority_targets": [row["target_task"] for row in box004],
            "box004_case_file": str(args.v2_root / "inputs/cases_e095_box004_priority_pipeline.tsv"),
        },
    )
    print(f"Wrote {len(rows)} candidates")
    print(f"Wrote {len(box004_cases)} box004 priority cases")
    print(args.v2_root / "inputs/cases_e095_box004_priority_pipeline.tsv")


if __name__ == "__main__":
    main()
