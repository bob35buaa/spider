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
FIRST_BATCH_TARGETS = {
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


def geometry_features(row: dict[str, Any]) -> tuple[float, float, float]:
    volume_ratio = as_float(row.get("size_vs_box023_volume_ratio"), 999.0)
    extents = [
        as_float(row.get("extent_x_m")),
        as_float(row.get("extent_y_m")),
        as_float(row.get("extent_z_m")),
    ]
    max_extent = max(extents)
    min_extent = min(extents)
    aspect = max_extent / max(min_extent, 1e-6)
    return volume_ratio, max_extent, aspect


def route_bucket(target: str, row: dict[str, Any], d2: dict[str, Any] | None) -> str:
    volume_ratio, max_extent, aspect = geometry_features(row)
    raw_pass = bool(d2 and d2.get("stage1_decision") == "raw_contact_pass")

    if target in KNOWN_WORK:
        return "tier0_known_work"
    if target in FIRST_BATCH_TARGETS:
        return "tier1_worklike_priority"
    if raw_pass and volume_ratio <= 1.45 and max_extent <= 0.50 and aspect <= 1.80:
        return "tier1_worklike_extra"
    if d2 is None or not raw_pass:
        if max_extent >= 0.60 or volume_ratio >= 2.0:
            return "tier3_missing_raw_contact_long_edge_review"
        return "tier5_other_review"
    if max_extent >= 0.55 or volume_ratio >= 2.50:
        return "tier4_large_reach_dynamics_holdout"
    if volume_ratio >= 1.45 or max_extent >= 0.48 or aspect >= 1.75:
        return "tier2_target_posture_gate_review"
    return "tier5_other_review"


def risk_label(row: dict[str, Any], d2: dict[str, Any] | None) -> tuple[str, str]:
    target = planned_target(row)
    volume_ratio, max_extent, aspect = geometry_features(row)
    action = str(row.get("action", ""))
    if action.startswith("pass") or action == "rot":
        return "reject_action_family", "non-lift/move action family"
    if target in KNOWN_WORK:
        return "known_positive_control", "previous full CEM WORK control"
    if target in FIRST_BATCH_TARGETS:
        return "worklike_priority", "box004/box023-scale geometry selected for first-batch execution"
    if d2 is None:
        if max_extent >= 0.60 or volume_ratio >= 2.0:
            return (
                "missing_raw_contact_long_edge_review",
                f"missing D002 raw-contact; long edge {max_extent:.3f}m / volume ratio {volume_ratio:.2f} needs review",
            )
        return "needs_raw_contact", "not in D002 raw-contact queue"
    if max_extent >= 0.55 or volume_ratio >= 2.50:
        return (
            "large_reach_dynamics_holdout",
            f"large geometry ({max_extent:.3f}m max edge, volume ratio {volume_ratio:.2f}); recent large-box CEM failures require separate repair route",
        )
    if volume_ratio >= 1.45 or max_extent >= 0.48 or aspect >= 1.75:
        return (
            "target_posture_gate_review",
            f"medium-large geometry ({max_extent:.3f}m max edge, volume ratio {volume_ratio:.2f}, aspect {aspect:.2f}); run target/posture gate before CEM",
        )
    return "generic_review", "not matched to first-batch worklike geometry"


def score_row(row: dict[str, Any], d2: dict[str, Any] | None) -> tuple[float, str, str, str]:
    target = planned_target(row)
    volume_ratio, max_extent, aspect = geometry_features(row)
    raw_score = as_float(d2.get("stage1_raw_contact_score") if d2 else "", -20.0)
    both = as_float(d2.get("target_both_active_frac_3cm") if d2 else "", 0.0)
    longest = as_float(d2.get("target_both_longest_run_active_frac_3cm") if d2 else "", 0.0)

    # box004/box023-like: close to box004 volume ratio 1.138, not just "between box023 and box025".
    size_score = 25.0 * clamp(1.0 - abs(volume_ratio - 1.138) / 1.15)
    raw_component = 0.35 * clamp(raw_score, 0.0, 100.0)
    contact_component = 18.0 * clamp(both) + 10.0 * clamp(longest)
    long_extent_penalty = max(0.0, max_extent - 0.50) * 90.0
    aspect_penalty = max(0.0, aspect - 1.8) * 12.0

    # Score is object-agnostic and excludes pipeline readiness.
    score = size_score + raw_component + contact_component
    score -= long_extent_penalty + aspect_penalty

    tier = route_bucket(target, row, d2)
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
        volume_ratio, max_extent, _aspect = geometry_features(row)
        in_medium_box_range = 0.75 <= volume_ratio <= 3.50 and max_extent <= 0.70
        in_missing_raw_review_range = 2.00 <= volume_ratio <= 2.60 and 0.55 <= max_extent <= 0.75
        if (
            obj.startswith("box")
            and d2
            and d2.get("stage1_decision") == "raw_contact_pass"
            and in_medium_box_range
            and not str(row.get("action", "")).startswith("pass")
        ):
            include = True
        if (
            obj.startswith("box")
            and row.get("stage0_v2_decision") == "stage0_boundary_review_to_stage1"
            and in_missing_raw_review_range
        ):
            include = True
        if not include:
            continue

        target = planned_target(row)
        source_task, source_ready = source_scene_task(obj, str(row["person"]), task_root)
        score, tier, risk, note = score_row(row, d2)
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
        "tier1_worklike_priority": 1,
        "tier1_worklike_extra": 2,
        "tier2_target_posture_gate_review": 3,
        "tier3_missing_raw_contact_long_edge_review": 4,
        "tier4_large_reach_dynamics_holdout": 5,
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
        "- Score is geometry/raw-contact only; source-scene readiness and object key are not numeric score terms.",
        "- Rows are sorted by execution tier first, then score; `rank` is therefore queue rank, not pure score rank.",
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
            "- `tier1_worklike_priority` is the only new batch to run immediately.",
            "- `tier2_target_posture_gate_review` is held for target/posture-gated review before CEM.",
            "- `tier4_large_reach_dynamics_holdout` remains in the bank for traceability but is not a first-batch data source after large-box E092/E094 failures.",
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

    box004 = [row for row in rows if row["target_task"] in FIRST_BATCH_TARGETS]
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
    box004_cases = pipeline_rows(rows, FIRST_BATCH_TARGETS)
    for path in (
        out_root / "cases_e095_box004_priority_pipeline.tsv",
        args.v2_root / "inputs/cases_e095_box004_priority_pipeline.tsv",
    ):
        write_tsv(path, box004_cases, case_fields)

    posture_review_targets = {row["target_task"] for row in rows if row["tier"] == "tier2_target_posture_gate_review"}
    posture_review_rows = pipeline_rows(rows, posture_review_targets, 0)
    write_tsv(out_root / "cases_e095_target_posture_gate_review_disabled.tsv", posture_review_rows, case_fields)

    write_summary(out_root / "summary.md", rows, box004)
    write_summary(v2_out / "summary.md", rows, box004)
    write_json(
        out_root / "summary.json",
        {
            "rows": len(rows),
            "tier_counts": dict(Counter(row["tier"] for row in rows)),
            "first_batch_targets": [row["target_task"] for row in box004],
            "box004_case_file": str(args.v2_root / "inputs/cases_e095_box004_priority_pipeline.tsv"),
        },
    )
    print(f"Wrote {len(rows)} candidates")
    print(f"Wrote {len(box004_cases)} first-batch worklike cases")
    print(args.v2_root / "inputs/cases_e095_box004_priority_pipeline.tsv")


if __name__ == "__main__":
    main()
