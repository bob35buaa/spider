#!/usr/bin/env python3
"""Feature-based data_construction_v2 candidate refresh.

This miner turns the E095/E096 lessons into a reusable queue:
raw contact -> reach proxy -> inside/support status -> CEM posture status.
Already verified cases are kept in the audit table but excluded from the new
execution queue.
"""

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
DEFAULT_OUT_ROOT = Path("workspace/core4d/results/E097/feature_candidate_mining")

PERSON_SHORT = {"person1": "p1", "person2": "p2"}

# "Verified" here means "do not rediscover as a new candidate in this round".
# It may be a positive, a CEM fail, or an upstream preprocess/gate reject.
KNOWN_OUTCOMES: dict[str, dict[str, str]] = {
    "e091_box004_20231003_2_083_p2": {
        "status": "verified_positive",
        "stage": "full_cem_work",
        "note": "E092/E094 known WORK control; E091 D005b PASS.",
    },
    "e091_box004_20231003_2_083_p1": {
        "status": "verified_positive",
        "stage": "full_cem_work",
        "note": "E096/E096b full CEM WORK.",
    },
    "e091_box004_20231003_2_082_p1": {
        "status": "verified_positive",
        "stage": "full_cem_work",
        "note": "E096/E096b full CEM WORK.",
    },
    "e091_box004_20231003_2_082_p2": {
        "status": "verified_reject",
        "stage": "omniretarget_infeasible",
        "note": "E095/E096 no-fingertip and fingertip retry both CVXPY infeasible.",
    },
    "e091_box026_20231018_039_p2": {
        "status": "verified_reject",
        "stage": "d005b_reject_and_full_cem_fail",
        "note": "E091 support reject; E092/E094 full CEM posture fail.",
    },
    "e091_box026_20231018_040_p2": {
        "status": "verified_reject",
        "stage": "omniretarget_infeasible",
        "note": "E091 OmniRetarget CVXPY infeasible.",
    },
    "e091_box026_20231020_135_p2": {
        "status": "verified_reject",
        "stage": "d005b_reject_and_full_cem_fail",
        "note": "E091 right-inside reject; E092/E094 full CEM posture fail.",
    },
    "e091_box021_20231018_029_p2": {
        "status": "verified_reject",
        "stage": "full_cem_fail",
        "note": "D003/E082-E088 Box021 representative full CEM fail.",
    },
    "e091_box021_20231011_035_p2": {
        "status": "verified_reject",
        "stage": "full_cem_fail",
        "note": "D003/E082-E083 Box021 representative full CEM fail.",
    },
    "e091_box021_20231020_019_p1": {
        "status": "verified_reject",
        "stage": "full_cem_fail",
        "note": "D003/E082-E083 Box021 representative full CEM fail.",
    },
    "e091_box021_20231018_031_p2": {
        "status": "verified_review",
        "stage": "topface_smoke_review",
        "note": "E089/E090 topface/smoke route already used this case.",
    },
    "e091_box021_20231020_020_p1": {
        "status": "verified_review",
        "stage": "topface_smoke_review",
        "note": "E089/E090 topface/smoke route already used this case.",
    },
    "e091_box021_20231011_034_p1": {
        "status": "verified_review",
        "stage": "preprocess_or_smoke_review",
        "note": "Older data_construction/D003 route already used this case.",
    },
    "e091_box021_20231011_034_p2": {
        "status": "verified_reject",
        "stage": "omniretarget_infeasible",
        "note": "Older data_construction route reported CVXPY infeasible.",
    },
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
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


def target_task(row: dict[str, Any]) -> str:
    return f"e091_{row['object_key']}_{row['date']}_{row['seq']}_{PERSON_SHORT.get(row['person'], row['person'])}"


def d002_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["sequence"]), str(row["person"])


def source_scene_task(object_key: str, person: str, task_root: Path) -> tuple[str, bool]:
    task = f"{object_key}_{person}"
    if (task_root / task / "scene.xml").is_file():
        return task, True
    fallback = f"{object_key}_person1"
    if (task_root / fallback / "scene.xml").is_file():
        return fallback, True
    return task, False


def raw_proxy_path(row: dict[str, Any], old_root: Path) -> str:
    slug = f"{row['date']}_{row['seq']}_{row['object_key']}"
    path = old_root / "results/d002_stage1_raw_contact_v2/per_sequence" / slug / "raw_contact_proxy.npz"
    return str(path) if path.is_file() else ""


def geometry(row: dict[str, Any]) -> dict[str, float]:
    extents = [
        as_float(row.get("extent_x_m")),
        as_float(row.get("extent_y_m")),
        as_float(row.get("extent_z_m")),
    ]
    max_extent = max(extents)
    min_extent = min(extents)
    return {
        "volume_ratio": as_float(row.get("size_vs_box023_volume_ratio"), 999.0),
        "max_extent": max_extent,
        "min_extent": min_extent,
        "aspect": max_extent / max(min_extent, 1e-6),
    }


def raw_features(d2: dict[str, Any] | None) -> dict[str, float | str]:
    if d2 is None:
        return {
            "stage1_decision": "not_run",
            "raw_score": 0.0,
            "both": 0.0,
            "left": 0.0,
            "right": 0.0,
            "balanced": 0.0,
            "longest": 0.0,
            "partner_any": 0.0,
        }
    left = as_float(d2.get("target_left_active_frac_3cm"))
    right = as_float(d2.get("target_right_active_frac_3cm"))
    return {
        "stage1_decision": str(d2.get("stage1_decision", "")),
        "raw_score": as_float(d2.get("stage1_raw_contact_score")),
        "both": as_float(d2.get("target_both_active_frac_3cm")),
        "left": left,
        "right": right,
        "balanced": min(left, right),
        "longest": as_float(d2.get("target_both_longest_run_active_frac_3cm")),
        "partner_any": as_float(d2.get("partner_any_active_frac_3cm")),
    }


def contact_score(feat: dict[str, float | str]) -> float:
    return (
        0.38 * clamp(float(feat["raw_score"]), 0.0, 100.0)
        + 24.0 * clamp(float(feat["both"]))
        + 14.0 * clamp(float(feat["balanced"]))
        + 10.0 * clamp(float(feat["longest"]))
        + 6.0 * clamp(float(feat["partner_any"]))
    )


def reach_score(geom: dict[str, float]) -> tuple[float, str]:
    volume_ratio = geom["volume_ratio"]
    max_extent = geom["max_extent"]
    aspect = geom["aspect"]

    score = 35.0
    if volume_ratio <= 1.25 and max_extent <= 0.46:
        label = "box004_like_low_reach_risk"
    elif volume_ratio <= 1.85 and max_extent <= 0.52:
        label = "medium_box_target_posture_gate"
        score -= 8.0
    elif volume_ratio <= 2.60 and max_extent <= 0.70:
        label = "long_edge_raw_contact_preflight"
        score -= 22.0
    else:
        label = "large_reach_holdout"
        score -= 35.0

    score -= max(0.0, max_extent - 0.50) * 120.0
    score -= max(0.0, aspect - 1.80) * 10.0
    return round(score, 3), label


def route_for_row(
    target: str,
    row: dict[str, Any],
    d2: dict[str, Any] | None,
    feat: dict[str, float | str],
    geom: dict[str, float],
    reach_label: str,
) -> tuple[str, str, int]:
    outcome = KNOWN_OUTCOMES.get(target)
    if outcome is not None:
        return "excluded_verified", outcome["note"], 0

    action = str(row.get("action", ""))
    if action.startswith("pass") or action == "rot":
        return "hold_action_family", "non-move action family is not comparable to box004 carry pattern", 0

    if d2 is None:
        if reach_label == "long_edge_raw_contact_preflight":
            return "raw_contact_preflight_disabled", "raw contact not run; run D002/raw-contact visual first", 0
        return "hold_missing_raw_contact", "missing D002 raw-contact evidence", 0

    stage1 = str(feat["stage1_decision"])
    if stage1 != "raw_contact_pass":
        return "hold_raw_contact_failed", f"D002 stage1_decision={stage1}", 0

    if reach_label == "box004_like_low_reach_risk":
        return "candidate_full_pipeline", "low reach-risk raw-contact candidate; run preprocess -> D005b -> full CEM", 1
    if reach_label == "medium_box_target_posture_gate":
        return "candidate_target_posture_gate", "raw contact strong; run preprocess plus inside/support target gate before CEM", 1
    if reach_label == "long_edge_raw_contact_preflight":
        return "raw_contact_preflight_disabled", "long edge requires raw-contact/contact-target preflight before preprocess", 0
    if geom["volume_ratio"] >= 2.8 or geom["max_extent"] >= 0.58:
        return "large_reach_holdout", "large reach risk after Box026 failures; needs separate repair route", 0
    return "review_disabled", "feature route did not meet automatic next-batch criteria", 0


def build_rows(old_root: Path, task_root: Path) -> list[dict[str, Any]]:
    d001 = read_json(old_root / "results/d001_stage0_inventory_v2/candidate_inventory_v2.json")
    d002 = read_json(old_root / "results/d002_stage1_raw_contact_v2/raw_contact_summary_v2.json")
    d2_idx = {d002_key(row): row for row in d002}

    rows: list[dict[str, Any]] = []
    for row0 in d001:
        object_key = str(row0.get("object_key", ""))
        if not object_key.startswith("box"):
            continue

        geom = geometry(row0)
        d2 = d2_idx.get(d002_key(row0))
        target = target_task(row0)

        include = False
        if d2 and str(d2.get("stage1_decision")) in {"raw_contact_pass", "raw_contact_review", "raw_contact_fail"}:
            include = geom["volume_ratio"] <= 3.6 and geom["max_extent"] <= 0.72
        elif row0.get("stage0_v2_decision") == "stage0_boundary_review_to_stage1":
            include = 1.8 <= geom["volume_ratio"] <= 2.7 and geom["max_extent"] <= 0.72
        if not include:
            continue

        feat = raw_features(d2)
        reach_component, reach_label = reach_score(geom)
        route, note, enabled = route_for_row(target, row0, d2, feat, geom, reach_label)
        src_task, src_ready = source_scene_task(object_key, str(row0["person"]), task_root)
        outcome = KNOWN_OUTCOMES.get(target, {})
        quality_score = contact_score(feat) + reach_component
        if route == "excluded_verified":
            quality_score = -999.0
        elif route in {"large_reach_holdout", "hold_raw_contact_failed", "hold_missing_raw_contact"}:
            quality_score -= 45.0
        elif route == "raw_contact_preflight_disabled":
            quality_score -= 20.0

        rows.append(
            {
                "rank": 0,
                "route_enabled_default": enabled,
                "target_task": target,
                "route": route,
                "quality_score": round(quality_score, 3),
                "known_status": outcome.get("status", "unverified"),
                "known_stage": outcome.get("stage", ""),
                "object_key": object_key,
                "object_name": row0.get("object_name", ""),
                "sequence": row0["sequence"],
                "date": row0["date"],
                "seq": row0["seq"],
                "person": row0["person"],
                "action": row0.get("action", ""),
                "reach_label": reach_label,
                "inside_support_status": "pending_gate" if enabled else outcome.get("stage", "not_evaluated"),
                "cem_posture_status": "pending_full_cem" if enabled else outcome.get("stage", "not_evaluated"),
                "stage0_v2_decision": row0.get("stage0_v2_decision", ""),
                "stage1_decision": feat["stage1_decision"],
                "raw_score": round(float(feat["raw_score"]), 3),
                "target_both_active_frac_3cm": feat["both"],
                "target_left_active_frac_3cm": feat["left"],
                "target_right_active_frac_3cm": feat["right"],
                "target_both_longest_run_active_frac_3cm": feat["longest"],
                "partner_any_active_frac_3cm": feat["partner_any"],
                "size_vs_box023_volume_ratio": row0.get("size_vs_box023_volume_ratio", ""),
                "extent_x_m": row0.get("extent_x_m", ""),
                "extent_y_m": row0.get("extent_y_m", ""),
                "extent_z_m": row0.get("extent_z_m", ""),
                "extent_max_m": round(geom["max_extent"], 6),
                "aspect_ratio": round(geom["aspect"], 6),
                "object_model_rel": row0.get("object_mesh_rel", ""),
                "source_scene_task": src_task,
                "source_scene_exists": src_ready,
                "raw_contact_proxy_path": raw_proxy_path(row0, old_root) if d2 else "",
                "feature_note": note,
            }
        )

    route_order = {
        "candidate_full_pipeline": 0,
        "candidate_target_posture_gate": 1,
        "raw_contact_preflight_disabled": 2,
        "review_disabled": 3,
        "large_reach_holdout": 4,
        "hold_raw_contact_failed": 5,
        "hold_missing_raw_contact": 6,
        "hold_action_family": 7,
        "excluded_verified": 8,
    }
    rows.sort(key=lambda r: (route_order.get(str(r["route"]), 99), -float(r["quality_score"]), str(r["target_task"])))
    for i, row in enumerate(rows, start=1):
        row["rank"] = i
    return rows


def pipeline_rows(rows: list[dict[str, Any]], max_enabled: int) -> list[dict[str, Any]]:
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
        "e097_route",
        "e097_score",
        "e097_note",
    ]
    out: list[dict[str, Any]] = []
    enabled_count = 0
    for row in rows:
        if row["route"] not in {"candidate_full_pipeline", "candidate_target_posture_gate"}:
            continue
        enabled = 1 if enabled_count < max_enabled else 0
        enabled_count += int(enabled)
        out.append(
            {
                "# enabled": enabled,
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
                "e097_route": row["route"],
                "e097_score": row["quality_score"],
                "e097_note": row["feature_note"],
            }
        )
    return out


def write_summary(path: Path, rows: list[dict[str, Any]], pipeline: list[dict[str, Any]]) -> None:
    route_counts = Counter(str(row["route"]) for row in rows)
    enabled = [row for row in pipeline if int(row["# enabled"]) == 1]
    excluded = [row for row in rows if row["route"] == "excluded_verified"]
    candidates = [row for row in rows if row["route"] in {"candidate_full_pipeline", "candidate_target_posture_gate"}]
    holdouts = [row for row in rows if row["route"] in {"large_reach_holdout", "raw_contact_preflight_disabled"}]

    lines = [
        "# E097 Feature-based Candidate Mining",
        "",
        "## Summary",
        "",
        f"- Candidate/audit rows: `{len(rows)}`",
        f"- Already verified excluded rows: `{len(excluded)}`",
        f"- Unverified candidate rows: `{len(candidates)}`",
        f"- Enabled next-batch rows: `{len(enabled)}`",
        "- Score uses raw contact and geometry/reach proxy only; source scene readiness and object key are metadata, not score terms.",
        "- Pipeline `# enabled=1` means run preprocess + inside/support target gate first, not direct RL.",
        "",
        "Route counts:",
        "",
        "| route | count |",
        "|---|---:|",
    ]
    for route, count in route_counts.most_common():
        lines.append(f"| `{route}` | {count} |")

    lines.extend(
        [
            "",
            "## Enabled Next Batch",
            "",
            "| target | route | score | raw | both | L/R | longest | reach | note |",
            "|---|---|---:|---:|---:|---|---:|---|---|",
        ]
    )
    row_by_target = {str(row["target_task"]): row for row in rows}
    for pipe in pipeline:
        if int(pipe["# enabled"]) != 1:
            continue
        row = row_by_target[str(pipe["target_task"])]
        lines.append(
            f"| `{row['target_task']}` | `{row['route']}` | {row['quality_score']} | "
            f"{row['raw_score']} | {row['target_both_active_frac_3cm']} | "
            f"{row['target_left_active_frac_3cm']}/{row['target_right_active_frac_3cm']} | "
            f"{row['target_both_longest_run_active_frac_3cm']} | `{row['reach_label']}` | {row['feature_note']} |"
        )

    lines.extend(
        [
            "",
            "## Top Unverified Candidates",
            "",
            "| rank | target | route | score | raw | reach | scene | note |",
            "|---:|---|---|---:|---:|---|---|---|",
        ]
    )
    for row in candidates[:12]:
        lines.append(
            f"| {row['rank']} | `{row['target_task']}` | `{row['route']}` | {row['quality_score']} | "
            f"{row['raw_score']} | `{row['reach_label']}` | `{row['source_scene_task']}` | {row['feature_note']} |"
        )

    lines.extend(
        [
            "",
            "## Holdouts / Preflight",
            "",
            "| target | route | raw | reach | note |",
            "|---|---|---:|---|---|",
        ]
    )
    for row in holdouts[:12]:
        lines.append(
            f"| `{row['target_task']}` | `{row['route']}` | {row['raw_score']} | `{row['reach_label']}` | {row['feature_note']} |"
        )

    lines.extend(
        [
            "",
            "## Excluded Verified",
            "",
            "| target | status | stage | note |",
            "|---|---|---|---|",
        ]
    )
    for row in excluded:
        lines.append(
            f"| `{row['target_task']}` | `{row['known_status']}` | `{row['known_stage']}` | {row['feature_note']} |"
        )

    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- New likely-work candidates are not yet RL-ready. They should run Stage2b/OmniRetarget, then D005b/E096-style inside/support target gate, then full CEM posture gate.",
            "- The current best new batch is box021, but only as `candidate_target_posture_gate`; this reflects both its strong raw contact and its known posture/target risk.",
            "- Box026 remains a large-reach holdout despite raw contact; Box022 needs raw-contact preflight before preprocess.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-root", type=Path, default=DEFAULT_OLD_ROOT)
    parser.add_argument("--v2-root", type=Path, default=DEFAULT_V2_ROOT)
    parser.add_argument("--spider-task-root", type=Path, default=DEFAULT_SPIDER_TASK_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--max-enabled", type=int, default=6)
    args = parser.parse_args()

    rows = build_rows(args.old_root, args.spider_task_root)
    fields = list(rows[0].keys()) if rows else []
    pipeline = pipeline_rows(rows, args.max_enabled)
    pipeline_fields = list(pipeline[0].keys()) if pipeline else [
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
        "e097_route",
        "e097_score",
        "e097_note",
    ]

    out_roots = [
        args.out_root,
        args.v2_root / "results/e097_feature_candidates",
    ]
    for root in out_roots:
        write_tsv(root / "feature_candidate_bank.tsv", rows, fields)
        write_json(root / "feature_candidate_bank.json", rows)
        excluded = [row for row in rows if row["route"] == "excluded_verified"]
        write_tsv(root / "excluded_verified_cases.tsv", excluded, fields)
        write_json(root / "excluded_verified_cases.json", excluded)
        write_tsv(root / "cases_e097_feature_candidate_pipeline.tsv", pipeline, pipeline_fields)
        write_summary(root / "summary.md", rows, pipeline)
        write_json(
            root / "summary.json",
            {
                "rows": len(rows),
                "route_counts": dict(Counter(str(row["route"]) for row in rows)),
                "enabled_targets": [
                    row["target_task"] for row in pipeline if int(row.get("# enabled", 0)) == 1
                ],
                "pipeline_file": str(args.v2_root / "inputs/cases_e097_feature_candidate_pipeline.tsv"),
            },
        )

    write_tsv(args.v2_root / "inputs/cases_e097_feature_candidate_pipeline.tsv", pipeline, pipeline_fields)
    print(f"Wrote {len(rows)} feature candidate rows")
    print(f"Wrote {sum(int(row['# enabled']) for row in pipeline)} enabled next-batch rows")
    print(args.v2_root / "inputs/cases_e097_feature_candidate_pipeline.tsv")


if __name__ == "__main__":
    main()
