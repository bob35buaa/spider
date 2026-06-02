#!/usr/bin/env python3
"""Build a unified case bank for Spider vs OmniRetarget fair evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import REPO, first_nonempty, infer_experiment, read_csv, repo_rel, write_json, write_tsv


CASE_BANK_FIELDS = [
    "bank_row_id",
    "case_id",
    "case_key",
    "object_key",
    "object_name",
    "date",
    "seq",
    "person",
    "person_idx",
    "method",
    "method_family",
    "experiment",
    "source_kind",
    "source_path",
    "variant",
    "trajectory_npz",
    "scene_xml",
    "video_path",
    "metrics_ref",
    "legobj_timeseries_csv",
    "cem_status",
    "rl_status",
    "downstream_decision",
    "target_variant_id",
    "retarget_variant_id",
    "fair_comparison_role",
    "self_ref_tracking_caveat",
    "notes",
]


def short_case_key(row: dict[str, Any]) -> str:
    case = first_nonempty(row, ["case", "case_id", "derived_task", "source_task"])
    if case:
        return case
    object_key = first_nonempty(row, ["object_key", "object", "object_name"])
    person = first_nonempty(row, ["person", "person_idx"])
    return f"{object_key}_{person}" if object_key or person else ""


def object_from_case(case: str) -> str:
    for token in case.replace("-", "_").split("_"):
        if token.startswith(("box", "bucket", "desk", "chair", "board", "stick")):
            return token
    return ""


def localize_artifact(raw_path: str, summary_path: Path) -> str:
    """Prefer artifacts colocated with a summary when old summaries retain /tmp paths."""
    if not raw_path:
        return ""
    path = Path(raw_path)
    colocated = summary_path.parent / path.name
    if path.is_absolute() and str(path).startswith("/tmp/") and colocated.exists():
        return repo_rel(colocated)
    if path.is_absolute() and path.exists():
        return repo_rel(path)
    if not path.is_absolute() and (REPO / path).exists():
        return repo_rel(path)
    if colocated.exists():
        return repo_rel(colocated)
    return repo_rel(raw_path)


def rows_from_e026(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in read_csv(path):
        method = row.get("method", "")
        case = row.get("case", "")
        out.append(
            {
                "case_id": case,
                "case_key": case,
                "object_key": object_from_case(case),
                "method": method,
                "method_family": "omniretarget" if "omni" in method else "spider",
                "experiment": "E026",
                "source_kind": "e026_method_metrics",
                "source_path": repo_rel(path),
                "variant": row.get("variant", ""),
                "metrics_ref": repo_rel(row.get("source_csv", "")),
                "cem_status": "historical_metric",
                "rl_status": "not_run",
                "fair_comparison_role": "diagnostic_only"
                if method == "omniretarget_kinematic"
                else "historical_spider_metric",
                "self_ref_tracking_caveat": "yes"
                if method == "omniretarget_kinematic"
                else "method_ref_possible",
                "notes": "E026 history; object/body tracking can be self-ref or method-ref.",
            }
        )
    return out


def rows_from_cem_summary(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    exp = infer_experiment(path)
    rows = read_csv(path)
    for row in rows:
        variant = row.get("variant", "")
        leg_name = f"legobj_timeseries_{variant}.csv"
        leg_path = path.parent / leg_name
        out.append(
            {
                "case_id": first_nonempty(row, ["case_id", "derived_task", "source_task", "variant"]),
                "case_key": short_case_key(row),
                "object_key": object_from_case(first_nonempty(row, ["case_id", "derived_task", "source_task", "variant"]))
                or row.get("object", ""),
                "object_name": row.get("object", ""),
                "method": f"spider_cem_{exp.lower()}" if exp else "spider_cem",
                "method_family": "spider",
                "experiment": exp,
                "source_kind": "spider_cem_summary",
                "source_path": repo_rel(path),
                "variant": variant,
                "trajectory_npz": localize_artifact(first_nonempty(row, ["npz_path", "root_npz_path"]), path),
                "scene_xml": repo_rel(row.get("scene_xml", "")),
                "video_path": localize_artifact(first_nonempty(row, ["video_path", "mp4_path"]), path),
                "metrics_ref": repo_rel(path),
                "legobj_timeseries_csv": repo_rel(leg_path) if leg_path.is_file() else "",
                "cem_status": first_nonempty(row, ["cem_status", "work_status", "work_status_lowerbody_strict"]),
                "rl_status": "not_run",
                "downstream_decision": "",
                "target_variant_id": row.get("target_route", row.get("route", "")),
                "fair_comparison_role": "spider_cem_candidate",
                "self_ref_tracking_caveat": "method_ref_object_tracking",
                "notes": "Spider CEM rollout; proxy metrics are fair only when based on geometry/SDF, not method-ref object tracking.",
            }
        )
    return out


def rows_from_existing_cases(path: Path, *, status_filter: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in read_csv(path, delimiter="\t"):
        cem_status = row.get("cem_status", "")
        rl_status = row.get("rl_status", "")
        if status_filter == "cem_pass" and cem_status != "pass":
            continue
        if status_filter == "cem_or_rl_pass" and cem_status != "pass" and rl_status != "pass":
            continue
        out.append(
            {
                "case_id": row.get("case_id", ""),
                "case_key": row.get("case_id", ""),
                "object_key": row.get("object_key", ""),
                "object_name": row.get("object_name", ""),
                "date": row.get("date", ""),
                "seq": row.get("seq", ""),
                "person": row.get("person", ""),
                "person_idx": row.get("person_idx", ""),
                "method": "spider_cem_registry",
                "method_family": "spider",
                "experiment": infer_experiment(row.get("downstream_evidence_root", "") or row.get("cem_metrics_ref", "")),
                "source_kind": "data_construction_v3_existing_cases",
                "source_path": repo_rel(path),
                "variant": row.get("cem_run_id", ""),
                "trajectory_npz": repo_rel(row.get("cem_result_npz", "")),
                "scene_xml": "",
                "video_path": repo_rel(row.get("cem_video", "")),
                "metrics_ref": repo_rel(row.get("cem_metrics_ref", "")),
                "cem_status": cem_status,
                "rl_status": rl_status,
                "downstream_decision": row.get("downstream_decision", ""),
                "target_variant_id": row.get("target_variant_id", ""),
                "retarget_variant_id": row.get("retarget_variant_id", ""),
                "fair_comparison_role": "verified_spider_cem_or_rl",
                "self_ref_tracking_caveat": "method_ref_object_tracking",
                "notes": row.get("notes", ""),
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--existing-cases", type=Path)
    parser.add_argument(
        "--existing-case-filter",
        choices=["cem_pass", "cem_or_rl_pass"],
        default="cem_pass",
        help="Filter for rows imported from existing_cases.tsv. Default uses only cem_status=pass.",
    )
    parser.add_argument("--e026-metrics", type=Path)
    parser.add_argument("--cem-summary", type=Path, action="append", default=[])
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    if args.e026_metrics:
        rows.extend(rows_from_e026(args.e026_metrics))
    if args.existing_cases:
        rows.extend(rows_from_existing_cases(args.existing_cases, status_filter=args.existing_case_filter))
    for path in args.cem_summary:
        rows.extend(rows_from_cem_summary(path))

    for idx, row in enumerate(rows, start=1):
        row["bank_row_id"] = f"bank_{idx:05d}"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(args.out_dir / "case_bank.tsv", rows, CASE_BANK_FIELDS)
    write_json(
        args.out_dir / "case_bank_summary.json",
        {
            "num_rows": len(rows),
            "num_e026_rows": sum(r["source_kind"] == "e026_method_metrics" for r in rows),
            "num_existing_case_rows": sum(r["source_kind"] == "data_construction_v3_existing_cases" for r in rows),
            "num_cem_summary_rows": sum(r["source_kind"] == "spider_cem_summary" for r in rows),
            "existing_case_filter": args.existing_case_filter,
        },
    )
    print(f"[build_case_bank] wrote {len(rows)} rows to {args.out_dir / 'case_bank.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
