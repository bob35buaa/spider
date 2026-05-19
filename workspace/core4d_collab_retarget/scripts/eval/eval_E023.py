#!/usr/bin/env python3
"""E023 evaluation for retarget-kinematic geometry repair."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
SCRIPT_EVAL = REPO / "workspace/core4d_collab_retarget/scripts/eval"
if str(SCRIPT_EVAL) not in sys.path:
    sys.path.insert(0, str(SCRIPT_EVAL))

import eval_E018b as e018b  # noqa: E402


RESULTS = REPO / "workspace/core4d_collab_retarget/results/E023"
MANIFEST = RESULTS / "manifest.tsv"
E018B_MANIFEST = REPO / "workspace/core4d_collab_retarget/results/E018b/manifest.tsv"


def _read_manifest(path: Path = MANIFEST) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return {row["variant"]: row for row in csv.DictReader(f, delimiter="\t")}


def _as_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _write_summary(summary: dict[str, Any]) -> None:
    variant = str(summary["variant"])
    (RESULTS / f"eval_summary_{variant}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (RESULTS / f"eval_summary_{variant}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def _support_proxy_unchanged(row: dict[str, str], source: dict[str, str]) -> bool:
    keys = [
        "scene_name",
        "support_proxy_point_local_x",
        "support_proxy_point_local_y",
        "support_proxy_point_local_z",
        "support_proxy_gravity_scale",
        "weld_solref_timeconst",
        "weld_solimp_1",
        "weld_solimp_2",
        "weld_solimp_width",
        "anchor_face",
        "canonical_z_frac",
    ]
    return all(str(row.get(key, "")) == str(source.get(key, "")) for key in keys)


def evaluate_variant(
    variant: str,
    manifest: dict[str, dict[str, str]],
    e018b_manifest: dict[str, dict[str, str]],
) -> dict[str, Any]:
    e018b.RESULTS = RESULTS
    e018b.MANIFEST = MANIFEST
    variants = e018b.read_manifest()
    summary = e018b.evaluate_variant(variant, variants)
    row = manifest[variant]
    source = e018b_manifest[row["E023_source_variant"]]

    full_ref = _as_float(summary.get("full_ref_leg_box_interference_frames_pct"), 100.0)
    case_ref = _as_float(summary.get("case_window_ref_leg_box_interference_frames_pct"), 100.0)
    contact = _as_float(summary.get("paper_omniretarget_contact_preservation_5cm_pct"), 0.0)
    deep = _as_float(summary.get("paper_omniretarget_robot_object_deep_penetration_duration_pct"), 100.0)
    max_pen = _as_float(summary.get("paper_omniretarget_robot_object_max_penetration_cm"), 999.0)

    summary["E023_source_variant"] = row["E023_source_variant"]
    summary["E023_source_derived_task"] = row["E023_source_derived_task"]
    summary["E023_patch_mode"] = row["E023_patch_mode"]
    summary["E023_disabled_leg_object_pairs"] = row["E023_disabled_leg_object_pairs"]
    summary["E023_shrunk_leg_geoms"] = row["E023_shrunk_leg_geoms"]
    summary["E023_lowerbody_radius"] = _as_float(row.get("E023_lowerbody_radius"))
    summary["E023_foot_radius"] = _as_float(row.get("E023_foot_radius"))
    summary["E023_support_proxy_unchanged"] = _support_proxy_unchanged(row, source)
    summary["E023_ref_geometry_repair_pass"] = bool(full_ref < 15.0 and case_ref < 15.0)
    summary["E023_contact_goal_pass"] = bool(contact >= 70.0)
    summary["E023_object_no_regression_pass"] = bool(
        _as_float(summary.get("paper_object_Epos_case_m"), 999.0) < 0.10
        and _as_float(summary.get("paper_object_Erot_case_deg"), 999.0) < 25.0
        and bool(summary.get("paper_transport_success", False))
    )
    summary["E023_artifact_guard_pass"] = bool(
        not bool(summary.get("E018b_robot_fall_detected", True))
        and deep <= 20.0
        and max_pen <= 5.0
    )
    summary["E023_success"] = bool(
        summary["E023_ref_geometry_repair_pass"]
        and summary["E023_contact_goal_pass"]
        and summary["E023_object_no_regression_pass"]
        and summary["E023_artifact_guard_pass"]
        and summary["E023_support_proxy_unchanged"]
    )
    _write_summary(summary)
    return summary


def _normalize_args(args: list[str], manifest: dict[str, dict[str, str]]) -> list[str]:
    if not args or args == ["--all"]:
        return list(manifest.keys())
    out: list[str] = []
    for arg in args:
        if arg == "--all":
            out.extend(manifest.keys())
        else:
            out.append(arg)
    return out


def main() -> None:
    manifest = _read_manifest()
    source_manifest = _read_manifest(E018B_MANIFEST)
    selected = _normalize_args(sys.argv[1:], manifest)
    summaries: list[dict[str, Any]] = []
    for variant in selected:
        if variant not in manifest:
            print(f"[SKIP] unknown variant {variant}")
            continue
        try:
            summaries.append(evaluate_variant(variant, manifest, source_manifest))
        except FileNotFoundError as exc:
            print(f"[SKIP] {exc}")
            continue
    if not summaries:
        raise SystemExit("No E023 variant results found.")

    keys = sorted({key for row in summaries for key in row.keys()})
    comparison = RESULTS / "comparison.csv"
    with comparison.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summaries)

    aggregate = {
        "num_results": len(summaries),
        "num_E023_success": sum(bool(r["E023_success"]) for r in summaries),
        "num_ref_geometry_repair_pass": sum(bool(r["E023_ref_geometry_repair_pass"]) for r in summaries),
        "num_contact_goal_pass": sum(bool(r["E023_contact_goal_pass"]) for r in summaries),
        "num_object_no_regression_pass": sum(bool(r["E023_object_no_regression_pass"]) for r in summaries),
        "num_artifact_guard_pass": sum(bool(r["E023_artifact_guard_pass"]) for r in summaries),
        "best_ref_leg_interference_pct": min(
            _as_float(r.get("full_ref_leg_box_interference_frames_pct"), 100.0) for r in summaries
        ),
        "best_variant_by_ref_leg_interference": min(
            summaries,
            key=lambda r: _as_float(r.get("full_ref_leg_box_interference_frames_pct"), 100.0),
        )["variant"],
        "variants": [str(r["variant"]) for r in summaries],
    }
    (RESULTS / "aggregate_summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Wrote {comparison}")
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
