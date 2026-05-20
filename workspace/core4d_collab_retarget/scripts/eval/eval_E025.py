#!/usr/bin/env python3
"""E025 evaluation for robot-side contact closure + collision penalty."""

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


RESULTS = REPO / "workspace/core4d_collab_retarget/results/E025"
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
        "derived_task",
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
    source = e018b_manifest[row["E025_source_variant"]]

    contact = _as_float(summary.get("paper_omniretarget_contact_preservation_5cm_pct"), 0.0)
    deep = _as_float(summary.get("paper_omniretarget_robot_object_deep_penetration_duration_pct"), 100.0)
    max_pen = _as_float(summary.get("paper_omniretarget_robot_object_max_penetration_cm"), 999.0)
    leg_deep = _as_float(summary.get("paper_omniretarget_leg_object_deep_penetration_duration_pct"), 100.0)
    sim_leg = _as_float(summary.get("case_window_sim_leg_box_interference_frames_pct"), 100.0)

    summary["E025_source_variant"] = row["E025_source_variant"]
    summary["E025_case_slug"] = row["E025_case_slug"]
    summary["E025_role"] = row["E025_role"]
    summary["E025_wave"] = row["E025_wave"]
    summary["E025_hold_contact_rew_scale"] = _as_float(row["E025_hold_contact_rew_scale"])
    summary["E025_contact_hdmi_gain"] = _as_float(row["E025_contact_hdmi_gain"])
    summary["E025_contact_hdmi_sigma"] = _as_float(row["E025_contact_hdmi_sigma"])
    summary["E025_contact_hdmi_ori_weight"] = _as_float(row["E025_contact_hdmi_ori_weight"])
    summary["E025_contact_hdmi_ori_mode"] = row["E025_contact_hdmi_ori_mode"]
    summary["E025_robot_object_penalty_scale"] = _as_float(row["E025_robot_object_penalty_scale"])
    summary["E025_robot_object_penalty_margin_m"] = _as_float(row["E025_robot_object_penalty_margin_m"])
    summary["E025_robot_object_penalty_deep_threshold_m"] = _as_float(row["E025_robot_object_penalty_deep_threshold_m"])
    summary["E025_leg_object_penalty_scale"] = _as_float(row["E025_leg_object_penalty_scale"])
    summary["E025_leg_object_penalty_margin_m"] = _as_float(row["E025_leg_object_penalty_margin_m"])
    summary["E025_leg_object_penalty_geom_names"] = row["E025_leg_object_penalty_geom_names"]
    summary["E025_notes"] = row["E025_notes"]
    summary["E025_support_proxy_unchanged"] = _support_proxy_unchanged(row, source)
    summary["E025_contact_closure_pass"] = bool(contact >= 70.0)
    summary["E025_penetration_guard_pass"] = bool(deep < 15.0 and max_pen <= 5.0)
    summary["E025_leg_guard_pass"] = bool(leg_deep < 15.0 and sim_leg <= 15.0)
    summary["E025_object_no_regression_pass"] = bool(
        _as_float(summary.get("paper_object_Epos_case_m"), 999.0) < 0.10
        and _as_float(summary.get("paper_object_Erot_case_deg"), 999.0) < 25.0
        and bool(summary.get("paper_transport_success", False))
    )
    summary["E025_no_fall_pass"] = bool(not bool(summary.get("E018b_robot_fall_detected", True)))
    summary["E025_strict_success"] = bool(
        summary["E025_contact_closure_pass"]
        and summary["E025_penetration_guard_pass"]
        and summary["E025_leg_guard_pass"]
        and summary["E025_object_no_regression_pass"]
        and summary["E025_no_fall_pass"]
        and summary["E025_support_proxy_unchanged"]
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
        raise SystemExit("No E025 variant results found.")

    keys = sorted({key for row in summaries for key in row.keys()})
    comparison = RESULTS / "comparison.csv"
    with comparison.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summaries)

    aggregate = {
        "num_results": len(summaries),
        "num_E025_strict_success": sum(bool(r["E025_strict_success"]) for r in summaries),
        "num_contact_closure_pass": sum(bool(r["E025_contact_closure_pass"]) for r in summaries),
        "num_penetration_guard_pass": sum(bool(r["E025_penetration_guard_pass"]) for r in summaries),
        "num_leg_guard_pass": sum(bool(r["E025_leg_guard_pass"]) for r in summaries),
        "num_object_no_regression_pass": sum(bool(r["E025_object_no_regression_pass"]) for r in summaries),
        "num_no_fall_pass": sum(bool(r["E025_no_fall_pass"]) for r in summaries),
        "best_contact_pct": max(
            _as_float(r.get("paper_omniretarget_contact_preservation_5cm_pct"), 0.0) for r in summaries
        ),
        "best_variant_by_contact": max(
            summaries,
            key=lambda r: _as_float(r.get("paper_omniretarget_contact_preservation_5cm_pct"), 0.0),
        )["variant"],
        "best_deep_penetration_pct": min(
            _as_float(r.get("paper_omniretarget_robot_object_deep_penetration_duration_pct"), 100.0)
            for r in summaries
        ),
        "best_variant_by_deep_penetration": min(
            summaries,
            key=lambda r: _as_float(r.get("paper_omniretarget_robot_object_deep_penetration_duration_pct"), 100.0),
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
