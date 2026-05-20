#!/usr/bin/env python3
"""E024 evaluation for bucket001 stability repair."""

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


RESULTS = REPO / "workspace/core4d_collab_retarget/results/E024"
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


def _contact_guard(row: dict[str, str], contact_pct: float) -> bool:
    case_slug = row.get("E024_case_slug", "")
    if case_slug.endswith("_p1"):
        return contact_pct > 0.0
    if case_slug.endswith("_p2"):
        return contact_pct >= 50.0
    return contact_pct >= 50.0


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
    source = e018b_manifest[row["E024_source_variant"]]

    pelvis_min = _as_float(summary.get("full_pelvis_z_min_m"), -999.0)
    deep = _as_float(summary.get("paper_omniretarget_robot_object_deep_penetration_duration_pct"), 100.0)
    sim_leg = _as_float(summary.get("case_window_sim_leg_box_interference_frames_pct"), 100.0)
    contact = _as_float(summary.get("paper_omniretarget_contact_preservation_5cm_pct"), 0.0)

    summary["E024_source_variant"] = row["E024_source_variant"]
    summary["E024_case_slug"] = row["E024_case_slug"]
    summary["E024_stability_penalty_scale"] = _as_float(row["E024_stability_penalty_scale"])
    summary["E024_stability_penalty_threshold"] = _as_float(row["E024_stability_penalty_threshold"])
    summary["E024_local_frame_root_sigma"] = _as_float(row["E024_local_frame_root_sigma"])
    summary["E024_contact_hdmi_gain"] = _as_float(row["E024_contact_hdmi_gain"])
    summary["E024_contact_hdmi_sigma"] = _as_float(row["E024_contact_hdmi_sigma"])
    summary["E024_support_proxy_unchanged"] = _support_proxy_unchanged(row, source)
    summary["E024_no_fall_pass"] = bool(not bool(summary.get("E018b_robot_fall_detected", True)))
    summary["E024_pelvis_min_pass"] = bool(pelvis_min >= 0.45)
    summary["E024_pelvis_target_pass"] = bool(pelvis_min >= 0.55)
    summary["E024_stability_pass"] = bool(summary["E024_no_fall_pass"] and summary["E024_pelvis_min_pass"])
    summary["E024_object_no_regression_pass"] = bool(
        _as_float(summary.get("paper_object_Epos_case_m"), 999.0) < 0.10
        and _as_float(summary.get("paper_object_Erot_case_deg"), 999.0) < 25.0
        and bool(summary.get("paper_transport_success", False))
    )
    summary["E024_artifact_guard_pass"] = bool(deep <= 20.0 and sim_leg <= 15.0)
    summary["E024_contact_guard_pass"] = _contact_guard(row, contact)
    summary["E024_success"] = bool(
        summary["E024_stability_pass"]
        and summary["E024_object_no_regression_pass"]
        and summary["E024_artifact_guard_pass"]
        and summary["E024_contact_guard_pass"]
        and summary["E024_support_proxy_unchanged"]
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
        raise SystemExit("No E024 variant results found.")

    keys = sorted({key for row in summaries for key in row.keys()})
    comparison = RESULTS / "comparison.csv"
    with comparison.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summaries)

    aggregate = {
        "num_results": len(summaries),
        "num_E024_success": sum(bool(r["E024_success"]) for r in summaries),
        "num_stability_pass": sum(bool(r["E024_stability_pass"]) for r in summaries),
        "num_pelvis_target_pass": sum(bool(r["E024_pelvis_target_pass"]) for r in summaries),
        "num_object_no_regression_pass": sum(bool(r["E024_object_no_regression_pass"]) for r in summaries),
        "num_artifact_guard_pass": sum(bool(r["E024_artifact_guard_pass"]) for r in summaries),
        "num_contact_guard_pass": sum(bool(r["E024_contact_guard_pass"]) for r in summaries),
        "best_pelvis_min_m": max(_as_float(r.get("full_pelvis_z_min_m"), -999.0) for r in summaries),
        "best_variant_by_pelvis_min": max(
            summaries,
            key=lambda r: _as_float(r.get("full_pelvis_z_min_m"), -999.0),
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
