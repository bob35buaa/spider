#!/usr/bin/env python3
"""E028 evaluation for hard no-penetration / surface feasibility."""

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


RESULTS = REPO / "workspace/core4d_collab_retarget/results/E028"
MANIFEST = RESULTS / "manifest.tsv"
SOURCE_MANIFESTS = {
    "E018b": REPO / "workspace/core4d_collab_retarget/results/E018b/manifest.tsv",
    "E024": REPO / "workspace/core4d_collab_retarget/results/E024/manifest.tsv",
    "E025": REPO / "workspace/core4d_collab_retarget/results/E025/manifest.tsv",
}
QUALITY_AUDIT = REPO / "workspace/core4d_collab_retarget/results/E027/data_quality/case_quality_audit.csv"
TARGET_CASES = {"bucket005_s2_p1", "bucket005_s2_p2", "bucket007_p1", "bucket001_p2"}
GUARD_CASES = {"box025_p2"}


def _read_manifest(path: Path = MANIFEST) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return {row["variant"]: row for row in csv.DictReader(f, delimiter="\t")}


def _read_source_manifests() -> dict[str, dict[str, dict[str, str]]]:
    out: dict[str, dict[str, dict[str, str]]] = {}
    for name, path in SOURCE_MANIFESTS.items():
        with path.open("r", encoding="utf-8", newline="") as f:
            out[name] = {row["variant"]: row for row in csv.DictReader(f, delimiter="\t")}
    return out


def _read_baselines() -> dict[str, dict[str, str]]:
    with QUALITY_AUDIT.open("r", encoding="utf-8", newline="") as f:
        return {row["case"]: row for row in csv.DictReader(f)}


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
        "weld_solref_timeconst",
        "weld_solimp_1",
        "weld_solimp_2",
        "weld_solimp_width",
        "support_proxy_gravity_scale",
        "anchor_face",
        "canonical_z_frac",
    ]
    return all(str(row.get(key, "")) == str(source.get(key, "")) for key in keys)


def _baseline_delta(summary: dict[str, Any], baseline: dict[str, str] | None) -> dict[str, Any]:
    case = str(summary["E028_case_slug"])
    contact = _as_float(summary.get("paper_omniretarget_contact_preservation_5cm_pct"), 0.0)
    deep = _as_float(summary.get("paper_omniretarget_robot_object_deep_penetration_duration_pct"), 100.0)
    max_pen = _as_float(summary.get("paper_omniretarget_robot_object_max_penetration_cm"), 999.0)
    obj = _as_float(summary.get("paper_object_Epos_case_m"), 999.0) * 100.0
    if not baseline:
        return {
            "variant": summary["variant"],
            "case": case,
            "baseline_found": False,
            "current_contact5_pct": contact,
            "current_deep_pen_pct": deep,
            "current_max_pen_cm": max_pen,
            "current_obj_pos_cm": obj,
        }
    base_contact = _as_float(baseline.get("best_contact5_pct"), 0.0)
    base_deep = _as_float(baseline.get("best_deep_pen_pct"), 100.0)
    base_max = _as_float(baseline.get("best_max_pen_cm"), 999.0)
    base_obj = _as_float(baseline.get("best_obj_pos_cm"), 999.0)
    return {
        "variant": summary["variant"],
        "case": case,
        "baseline_found": True,
        "baseline_variant": baseline.get("selected_best_variant", ""),
        "quality_label": baseline.get("quality_label", ""),
        "baseline_contact5_pct": base_contact,
        "current_contact5_pct": contact,
        "delta_contact5_pp": contact - base_contact,
        "baseline_deep_pen_pct": base_deep,
        "current_deep_pen_pct": deep,
        "delta_deep_pen_pp": deep - base_deep,
        "improvement_deep_pen_pp": base_deep - deep,
        "baseline_max_pen_cm": base_max,
        "current_max_pen_cm": max_pen,
        "delta_max_pen_cm": max_pen - base_max,
        "baseline_obj_pos_cm": base_obj,
        "current_obj_pos_cm": obj,
        "delta_obj_pos_cm": obj - base_obj,
    }


def evaluate_variant(
    variant: str,
    manifest: dict[str, dict[str, str]],
    source_manifests: dict[str, dict[str, dict[str, str]]],
    baselines: dict[str, dict[str, str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    e018b.RESULTS = RESULTS
    e018b.MANIFEST = MANIFEST
    variants = e018b.read_manifest()
    summary = e018b.evaluate_variant(variant, variants)
    row = manifest[variant]
    source = source_manifests[row["E028_source_manifest"]][row["E028_source_variant"]]
    case = row["E028_case_slug"]

    contact = _as_float(summary.get("paper_omniretarget_contact_preservation_5cm_pct"), 0.0)
    deep = _as_float(summary.get("paper_omniretarget_robot_object_deep_penetration_duration_pct"), 100.0)
    max_pen = _as_float(summary.get("paper_omniretarget_robot_object_max_penetration_cm"), 999.0)
    hand_deep = _as_float(summary.get("paper_omniretarget_hand_object_deep_penetration_duration_pct"), 100.0)
    leg_deep = _as_float(summary.get("paper_omniretarget_leg_object_deep_penetration_duration_pct"), 100.0)
    obj_pos = _as_float(summary.get("paper_object_Epos_case_m"), 999.0)
    obj_ori = _as_float(summary.get("paper_object_Erot_case_deg"), 999.0)
    no_fall = not bool(summary.get("E018b_robot_fall_detected", True))

    summary["E028_source_manifest"] = row["E028_source_manifest"]
    summary["E028_source_variant"] = row["E028_source_variant"]
    summary["E028_case_slug"] = case
    summary["E028_role"] = row["E028_role"]
    summary["E028_wave"] = row["E028_wave"]
    summary["E028_hold_contact_rew_scale"] = _as_float(row["E028_hold_contact_rew_scale"])
    summary["E028_contact_hdmi_gain"] = _as_float(row["E028_contact_hdmi_gain"])
    summary["E028_contact_hdmi_sigma"] = _as_float(row["E028_contact_hdmi_sigma"])
    summary["E028_contact_hdmi_ori_weight"] = _as_float(row["E028_contact_hdmi_ori_weight"])
    summary["E028_contact_hdmi_ori_mode"] = row["E028_contact_hdmi_ori_mode"]
    summary["E028_robot_object_barrier_scale"] = _as_float(row["E028_robot_object_barrier_scale"])
    summary["E028_robot_object_barrier_margin_m"] = _as_float(row["E028_robot_object_barrier_margin_m"])
    summary["E028_robot_object_score_cap_scale"] = _as_float(row["E028_robot_object_score_cap_scale"])
    summary["E028_robot_object_score_cap_threshold_m"] = _as_float(
        row["E028_robot_object_score_cap_threshold_m"]
    )
    summary["E028_contact_penetration_gate_enabled"] = row[
        "E028_contact_penetration_gate_enabled"
    ]
    summary["E028_stability_penalty_scale"] = _as_float(row["E028_stability_penalty_scale"])
    summary["E028_notes"] = row["E028_notes"]
    summary["E028_support_proxy_unchanged"] = _support_proxy_unchanged(row, source)
    summary["E028_contact_closure_pass"] = bool(contact >= 70.0)
    summary["E028_penetration_guard_pass"] = bool(deep <= 15.0 and max_pen <= 5.0)
    summary["E028_hand_penetration_guard_pass"] = bool(hand_deep <= 15.0)
    summary["E028_leg_penetration_guard_pass"] = bool(leg_deep <= 20.0)
    summary["E028_object_no_regression_pass"] = bool(obj_pos <= 0.08 and obj_ori <= 25.0)
    summary["E028_no_fall_pass"] = bool(no_fall)
    summary["E028_strict_success"] = bool(
        summary["E028_contact_closure_pass"]
        and summary["E028_penetration_guard_pass"]
        and summary["E028_object_no_regression_pass"]
        and summary["E028_no_fall_pass"]
        and summary["E028_support_proxy_unchanged"]
    )
    delta = _baseline_delta(summary, baselines.get(case))
    summary["E028_deep_pen_improvement_pp"] = _as_float(delta.get("improvement_deep_pen_pp"), 0.0)
    summary["E028_contact_delta_pp"] = _as_float(delta.get("delta_contact5_pp"), 0.0)
    _write_summary(summary)
    return summary, delta


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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    manifest = _read_manifest()
    sources = _read_source_manifests()
    baselines = _read_baselines()
    selected = _normalize_args(sys.argv[1:], manifest)
    summaries: list[dict[str, Any]] = []
    deltas: list[dict[str, Any]] = []
    for variant in selected:
        if variant not in manifest:
            print(f"[SKIP] unknown variant {variant}")
            continue
        try:
            summary, delta = evaluate_variant(variant, manifest, sources, baselines)
        except FileNotFoundError as exc:
            print(f"[SKIP] {exc}")
            continue
        summaries.append(summary)
        deltas.append(delta)
    if not summaries:
        raise SystemExit("No E028 variant results found.")

    _write_csv(RESULTS / "comparison.csv", summaries)
    _write_csv(RESULTS / "baseline_delta.csv", deltas)

    target_rows = [r for r in summaries if str(r.get("E028_case_slug")) in TARGET_CASES]
    guard_rows = [r for r in summaries if str(r.get("E028_case_slug")) in GUARD_CASES]
    target_improvements = [
        _as_float(r.get("E028_deep_pen_improvement_pp"), 0.0) for r in target_rows
    ]
    aggregate = {
        "num_results": len(summaries),
        "num_target_results": len(target_rows),
        "num_guard_results": len(guard_rows),
        "num_E028_strict_success": sum(bool(r["E028_strict_success"]) for r in summaries),
        "num_target_E028_strict_success": sum(bool(r["E028_strict_success"]) for r in target_rows),
        "num_target_penetration_guard_pass": sum(
            bool(r["E028_penetration_guard_pass"]) for r in target_rows
        ),
        "num_target_object_no_regression_pass": sum(
            bool(r["E028_object_no_regression_pass"]) for r in target_rows
        ),
        "num_target_no_fall_pass": sum(bool(r["E028_no_fall_pass"]) for r in target_rows),
        "num_guard_strict_success": sum(bool(r["E028_strict_success"]) for r in guard_rows),
        "mean_target_deep_pen_improvement_pp": (
            float(sum(target_improvements) / len(target_improvements)) if target_improvements else 0.0
        ),
        "best_target_deep_penetration_pct": min(
            (
                _as_float(r.get("paper_omniretarget_robot_object_deep_penetration_duration_pct"), 100.0)
                for r in target_rows
            ),
            default=100.0,
        ),
        "best_target_variant_by_deep_penetration": (
            min(
                target_rows,
                key=lambda r: _as_float(
                    r.get("paper_omniretarget_robot_object_deep_penetration_duration_pct"),
                    100.0,
                ),
            )["variant"]
            if target_rows
            else ""
        ),
        "variants": [str(r["variant"]) for r in summaries],
    }
    (RESULTS / "aggregate_summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Wrote {RESULTS / 'comparison.csv'}")
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
