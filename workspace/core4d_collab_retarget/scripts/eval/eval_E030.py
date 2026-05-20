#!/usr/bin/env python3
"""E030 evaluation for lower-body geometry / surface-control repair."""

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


RESULTS = REPO / "workspace/core4d_collab_retarget/results/E030"
MANIFEST = RESULTS / "manifest.tsv"
QUALITY_AUDIT = REPO / "workspace/core4d_collab_retarget/results/E027/data_quality/case_quality_audit.csv"
SOURCE_MANIFESTS = {
    "E018b": REPO / "workspace/core4d_collab_retarget/results/E018b/manifest.tsv",
    "E022": REPO / "workspace/core4d_collab_retarget/results/E022/manifest.tsv",
    "E023": REPO / "workspace/core4d_collab_retarget/results/E023/manifest.tsv",
    "E024": REPO / "workspace/core4d_collab_retarget/results/E024/manifest.tsv",
    "E025": REPO / "workspace/core4d_collab_retarget/results/E025/manifest.tsv",
    "E029": REPO / "workspace/core4d_collab_retarget/results/E029/manifest.tsv",
}
TARGET_GEOMETRY_CASES = {"box025_p1", "bucket007_p2"}
DIAGNOSTIC_CASES = {"box023_p1", "box023_p2"}
CLEAN_GUARD_CASES = {"box025_p2"}
SHORTCUT_GUARD_CASES = {"bucket005_s2_p1"}


def _read_manifest(path: Path = MANIFEST) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return {row["variant"]: row for row in csv.DictReader(f, delimiter="\t")}


def _read_source_manifests() -> dict[str, dict[str, dict[str, str]]]:
    out: dict[str, dict[str, dict[str, str]]] = {}
    for name, path in SOURCE_MANIFESTS.items():
        if path.is_file():
            with path.open("r", encoding="utf-8", newline="") as f:
                out[name] = {row["variant"]: row for row in csv.DictReader(f, delimiter="\t")}
    return out


def _read_quality() -> dict[str, dict[str, str]]:
    with QUALITY_AUDIT.open("r", encoding="utf-8", newline="") as f:
        return {row["case"]: row for row in csv.DictReader(f)}


def _as_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


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
    case = str(summary["E030_case_slug"])
    contact = _as_float(summary.get("paper_omniretarget_contact_preservation_5cm_pct"), 0.0)
    deep = _as_float(summary.get("paper_omniretarget_robot_object_deep_penetration_duration_pct"), 100.0)
    max_pen = _as_float(summary.get("paper_omniretarget_robot_object_max_penetration_cm"), 999.0)
    obj = _as_float(summary.get("paper_object_Epos_case_m"), 999.0) * 100.0
    ref_leg = _as_float(summary.get("full_ref_leg_box_interference_frames_pct"), 100.0)
    sim_leg = _as_float(summary.get("full_sim_leg_box_interference_frames_pct"), 100.0)
    row: dict[str, Any] = {
        "variant": summary["variant"],
        "case": case,
        "current_contact5_pct": contact,
        "current_deep_pen_pct": deep,
        "current_max_pen_cm": max_pen,
        "current_obj_pos_cm": obj,
        "current_ref_leg_interference_pct": ref_leg,
        "current_sim_leg_interference_pct": sim_leg,
    }
    if not baseline:
        row["baseline_found"] = False
        return row
    base_contact = _as_float(baseline.get("best_contact5_pct"), 0.0)
    base_deep = _as_float(baseline.get("best_deep_pen_pct"), 100.0)
    base_max = _as_float(baseline.get("best_max_pen_cm"), 999.0)
    base_obj = _as_float(baseline.get("best_obj_pos_cm"), 999.0)
    base_ref_leg = _as_float(baseline.get("ref_leg_interference_pct"), float("nan"))
    row.update(
        {
            "baseline_found": True,
            "baseline_variant": baseline.get("selected_best_variant", ""),
            "quality_label": baseline.get("quality_label", ""),
            "baseline_contact5_pct": base_contact,
            "delta_contact5_pp": contact - base_contact,
            "baseline_deep_pen_pct": base_deep,
            "delta_deep_pen_pp": deep - base_deep,
            "improvement_deep_pen_pp": base_deep - deep,
            "baseline_max_pen_cm": base_max,
            "delta_max_pen_cm": max_pen - base_max,
            "baseline_obj_pos_cm": base_obj,
            "delta_obj_pos_cm": obj - base_obj,
            "baseline_ref_leg_interference_pct": base_ref_leg,
            "delta_ref_leg_interference_pp": (
                ref_leg - base_ref_leg if base_ref_leg == base_ref_leg else float("nan")
            ),
        }
    )
    return row


def evaluate_variant(
    variant: str,
    manifest: dict[str, dict[str, str]],
    source_manifests: dict[str, dict[str, dict[str, str]]],
    quality: dict[str, dict[str, str]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    e018b.RESULTS = RESULTS
    e018b.MANIFEST = MANIFEST
    variants = e018b.read_manifest()
    summary = e018b.evaluate_variant(variant, variants)
    row = manifest[variant]
    source = source_manifests[row["E030_source_manifest"]][row["E030_source_variant"]]
    case = row["E030_case_slug"]

    contact = _as_float(summary.get("paper_omniretarget_contact_preservation_5cm_pct"), 0.0)
    deep = _as_float(summary.get("paper_omniretarget_robot_object_deep_penetration_duration_pct"), 100.0)
    max_pen = _as_float(summary.get("paper_omniretarget_robot_object_max_penetration_cm"), 999.0)
    obj_pos = _as_float(summary.get("paper_object_Epos_case_m"), 999.0)
    obj_ori = _as_float(summary.get("paper_object_Erot_case_deg"), 999.0)
    no_fall = not _as_bool(summary.get("E018b_robot_fall_detected", True))
    full_ref = _as_float(summary.get("full_ref_leg_box_interference_frames_pct"), 100.0)
    case_ref = _as_float(summary.get("case_window_ref_leg_box_interference_frames_pct"), 100.0)
    full_sim = _as_float(summary.get("full_sim_leg_box_interference_frames_pct"), 100.0)
    case_sim = _as_float(summary.get("case_window_sim_leg_box_interference_frames_pct"), 100.0)
    disabled_pairs = [x for x in row.get("E030_disabled_leg_object_pairs", "").split(",") if x]

    summary["E030_source_manifest"] = row["E030_source_manifest"]
    summary["E030_source_variant"] = row["E030_source_variant"]
    summary["E030_source_derived_task"] = row["E030_source_derived_task"]
    summary["E030_case_slug"] = case
    summary["E030_role"] = row["E030_role"]
    summary["E030_wave"] = row["E030_wave"]
    summary["E030_patch_mode"] = row["E030_patch_mode"]
    summary["E030_disabled_leg_object_pairs"] = row["E030_disabled_leg_object_pairs"]
    summary["E030_disabled_pair_count"] = len(disabled_pairs)
    summary["E030_shrunk_leg_geoms"] = row["E030_shrunk_leg_geoms"]
    summary["E030_lowerbody_radius"] = _as_float(row["E030_lowerbody_radius"])
    summary["E030_foot_radius"] = _as_float(row["E030_foot_radius"])
    summary["E030_hold_contact_rew_scale"] = _as_float(row["E030_hold_contact_rew_scale"])
    summary["E030_contact_hdmi_gain"] = _as_float(row["E030_contact_hdmi_gain"])
    summary["E030_robot_object_barrier_scale"] = _as_float(row["E030_robot_object_barrier_scale"])
    summary["E030_contact_penetration_gate_enabled"] = row[
        "E030_contact_penetration_gate_enabled"
    ]
    summary["E030_leg_object_penalty_scale"] = _as_float(row["E030_leg_object_penalty_scale"])
    summary["E030_posture_contact_gate_enabled"] = row["E030_posture_contact_gate_enabled"]
    summary["E030_support_proxy_unchanged"] = _support_proxy_unchanged(row, source)
    summary["E030_quality_label"] = quality.get(case, {}).get("quality_label", "")

    summary["E030_no_pair_deletion_pass"] = len(disabled_pairs) == 0
    summary["E030_ref_geometry_pass"] = bool(full_ref < 15.0 and case_ref < 15.0)
    summary["E030_sim_leg_artifact_pass"] = bool(full_sim < 15.0 and case_sim < 15.0)
    summary["E030_penetration_guard_pass"] = bool(deep <= 15.0 and max_pen <= 5.0)
    summary["E030_object_no_regression_pass"] = bool(obj_pos <= 0.08 and obj_ori <= 25.0)
    summary["E030_no_fall_pass"] = bool(no_fall)

    if case == "box025_p1":
        contact_goal = contact >= 70.0
    elif case == "bucket007_p2":
        contact_goal = contact >= 60.0
    elif case in DIAGNOSTIC_CASES:
        base_contact = _as_float(quality.get(case, {}).get("best_contact5_pct"), 0.0)
        contact_goal = (contact - base_contact) >= 10.0
    elif case == "box025_p2":
        contact_goal = contact >= 80.0
    else:
        contact_goal = contact >= 70.0
    summary["E030_contact_goal_pass"] = bool(contact_goal)

    role = row["E030_role"]
    summary["E030_target_geometry_success"] = bool(
        role == "target_geometry"
        and summary["E030_ref_geometry_pass"]
        and summary["E030_sim_leg_artifact_pass"]
        and summary["E030_contact_goal_pass"]
        and summary["E030_penetration_guard_pass"]
        and summary["E030_object_no_regression_pass"]
        and summary["E030_no_fall_pass"]
        and summary["E030_no_pair_deletion_pass"]
        and summary["E030_support_proxy_unchanged"]
    )
    summary["E030_diagnostic_surface_signal"] = bool(
        role == "diagnostic_surface" and summary["E030_contact_goal_pass"]
    )
    summary["E030_clean_guard_pass"] = bool(
        role == "clean_guard"
        and summary["E030_contact_goal_pass"]
        and deep <= 10.0
        and max_pen <= 5.0
        and summary["E030_object_no_regression_pass"]
        and summary["E030_no_fall_pass"]
    )
    summary["E030_shortcut_guard_pass"] = bool(
        role == "shortcut_guard"
        and summary["E030_penetration_guard_pass"]
        and summary["E030_sim_leg_artifact_pass"]
        and summary["E030_object_no_regression_pass"]
        and summary["E030_no_fall_pass"]
    )
    summary["E030_success"] = bool(
        summary["E030_target_geometry_success"]
        or summary["E030_diagnostic_surface_signal"]
        or summary["E030_clean_guard_pass"]
        or summary["E030_shortcut_guard_pass"]
    )

    delta = _baseline_delta(summary, quality.get(case))
    ref_row = {
        "variant": variant,
        "case": case,
        "role": role,
        "patch_mode": row["E030_patch_mode"],
        "full_ref_leg_box_interference_pct": full_ref,
        "case_ref_leg_box_interference_pct": case_ref,
        "ref_geometry_pass": summary["E030_ref_geometry_pass"],
        "disabled_pair_count": len(disabled_pairs),
        "shrunk_leg_geoms": row["E030_shrunk_leg_geoms"],
    }
    sim_row = {
        "variant": variant,
        "case": case,
        "role": role,
        "full_sim_leg_box_interference_pct": full_sim,
        "case_sim_leg_box_interference_pct": case_sim,
        "contact5_pct": contact,
        "deep_pen_pct": deep,
        "max_pen_cm": max_pen,
        "object_pos_cm": obj_pos * 100.0,
        "object_ori_deg": obj_ori,
        "no_fall": no_fall,
        "sim_leg_artifact_pass": summary["E030_sim_leg_artifact_pass"],
        "penetration_guard_pass": summary["E030_penetration_guard_pass"],
    }
    _write_summary(summary)
    return summary, delta, ref_row, sim_row


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
    quality = _read_quality()
    selected = _normalize_args(sys.argv[1:], manifest)
    summaries: list[dict[str, Any]] = []
    deltas: list[dict[str, Any]] = []
    ref_rows: list[dict[str, Any]] = []
    sim_rows: list[dict[str, Any]] = []
    for variant in selected:
        if variant not in manifest:
            print(f"[SKIP] unknown variant {variant}")
            continue
        try:
            summary, delta, ref_row, sim_row = evaluate_variant(variant, manifest, sources, quality)
        except FileNotFoundError as exc:
            print(f"[SKIP] {exc}")
            continue
        summaries.append(summary)
        deltas.append(delta)
        ref_rows.append(ref_row)
        sim_rows.append(sim_row)
    if not summaries:
        raise SystemExit("No E030 variant results found.")

    _write_csv(RESULTS / "comparison.csv", summaries)
    _write_csv(RESULTS / "baseline_delta.csv", deltas)
    _write_csv(RESULTS / "ref_interference.csv", ref_rows)
    _write_csv(RESULTS / "sim_interference.csv", sim_rows)

    target_rows = [r for r in summaries if str(r.get("E030_case_slug")) in TARGET_GEOMETRY_CASES]
    diagnostic_rows = [r for r in summaries if str(r.get("E030_case_slug")) in DIAGNOSTIC_CASES]
    clean_guard_rows = [r for r in summaries if str(r.get("E030_case_slug")) in CLEAN_GUARD_CASES]
    shortcut_guard_rows = [r for r in summaries if str(r.get("E030_case_slug")) in SHORTCUT_GUARD_CASES]
    aggregate = {
        "num_results": len(summaries),
        "num_target_geometry_results": len(target_rows),
        "num_target_geometry_success": sum(bool(r["E030_target_geometry_success"]) for r in target_rows),
        "num_diagnostic_results": len(diagnostic_rows),
        "num_diagnostic_surface_signal": sum(
            bool(r["E030_diagnostic_surface_signal"]) for r in diagnostic_rows
        ),
        "num_clean_guard_results": len(clean_guard_rows),
        "num_clean_guard_pass": sum(bool(r["E030_clean_guard_pass"]) for r in clean_guard_rows),
        "num_shortcut_guard_results": len(shortcut_guard_rows),
        "num_shortcut_guard_pass": sum(
            bool(r["E030_shortcut_guard_pass"]) for r in shortcut_guard_rows
        ),
        "num_no_pair_deletion_pass": sum(bool(r["E030_no_pair_deletion_pass"]) for r in summaries),
        "best_target_ref_leg_interference_pct": min(
            (
                _as_float(r.get("full_ref_leg_box_interference_frames_pct"), 100.0)
                for r in target_rows
            ),
            default=100.0,
        ),
        "best_target_contact5_pct": max(
            (
                _as_float(r.get("paper_omniretarget_contact_preservation_5cm_pct"), 0.0)
                for r in target_rows
            ),
            default=0.0,
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
