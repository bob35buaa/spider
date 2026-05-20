#!/usr/bin/env python3
"""E029 evaluation for bucket001 stability / posture-valid contact control."""

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


RESULTS = REPO / "workspace/core4d_collab_retarget/results/E029"
MANIFEST = RESULTS / "manifest.tsv"
SOURCE_MANIFESTS = {
    "E018b": REPO / "workspace/core4d_collab_retarget/results/E018b/manifest.tsv",
    "E024": REPO / "workspace/core4d_collab_retarget/results/E024/manifest.tsv",
}
SOURCE_RESULTS = {
    "E018b": REPO / "workspace/core4d_collab_retarget/results/E018b",
    "E024": REPO / "workspace/core4d_collab_retarget/results/E024",
}
TARGET_CASE = "bucket001_p1"
GUARD_CASES = {"bucket001_p2", "box025_p2"}


def _read_manifest(path: Path = MANIFEST) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return {row["variant"]: row for row in csv.DictReader(f, delimiter="\t")}


def _read_source_manifests() -> dict[str, dict[str, dict[str, str]]]:
    out: dict[str, dict[str, dict[str, str]]] = {}
    for name, path in SOURCE_MANIFESTS.items():
        out[name] = _read_manifest(path)
    return out


def _as_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _source_summary(source_name: str, source_variant: str) -> dict[str, Any]:
    path = SOURCE_RESULTS[source_name] / f"eval_summary_{source_variant}.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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


def _baseline_delta(summary: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    def cur(name: str, default: float = 0.0) -> float:
        return _as_float(summary.get(name), default)

    def base(name: str, default: float = 0.0) -> float:
        return _as_float(baseline.get(name), default)

    pelvis = cur("full_pelvis_z_min_m", -999.0)
    base_pelvis = base("full_pelvis_z_min_m", -999.0)
    contact = cur("paper_omniretarget_contact_preservation_5cm_pct", 0.0)
    base_contact = base("paper_omniretarget_contact_preservation_5cm_pct", 0.0)
    deep = cur("paper_omniretarget_robot_object_deep_penetration_duration_pct", 100.0)
    base_deep = base("paper_omniretarget_robot_object_deep_penetration_duration_pct", 100.0)
    max_pen = cur("paper_omniretarget_robot_object_max_penetration_cm", 999.0)
    base_max_pen = base("paper_omniretarget_robot_object_max_penetration_cm", 999.0)
    obj = cur("paper_object_Epos_case_m", 999.0) * 100.0
    base_obj = base("paper_object_Epos_case_m", 999.0) * 100.0
    return {
        "variant": summary["variant"],
        "case": summary["E029_case_slug"],
        "source_variant": summary["E029_source_variant"],
        "baseline_found": bool(baseline),
        "baseline_pelvis_min_m": base_pelvis,
        "current_pelvis_min_m": pelvis,
        "delta_pelvis_min_m": pelvis - base_pelvis,
        "baseline_contact5_pct": base_contact,
        "current_contact5_pct": contact,
        "delta_contact5_pp": contact - base_contact,
        "baseline_deep_pen_pct": base_deep,
        "current_deep_pen_pct": deep,
        "delta_deep_pen_pp": deep - base_deep,
        "improvement_deep_pen_pp": base_deep - deep,
        "baseline_max_pen_cm": base_max_pen,
        "current_max_pen_cm": max_pen,
        "delta_max_pen_cm": max_pen - base_max_pen,
        "baseline_obj_pos_cm": base_obj,
        "current_obj_pos_cm": obj,
        "delta_obj_pos_cm": obj - base_obj,
    }


def evaluate_variant(
    variant: str,
    manifest: dict[str, dict[str, str]],
    source_manifests: dict[str, dict[str, dict[str, str]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    e018b.RESULTS = RESULTS
    e018b.MANIFEST = MANIFEST
    variants = e018b.read_manifest()
    summary = e018b.evaluate_variant(variant, variants)
    row = manifest[variant]
    source_name = row["E029_source_manifest"]
    source_variant = row["E029_source_variant"]
    source = source_manifests[source_name][source_variant]
    baseline = _source_summary(source_name, source_variant)

    case = row["E029_case_slug"]
    pelvis_min = _as_float(summary.get("full_pelvis_z_min_m"), -999.0)
    contact = _as_float(summary.get("paper_omniretarget_contact_preservation_5cm_pct"), 0.0)
    deep = _as_float(summary.get("paper_omniretarget_robot_object_deep_penetration_duration_pct"), 100.0)
    max_pen = _as_float(summary.get("paper_omniretarget_robot_object_max_penetration_cm"), 999.0)
    obj_pos = _as_float(summary.get("paper_object_Epos_case_m"), 999.0)
    obj_ori = _as_float(summary.get("paper_object_Erot_case_deg"), 999.0)
    no_fall = not bool(summary.get("E018b_robot_fall_detected", True))
    base_contact = _as_float(
        baseline.get("paper_omniretarget_contact_preservation_5cm_pct"), 0.0
    )
    base_deep = _as_float(
        baseline.get("paper_omniretarget_robot_object_deep_penetration_duration_pct"),
        100.0,
    )

    for key in [
        "source_manifest",
        "source_variant",
        "case_slug",
        "role",
        "wave",
        "stability_penalty_scale",
        "stability_penalty_threshold",
        "local_frame_root_sigma",
        "contact_hdmi_gain",
        "contact_hdmi_sigma",
        "hold_contact_rew_scale",
        "robot_object_penalty_scale",
        "upright_barrier_scale",
        "upright_barrier_threshold_m",
        "upright_score_cap_scale",
        "upright_score_cap_threshold_m",
        "root_tilt_penalty_scale",
        "root_tilt_penalty_max_deg",
        "foot_support_penalty_scale",
        "posture_contact_gate_enabled",
        "posture_contact_gate_require_foot_support",
        "notes",
    ]:
        row_key = f"E029_{key}"
        summary[row_key] = row.get(row_key, row.get(key, ""))

    summary["E029_support_proxy_unchanged"] = _support_proxy_unchanged(row, source)
    summary["E029_no_fall_pass"] = bool(no_fall)
    summary["E029_pelvis_min_pass"] = bool(pelvis_min >= 0.45)
    summary["E029_pelvis_target_pass"] = bool(pelvis_min >= 0.55)
    summary["E029_object_no_regression_pass"] = bool(obj_pos <= 0.10 and obj_ori <= 25.0)
    summary["E029_stability_pass"] = bool(
        summary["E029_no_fall_pass"]
        and summary["E029_pelvis_min_pass"]
        and summary["E029_object_no_regression_pass"]
    )
    summary["E029_useful_signal"] = bool(
        case == TARGET_CASE
        and summary["E029_stability_pass"]
        and contact > 20.0
    )
    summary["E029_p1_strict_target"] = bool(
        case == TARGET_CASE
        and summary["E029_stability_pass"]
        and contact >= 50.0
        and deep <= 20.0
        and max_pen <= 5.0
    )
    summary["E029_p2_guard_pass"] = bool(
        case == "bucket001_p2"
        and no_fall
        and pelvis_min >= 0.55
        and contact >= 50.0
        and obj_pos <= 0.08
    )
    summary["E029_box025_guard_pass"] = bool(
        case == "box025_p2"
        and no_fall
        and obj_pos <= 0.08
        and contact >= max(0.0, base_contact - 20.0)
        and deep <= max(15.0, base_deep + 5.0)
    )
    summary["E029_guard_pass"] = bool(
        summary["E029_p2_guard_pass"] or summary["E029_box025_guard_pass"]
    )
    summary["E029_success"] = bool(
        summary["E029_p1_strict_target"] or summary["E029_guard_pass"]
    )
    delta = _baseline_delta(summary, baseline)
    summary["E029_contact_delta_pp"] = _as_float(delta.get("delta_contact5_pp"), 0.0)
    summary["E029_pelvis_delta_m"] = _as_float(delta.get("delta_pelvis_min_m"), 0.0)
    summary["E029_deep_pen_improvement_pp"] = _as_float(
        delta.get("improvement_deep_pen_pp"), 0.0
    )
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
    selected = _normalize_args(sys.argv[1:], manifest)
    summaries: list[dict[str, Any]] = []
    deltas: list[dict[str, Any]] = []
    for variant in selected:
        if variant not in manifest:
            print(f"[SKIP] unknown variant {variant}")
            continue
        try:
            summary, delta = evaluate_variant(variant, manifest, sources)
        except FileNotFoundError as exc:
            print(f"[SKIP] {exc}")
            continue
        summaries.append(summary)
        deltas.append(delta)
    if not summaries:
        raise SystemExit("No E029 variant results found.")

    _write_csv(RESULTS / "comparison.csv", summaries)
    _write_csv(RESULTS / "baseline_delta.csv", deltas)

    target_rows = [r for r in summaries if str(r.get("E029_case_slug")) == TARGET_CASE]
    guard_rows = [r for r in summaries if str(r.get("E029_case_slug")) in GUARD_CASES]
    aggregate = {
        "num_results": len(summaries),
        "num_target_results": len(target_rows),
        "num_guard_results": len(guard_rows),
        "num_E029_success": sum(bool(r["E029_success"]) for r in summaries),
        "num_target_stability_pass": sum(bool(r["E029_stability_pass"]) for r in target_rows),
        "num_target_useful_signal": sum(bool(r["E029_useful_signal"]) for r in target_rows),
        "num_target_p1_strict_target": sum(
            bool(r["E029_p1_strict_target"]) for r in target_rows
        ),
        "num_guard_pass": sum(bool(r["E029_guard_pass"]) for r in guard_rows),
        "best_target_pelvis_min_m": max(
            (_as_float(r.get("full_pelvis_z_min_m"), -999.0) for r in target_rows),
            default=-999.0,
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
    if target_rows:
        aggregate["best_target_variant_by_pelvis"] = max(
            target_rows,
            key=lambda r: _as_float(r.get("full_pelvis_z_min_m"), -999.0),
        )["variant"]
        aggregate["best_target_variant_by_contact"] = max(
            target_rows,
            key=lambda r: _as_float(
                r.get("paper_omniretarget_contact_preservation_5cm_pct"), 0.0
            ),
        )["variant"]
    (RESULTS / "aggregate_summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Wrote {RESULTS / 'comparison.csv'}")
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
