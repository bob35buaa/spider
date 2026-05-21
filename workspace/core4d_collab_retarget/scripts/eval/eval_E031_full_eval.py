#!/usr/bin/env python3
"""E031 conservative best-dynamic assembly after E027-E030.

This is an offline post-processor. It extends the E026 full-eval ledger with
E027 data-quality caveats and E028-E030 diagnostic rows, while keeping the
best-positive dynamic pool conservative: E018b + E022-E025 only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


THIS = Path(__file__).resolve()
REPO = THIS.parents[4]

CASES_13 = [
    "box021_p1",
    "box021_p2",
    "box023_p1",
    "box023_p2",
    "box025_p1",
    "box025_p2",
    "bucket001_p1",
    "bucket001_p2",
    "bucket005_s2_p1",
    "bucket005_s2_p2",
    "bucket007_p1",
    "bucket007_p2",
    "desk021_p1",
]
P0_EXCLUDED = {"desk021_p1", "box021_p1", "box021_p2", "bucket001_p1"}
CASES_9 = [case for case in CASES_13 if case not in P0_EXCLUDED]

RESULTS = REPO / "workspace/core4d_collab_retarget/results/E031_full_eval"
COLLAB_RESULTS = REPO / "workspace/core4d_collab_retarget/results"
E081_RESULTS = REPO / "workspace/core4d/results/E081"
QUALITY_AUDIT = COLLAB_RESULTS / "E027/data_quality/case_quality_audit.csv"

BEST_POSITIVE_METHODS = {
    "spider_E018b",
    "spider_E022",
    "spider_E023",
    "spider_E024",
    "spider_E025",
}
DIAGNOSTIC_METHODS = {"spider_E028", "spider_E029", "spider_E030"}
BEST_METHOD_NAME = "spider_best_conservative_E018b_E022_E025"


def _coerce(value: str | None) -> Any:
    if value is None or value == "":
        return None
    low = value.lower()
    if low in {"true", "false"}:
        return low == "true"
    try:
        int_value = int(value)
        if str(int_value) == value:
            return int_value
    except ValueError:
        pass
    try:
        float_value = float(value)
    except ValueError:
        return value
    if math.isnan(float_value):
        return None
    return float_value


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return [{key: _coerce(value) for key, value in row.items()} for row in csv.DictReader(f)]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in keys} for row in rows])


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _as_float(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if value is None or isinstance(value, bool):
            continue
        try:
            out = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(out):
            return out
    return None


def _as_bool(row: dict[str, Any], *keys: str) -> bool | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in {"true", "false"}:
            return value.lower() == "true"
    return None


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "-"
        return f"{value:.{digits}f}"
    return str(value)


def _mean(values: list[Any]) -> float | None:
    floats = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
    ]
    return sum(floats) / len(floats) if floats else None


def _source_exp(variant: str) -> str:
    match = re.match(r"(E\d+[a-z]?)_", variant)
    return match.group(1) if match else "unknown"


def short_case(value: Any) -> str | None:
    text = str(value or "")
    if text in CASES_13:
        return text
    for prefix in (
        "E018b_",
        "E018_",
        "E022_",
        "E023_",
        "E024_",
        "E025_",
        "E026_E081_",
        "E028_",
        "E029_",
        "E030_",
        "E081_",
        "holosoma_v2_kinematic_",
    ):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    text = text.replace("_person1", "_p1").replace("_person2", "_p2")
    for case in CASES_13:
        if case in text:
            return case
    match = re.search(r"(box\d+|desk\d+|bucket\d+(?:_s2)?)[_-]p([12])", text)
    if match:
        candidate = f"{match.group(1)}_p{match.group(2)}"
        return candidate if candidate in CASES_13 else None
    match = re.search(r"(box\d+|desk\d+|bucket\d+(?:_s2)?)[_-]person([12])", text)
    if match:
        candidate = f"{match.group(1)}_p{match.group(2)}"
        return candidate if candidate in CASES_13 else None
    return None


def _case_from_row(row: dict[str, Any]) -> str | None:
    for key in ("E028_case_slug", "E029_case_slug", "E030_case_slug", "case", "variant", "source_task"):
        case = short_case(row.get(key))
        if case:
            return case
    return None


def _strict_success_for_method(method: str, row: dict[str, Any]) -> bool | None:
    if method == "spider_E028":
        return _as_bool(row, "E028_strict_success")
    if method == "spider_E029":
        return _as_bool(row, "E029_success", "E029_guard_pass", "E029_p1_strict_target")
    if method == "spider_E030":
        return _as_bool(
            row,
            "E030_success",
            "E030_target_geometry_success",
            "E030_clean_guard_pass",
            "E030_shortcut_guard_pass",
        )
    if method in {"spider_E081", "spider_E081_full_rerun"}:
        return _as_bool(row, "E081_success_legobj_strict_proxy")
    return _as_bool(
        row,
        "E018b_generalization_pass",
        "E022_success",
        "E023_success",
        "E024_success",
        "E025_strict_success",
    )


def normalize_row(method: str, source_csv: Path, row: dict[str, Any]) -> dict[str, Any] | None:
    case = _case_from_row(row)
    if case not in CASES_13:
        return None
    variant = str(row.get("variant") or row.get("case") or case)
    obj_pos_cm = _as_float(row, "paper_spider_obj_pos_err_cm")
    if obj_pos_cm is None:
        pos_m = _as_float(row, "paper_object_Epos_case_m", "case_window_obj_err_mean_m", "post2_obj_err_mean_m")
        obj_pos_cm = None if pos_m is None else pos_m * 100.0
    contact5 = _as_float(row, "paper_omniretarget_contact_preservation_5cm_pct")
    contact_kind = "5cm_mask"
    if contact5 is None:
        contact5 = _as_float(row, "case_window_sim_contact_frames_pct", "post2_sim_contact_frames_pct")
        contact_kind = "legacy_contact_count"
    fall = _as_bool(row, "E018b_robot_fall_detected")
    if fall is None:
        pelvis = _as_float(row, "case_window_pelvis_z_min_m", "full_pelvis_z_min_m", "post2_pelvis_z_min_m")
        fall = bool(pelvis is not None and pelvis < 0.45)
    strict = _strict_success_for_method(method, row)
    out = {
        "method": method,
        "case": case,
        "variant": variant,
        "source_experiment": _source_exp(variant),
        "source_csv": str(source_csv.relative_to(REPO)) if source_csv.is_relative_to(REPO) else str(source_csv),
        "schema": "paper_metrics" if row.get("paper_metrics_version") else ("legacy_E081" if method == "spider_E081" else "unknown"),
        "role": row.get("role") or row.get("E028_role") or row.get("E029_role") or row.get("E030_role"),
        "obj_pos_cm": obj_pos_cm,
        "obj_ori_deg": _as_float(row, "paper_spider_obj_ori_err_deg", "paper_object_Erot_case_deg"),
        "joint_err_deg": _as_float(row, "paper_spider_joint_err_deg"),
        "mpkpe_cm": _as_float(row, "paper_spider_pos_err_cm"),
        "contact_proxy_pct": contact5,
        "contact_proxy_kind": contact_kind,
        "contact28_pct": _as_float(row, "paper_omniretarget_contact_preservation_local_case_pct"),
        "deep_pen_pct": _as_float(row, "paper_omniretarget_robot_object_deep_penetration_duration_pct"),
        "max_pen_cm": _as_float(row, "paper_omniretarget_robot_object_max_penetration_cm"),
        "mj_pen_duration_pct": _as_float(row, "paper_omniretarget_mj_penetration_duration_pct"),
        "mj_pen_max_depth_cm": _as_float(row, "paper_omniretarget_mj_penetration_max_depth_cm"),
        "smoothness": _as_float(row, "paper_dynaretarget_smoothness"),
        "relative_smoothness": _as_float(row, "paper_dynaretarget_relative_smoothness"),
        "pelvis_min_m": _as_float(row, "case_window_pelvis_z_min_m", "full_pelvis_z_min_m", "post2_pelvis_z_min_m"),
        "fall": fall,
        "object_success": _as_bool(row, "paper_spider_object_success", "paper_dynaretarget_object_success"),
        "transport_success": _as_bool(row, "paper_transport_success", "E081_success_case_window"),
        "strict_success": strict,
        "experiment_success": _as_bool(
            row,
            "E028_strict_success",
            "E029_success",
            "E029_guard_pass",
            "E030_success",
            "E030_target_geometry_success",
            "E030_clean_guard_pass",
            "E030_shortcut_guard_pass",
        ),
        "ref_leg_interference_pct": _as_float(row, "full_ref_leg_box_interference_frames_pct"),
        "sim_leg_interference_pct": _as_float(row, "full_sim_leg_box_interference_frames_pct"),
        "case_ref_leg_interference_pct": _as_float(row, "case_window_ref_leg_box_interference_frames_pct"),
        "case_sim_leg_interference_pct": _as_float(row, "case_window_sim_leg_box_interference_frames_pct"),
    }
    for key in (
        "E028_strict_success",
        "E029_success",
        "E029_guard_pass",
        "E029_p1_strict_target",
        "E030_success",
        "E030_target_geometry_success",
        "E030_clean_guard_pass",
        "E030_shortcut_guard_pass",
        "E030_diagnostic_surface_signal",
    ):
        if key in row:
            out[key] = row.get(key)
    return out


def load_sources() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sources = {
        "omniretarget_kinematic": COLLAB_RESULTS / "holosoma_v2_kinematic/comparison.csv",
        "spider_E081": E081_RESULTS / "comparison.csv",
        "spider_E081_full_rerun": COLLAB_RESULTS / "E026_E081_full/comparison.csv",
        "spider_E018b": COLLAB_RESULTS / "E018b/comparison.csv",
        "spider_E022": COLLAB_RESULTS / "E022/comparison.csv",
        "spider_E023": COLLAB_RESULTS / "E023/comparison.csv",
        "spider_E024": COLLAB_RESULTS / "E024/comparison.csv",
        "spider_E025": COLLAB_RESULTS / "E025/comparison.csv",
        "spider_E028": COLLAB_RESULTS / "E028/comparison.csv",
        "spider_E029": COLLAB_RESULTS / "E029/comparison.csv",
        "spider_E030": COLLAB_RESULTS / "E030/comparison.csv",
    }
    rows: list[dict[str, Any]] = []
    coverage: dict[str, Any] = {}
    for method, path in sources.items():
        raw_rows = _read_csv(path)
        normalized = [out for row in raw_rows if (out := normalize_row(method, path, row)) is not None]
        rows.extend(normalized)
        present = sorted({row["case"] for row in normalized})
        coverage[method] = {
            "comparison_csv": str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path),
            "file_exists": path.is_file(),
            "raw_rows": len(raw_rows),
            "normalized_rows": len(normalized),
            "cases_present": present,
            "cases_missing_13": [case for case in CASES_13 if case not in present],
            "cases_missing_9": [case for case in CASES_9 if case not in present],
            "best_positive_pool": method in BEST_POSITIVE_METHODS,
            "diagnostic_only": method in DIAGNOSTIC_METHODS,
        }
    coverage["E027_quality_audit"] = {
        "csv": str(QUALITY_AUDIT.relative_to(REPO)),
        "file_exists": QUALITY_AUDIT.is_file(),
        "rows": len(_read_csv(QUALITY_AUDIT)),
    }
    coverage["known_missing_reasons"] = {
        "omniretarget_kinematic:desk021_p1": "Holosoma / OmniRetarget SOCP infeasible; E027 marks desk021_p1 as discard_from_success_denominator",
        "spider_E081:most_cases": "legacy E081 baseline ran only the original two cases; use spider_E081_full_rerun for 13case paper-metrics coverage",
        "E027:rollout_candidates": "E027 was an offline audit; full_variant_candidates.tsv is empty by design",
    }
    return rows, coverage


def _candidate_score(row: dict[str, Any]) -> float:
    strict = 1.0 if row.get("strict_success") is True else 0.0
    obj = 1.0 if row.get("object_success") is True else 0.0
    transport = 1.0 if row.get("transport_success") is True else 0.0
    no_fall = 1.0 if row.get("fall") is False else 0.0
    contact = float(row.get("contact_proxy_pct") or 0.0)
    deep = float(row.get("deep_pen_pct") if row.get("deep_pen_pct") is not None else 100.0)
    max_pen = float(row.get("max_pen_cm") if row.get("max_pen_cm") is not None else 20.0)
    obj_pos = float(row.get("obj_pos_cm") if row.get("obj_pos_cm") is not None else 100.0)
    return (
        strict * 1_000_000
        + obj * 100_000
        + transport * 50_000
        + no_fall * 20_000
        + min(contact, 100.0) * 100.0
        - deep * 80.0
        - max_pen * 200.0
        - obj_pos * 20.0
    )


def build_best_dynamic(rows: list[dict[str, Any]], quality_by_case: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = [row for row in rows if row["method"] in BEST_POSITIVE_METHODS]
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_case[row["case"]].append(row)
    best_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for case in CASES_13:
        opts = by_case.get(case, [])
        if not opts:
            continue
        score, best = sorted(((_candidate_score(row), row) for row in opts), key=lambda item: item[0], reverse=True)[0]
        quality = quality_by_case.get(case, {})
        out = dict(best)
        out["method"] = BEST_METHOD_NAME
        out["selection_score"] = score
        out.update(_quality_projection(quality))
        best_rows.append(out)
        selection_rows.append(
            {
                "case": case,
                "selected_method": best["method"],
                "selected_variant": best["variant"],
                "selected_score": score,
                "num_positive_candidates": len(opts),
                "candidate_variants": ";".join(row["variant"] for row in opts),
                "obj_pos_cm": best.get("obj_pos_cm"),
                "contact_proxy_pct": best.get("contact_proxy_pct"),
                "deep_pen_pct": best.get("deep_pen_pct"),
                "max_pen_cm": best.get("max_pen_cm"),
                "fall": best.get("fall"),
                "strict_success": best.get("strict_success"),
                **_quality_projection(quality),
            }
        )
    return best_rows, selection_rows


def _quality_projection(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "quality_label": row.get("quality_label"),
        "discard_from_p0": row.get("discard_from_p0"),
        "discard_from_success_denominator": row.get("discard_from_success_denominator"),
        "recommended_owner": row.get("recommended_owner"),
        "decision_rationale": row.get("decision_rationale"),
    }


def load_quality() -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    rows = _read_csv(QUALITY_AUDIT)
    by_case = {str(row["case"]): row for row in rows if row.get("case")}
    caveats: list[dict[str, Any]] = []
    for case in CASES_13:
        row = by_case.get(case, {"case": case})
        caveats.append(
            {
                "case": case,
                "in_P0_case_set": case in CASES_9,
                "in_P1_case_set": True,
                "quality_label": row.get("quality_label"),
                "discard_from_p0": row.get("discard_from_p0"),
                "discard_from_success_denominator": row.get("discard_from_success_denominator"),
                "num_failed_evidence_classes": row.get("num_failed_evidence_classes"),
                "primary_evidence": row.get("primary_evidence"),
                "secondary_evidence": row.get("secondary_evidence"),
                "counter_evidence": row.get("counter_evidence"),
                "decision_rationale": row.get("decision_rationale"),
                "recommended_owner": row.get("recommended_owner"),
                "visual_evidence_path": row.get("visual_evidence_path"),
            }
        )
    return by_case, caveats


def _reject_reason(row: dict[str, Any]) -> str:
    method = str(row["method"])
    reasons: list[str] = []
    contact = row.get("contact_proxy_pct")
    deep = row.get("deep_pen_pct")
    max_pen = row.get("max_pen_cm")
    if row.get("fall") is True:
        reasons.append("fall")
    if isinstance(contact, (int, float)) and contact < 50.0:
        reasons.append(f"low_or_collapsed_contact={contact:.2f}%")
    if isinstance(deep, (int, float)) and deep > 20.0:
        reasons.append(f"high_deep_pen={deep:.2f}%")
    if isinstance(max_pen, (int, float)) and max_pen > 5.0:
        reasons.append(f"max_pen_over_5cm={max_pen:.2f}cm")
    if method == "spider_E028":
        reasons.append("E028_hard_barrier_diagnostic_not_positive_pool")
    elif method == "spider_E029":
        case = row["case"]
        if case == "box025_p2":
            reasons.append("non_regression_guard_only_E018b_strict_baseline_preferred")
        elif case == "bucket001_p1":
            reasons.append("stability_only_contact_remains_zero")
        else:
            reasons.append("E029_stability_diagnostic_not_positive_pool")
    elif method == "spider_E030":
        reasons.append("E030_geometry_surface_negative_result_not_positive_pool")
    if not reasons:
        reasons.append("diagnostic_only_by_E031_rule")
    return "; ".join(reasons)


def rejected_diagnostics(rows: list[dict[str, Any]], quality_by_case: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if row["method"] not in DIAGNOSTIC_METHODS:
            continue
        quality = quality_by_case.get(row["case"], {})
        out.append(
            {
                "method": row["method"],
                "case": row["case"],
                "variant": row["variant"],
                "role": row.get("role"),
                "contact_proxy_pct": row.get("contact_proxy_pct"),
                "deep_pen_pct": row.get("deep_pen_pct"),
                "max_pen_cm": row.get("max_pen_cm"),
                "obj_pos_cm": row.get("obj_pos_cm"),
                "fall": row.get("fall"),
                "strict_success": row.get("strict_success"),
                "experiment_success": row.get("experiment_success"),
                "reject_reason": _reject_reason(row),
                **_quality_projection(quality),
            }
        )
    return out


def _strict_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if row.get("strict_success") is True)


def _fall_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if row.get("fall") is True)


def _method_rows(rows: list[dict[str, Any]], method: str, cases: list[str]) -> list[dict[str, Any]]:
    by_case: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["method"] == method and row["case"] in cases and row["case"] not in by_case:
            by_case[row["case"]] = row
    return list(by_case.values())


def aggregate(rows: list[dict[str, Any]], rejected: list[dict[str, Any]], caveats: list[dict[str, Any]]) -> dict[str, Any]:
    p0_best = _method_rows(rows, BEST_METHOD_NAME, CASES_9)
    p1_best = _method_rows(rows, BEST_METHOD_NAME, CASES_13)
    return {
        "best_method": BEST_METHOD_NAME,
        "p0_cases": CASES_9,
        "p1_cases": CASES_13,
        "p0_num_cases": len(CASES_9),
        "p1_num_cases": len(CASES_13),
        "p0_strict_success": _strict_count(p0_best),
        "p1_strict_success": _strict_count(p1_best),
        "p0_object_pos_mean_cm": _mean([row.get("obj_pos_cm") for row in p0_best]),
        "p1_object_pos_mean_cm": _mean([row.get("obj_pos_cm") for row in p1_best]),
        "p0_contact5_mean_pct": _mean([row.get("contact_proxy_pct") for row in p0_best]),
        "p1_contact5_mean_pct": _mean([row.get("contact_proxy_pct") for row in p1_best]),
        "p0_deep_pen_mean_pct": _mean([row.get("deep_pen_pct") for row in p0_best]),
        "p1_deep_pen_mean_pct": _mean([row.get("deep_pen_pct") for row in p1_best]),
        "p0_falls": _fall_count(p0_best),
        "p1_falls": _fall_count(p1_best),
        "num_rejected_diagnostic_candidates": len(rejected),
        "num_success_denominator_discards": sum(row.get("discard_from_success_denominator") is True for row in caveats),
        "success_denominator_discards": [
            row["case"] for row in caveats if row.get("discard_from_success_denominator") is True
        ],
        "num_retarget_questionable": sum(row.get("quality_label") == "retarget_questionable" for row in caveats),
        "retarget_questionable_cases": [
            row["case"] for row in caveats if row.get("quality_label") == "retarget_questionable"
        ],
    }


def summarize_markdown(title: str, cases: list[str], rows: list[dict[str, Any]], caveats: list[dict[str, Any]]) -> str:
    methods = [
        "omniretarget_kinematic",
        "spider_E081_full_rerun",
        "spider_E018b",
        BEST_METHOD_NAME,
        "spider_E028",
        "spider_E029",
        "spider_E030",
    ]
    caveat_by_case = {row["case"]: row for row in caveats}
    out = [f"# {title}", ""]
    out.append(f"Cases ({len(cases)}): " + ", ".join(cases))
    out.append("")
    out.append("| Method | Role | N | Missing | Obj Pos cm ↓ | Contact 5cm/proxy ↑ | Deep Pen % ↓ | Falls ↓ | Strict ↑ |")
    out.append("|---|---|---:|---|---:|---:|---:|---:|---:|")
    for method in methods:
        method_rows = _method_rows(rows, method, cases)
        present = sorted({row["case"] for row in method_rows})
        missing = [case for case in cases if case not in present]
        role = "best-positive" if method in {BEST_METHOD_NAME, "spider_E018b"} else ("diagnostic" if method in DIAGNOSTIC_METHODS else "baseline")
        out.append(
            "| "
            + " | ".join(
                [
                    method,
                    role,
                    f"{len(present)}/{len(cases)}",
                    ", ".join(missing) if missing else "-",
                    _fmt(_mean([row.get("obj_pos_cm") for row in method_rows])),
                    _fmt(_mean([row.get("contact_proxy_pct") for row in method_rows])),
                    _fmt(_mean([row.get("deep_pen_pct") for row in method_rows])),
                    str(_fall_count(method_rows)),
                    f"{_strict_count(method_rows)}/{len(present)}" if present else "0/0",
                ]
            )
            + " |"
        )
    out.append("")
    out.append("## Data Caveats")
    out.append("")
    out.append("| Case | Quality | P0 | Success-denominator discard | Rationale |")
    out.append("|---|---|---:|---:|---|")
    for case in cases:
        row = caveat_by_case.get(case, {"case": case})
        out.append(
            f"| {case} | {row.get('quality_label') or '-'} | {case in CASES_9} | "
            f"{row.get('discard_from_success_denominator') is True} | {row.get('decision_rationale') or '-'} |"
        )
    out.append("")
    out.append("Notes:")
    out.append("- E028-E030 rows are shown as diagnostic baselines only; they are not eligible for `best-positive` selection.")
    out.append("- `desk021_p1` remains in P1 caveats but is the only success-denominator discard from E027.")
    out.append("- If the conservative best strict count remains unchanged, E031 is a ledger result, not a new optimization gain.")
    return "\n".join(out) + "\n"


def data_quality_markdown(caveats: list[dict[str, Any]]) -> str:
    out = ["# E031 Data Quality Caveats", ""]
    out.append("| Case | Quality label | P0 | Discard from success denominator | Evidence | Counter-evidence |")
    out.append("|---|---|---:|---:|---|---|")
    for row in caveats:
        out.append(
            f"| {row['case']} | {row.get('quality_label') or '-'} | {row.get('in_P0_case_set')} | "
            f"{row.get('discard_from_success_denominator') is True} | {row.get('primary_evidence') or '-'}; "
            f"{row.get('secondary_evidence') or '-'} | {row.get('counter_evidence') or '-'} |"
        )
    out.append("")
    out.append("Only `desk021_p1` satisfies the multi-evidence discard protocol. E028-E030 do not introduce new discarded cases.")
    return "\n".join(out) + "\n"


def visual_metric_audit(best_selection: list[dict[str, Any]], rejected: list[dict[str, Any]]) -> str:
    out = ["# E031 Visual / Metric Audit", ""]
    out.append("E031 reuses existing videos/keyframes from E026-E030 and records why diagnostic variants are not positive candidates.")
    out.append("")
    out.append("## Conservative Best Selection")
    out.append("")
    out.append("| Case | Selected variant | Signal | Visual evidence path |")
    out.append("|---|---|---|---|")
    for row in best_selection:
        variant = str(row["selected_variant"])
        exp = _source_exp(variant)
        evidence = f"workspace/core4d_collab_retarget/results/{exp}/online_video/{variant}.mp4"
        signal = f"contact={_fmt(row.get('contact_proxy_pct'))}%, deep={_fmt(row.get('deep_pen_pct'))}%, fall={row.get('fall')}, strict={row.get('strict_success')}"
        out.append(f"| {row['case']} | `{variant}` | {signal} | `{evidence}` |")
    out.append("")
    out.append("## Rejected Diagnostics")
    out.append("")
    out.append("| Method | Case | Variant | Reject reason | Visual/keyframe root |")
    out.append("|---|---|---|---|---|")
    for row in rejected:
        exp = row["method"].replace("spider_", "")
        root = f"workspace/core4d_collab_retarget/results/{exp}/keyframes/{row['variant']}/"
        out.append(f"| {row['method']} | {row['case']} | `{row['variant']}` | {row['reject_reason']} | `{root}` |")
    out.append("")
    out.append("Guard rule: high-contact high-penetration rows and contact-collapse rows are diagnostic negatives, even when object tracking remains good.")
    return "\n".join(out) + "\n"


def write_index() -> None:
    text = """# E031 Full Eval Outputs

- `coverage.json`
- `method_case_metrics.csv`
- `best_dynamic_selection.csv`
- `rejected_diagnostic_candidates.csv`
- `data_quality_caveats.csv` / `data_quality_caveats.md`
- `summary_9case.md`
- `summary_13case.md`
- `visual_metric_audit.md`
- `aggregate_summary.json`

E031 is an offline ledger assembly. It does not run new CEM rollouts and does not use the remote GPU wrapper.
"""
    (RESULTS / "INDEX.md").write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="run the full E031 offline assembly")
    args = parser.parse_args()
    if not args.all:
        parser.print_help()
        return 2

    RESULTS.mkdir(parents=True, exist_ok=True)
    quality_by_case, caveats = load_quality()
    source_rows, coverage = load_sources()
    best_rows, best_selection = build_best_dynamic(source_rows, quality_by_case)
    all_rows: list[dict[str, Any]] = []
    for row in source_rows:
        out = dict(row)
        out.update(_quality_projection(quality_by_case.get(row["case"], {})))
        all_rows.append(out)
    all_rows.extend(best_rows)
    rejected = rejected_diagnostics(source_rows, quality_by_case)
    agg = aggregate(all_rows, rejected, caveats)

    _write_csv(RESULTS / "method_case_metrics.csv", all_rows)
    _write_csv(RESULTS / "best_dynamic_selection.csv", best_selection)
    _write_csv(RESULTS / "rejected_diagnostic_candidates.csv", rejected)
    _write_csv(RESULTS / "data_quality_caveats.csv", caveats)
    _write_json(RESULTS / "coverage.json", coverage)
    _write_json(RESULTS / "aggregate_summary.json", agg)
    (RESULTS / "summary_9case.md").write_text(
        summarize_markdown("E031 P0 9case Conservative Summary", CASES_9, all_rows, caveats),
        encoding="utf-8",
    )
    (RESULTS / "summary_13case.md").write_text(
        summarize_markdown("E031 P1 13case Conservative Summary", CASES_13, all_rows, caveats),
        encoding="utf-8",
    )
    (RESULTS / "data_quality_caveats.md").write_text(data_quality_markdown(caveats), encoding="utf-8")
    (RESULTS / "visual_metric_audit.md").write_text(visual_metric_audit(best_selection, rejected), encoding="utf-8")
    write_index()
    print(json.dumps({"out": str(RESULTS), "rows": len(all_rows), "best_rows": len(best_rows), **agg}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
