#!/usr/bin/env python3
"""Evaluate the E169 2^3 lower-body/object factorial experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval.core.core_metrics import (  # noqa: E402
    EVAL_METRIC_STANDARD_ID,
    METRIC_FIELDS,
    EvalConfig,
    evaluate_sequence,
    npz_qpos,
)
from eval.core.motion_health import HEALTH_AGGS, METRIC_KEYS, run_health  # noqa: E402
from eval_E168_e167a_metrics import (  # noqa: E402
    fixed_reference_z_metrics,
    release_window_info,
    source_person_idx,
)


REPO = Path(__file__).resolve().parents[5]
DEFAULT_MANIFEST = REPO / "workspace/core4d/results/E169/manifests/factorial_analysis_manifest.tsv"
DEFAULT_OUT_DIR = REPO / "workspace/core4d/results/E169/eval/full"
DEFAULT_REVIEW = DEFAULT_OUT_DIR / "manual_review_template.tsv"
E168_MANIFEST = (
    REPO
    / "workspace/core4d/results/E168/s6_downstream/cem/manifests/cem_production_manifest.tsv"
)

CONTACT_MIN = 0.50
BODY_Z_P95_MAX_M = 0.20
PENETRATION_3MM_MAX = 0.30
LOWER_BODY_MAX = 0.10
RELEASE_FALSE_MAX = 0.30
LEG_GATE_FALLBACK_MAX = 0.10
LEG_GATE_VALID_LAST_MIN = 0.05

FACTOR_ORDER = ("P", "R", "G", "PR", "PG", "RG", "PRG")
FACTOR_BITS = {"P": "p_enabled", "R": "r_enabled", "G": "g_enabled"}
E168_SUMMARY_METRICS = (
    "track_pelvis_z_err_terminal_m",
    "body_z_err_p95_m",
    "body_z_err_peak_m",
    "sugar_3d_err_peak_m",
    "track_joint_err_deg_mean",
    "track_eef_pos_err_cm_mean",
    "track_eef_ori_err_deg_mean",
    "track_obj_pos_err_cm_mean",
    "track_obj_ori_err_deg_mean",
    "track_root_pos_err_cm_mean",
    "track_root_ori_err_deg_mean",
    "hand_object_physics_contact_in_mask_frac",
    "hand_object_physics_contact_3mm_in_mask_frac",
    "hand_object_release_false_contact_3mm_frac",
    "hand_object_physics_penetration_3mm_frame_frac",
    "hand_geom_penetration_2mm_frac",
    "hand_geom_penetration_5mm_frac",
    "leg_penetration_frac",
    "body_penetration_frac",
    "qpos_jerk_l2_p95",
    "trackbody_jerk_p95",
    "ankle_jerk_p95",
    "foot_slip_max_m",
)
CONTRAST_METRICS = {
    "track_pelvis_z_err_terminal_m": "lower",
    "body_z_err_p95_m": "lower",
    "body_z_err_peak_m": "lower",
    "sugar_3d_err_peak_m": "lower",
    "track_joint_err_deg_mean": "lower",
    "track_eef_pos_err_cm_mean": "lower",
    "track_eef_ori_err_deg_mean": "lower",
    "track_obj_pos_err_cm_mean": "lower",
    "track_obj_ori_err_deg_mean": "lower",
    "track_root_pos_err_cm_mean": "lower",
    "track_root_ori_err_deg_mean": "lower",
    "hand_object_physics_contact_in_mask_frac": "higher",
    "hand_object_physics_contact_3mm_in_mask_frac": "higher",
    "hand_object_release_false_contact_3mm_frac": "lower",
    "hand_object_physics_penetration_3mm_frame_frac": "lower",
    "hand_geom_penetration_2mm_frac": "lower",
    "hand_geom_penetration_5mm_frac": "lower",
    "leg_penetration_frac": "lower",
    "leg_object_physics_contact_frac": "lower",
    "body_penetration_frac": "lower",
    "qpos_jerk_l2_p95": "lower",
    "trackbody_jerk_p95": "lower",
    "ankle_jerk_p95": "lower",
    "foot_slip_max_m": "lower",
}
RANKING_METRICS = {
    "track_pelvis_z_err_terminal_m": "high",
    "body_z_err_p95_m": "high",
    "body_z_err_peak_m": "high",
    "sugar_3d_err_peak_m": "high",
    "track_joint_err_deg_mean": "high",
    "track_eef_pos_err_cm_mean": "high",
    "track_obj_pos_err_cm_mean": "high",
    "hand_object_physics_contact_in_mask_frac": "low",
    "hand_object_release_false_contact_3mm_frac": "high",
    "hand_object_physics_penetration_3mm_frame_frac": "high",
    "leg_penetration_frac": "high",
    "qpos_jerk_l2_p95": "high",
}
LEG_GATE_KEYS = (
    "cem_leg_gate_valid_frac",
    "cem_leg_gate_selected_valid_frac",
    "cem_leg_gate_fallback_used",
    "cem_leg_gate_min_sdf_min_m",
    "cem_leg_gate_min_sdf_p05_m",
    "cem_leg_gate_violation_pct_mean",
    "cem_leg_gate_selected_all_valid",
)


def repo_path(raw: str | Path) -> Path:
    path = Path(raw)
    if path.exists():
        return path.resolve()
    text = str(raw)
    for marker in ("example_datasets/", "workspace/", "logs/"):
        if marker in text:
            return REPO / (marker + text.split(marker, 1)[1])
    return path if path.is_absolute() else REPO / path


def rel(path: str | Path) -> str:
    path = Path(path)
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        pass
    results_target = (REPO / "workspace/core4d/results").resolve()
    try:
        return str(Path("workspace/core4d/results") / path.resolve().relative_to(results_target))
    except ValueError:
        return str(path)


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serial(row.get(key, "")) for key in fields})


def serial(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def finite(value: Any, default: float = math.nan) -> float:
    try:
        value = float(value)
    except Exception:
        return default
    return value if math.isfinite(value) else default


def truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def mean(values: list[Any]) -> float:
    valid = [finite(value) for value in values]
    valid = [value for value in valid if math.isfinite(value)]
    return statistics.fmean(valid) if valid else math.nan


def percentile(values: list[Any], q: float) -> float:
    valid = np.asarray([finite(value) for value in values], dtype=np.float64)
    valid = valid[np.isfinite(valid)]
    return float(np.percentile(valid, q)) if valid.size else math.nan


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_stat(data: np.lib.npyio.NpzFile, key: str, stat: str) -> float:
    if key not in data.files:
        return math.nan
    arr = np.asarray(data[key], dtype=np.float64)
    if stat == "last":
        if arr.ndim != 2 or arr.shape[1] == 0:
            return math.nan
        arr = arr[:, -1]
    arr = arr[np.isfinite(arr)]
    if not arr.size:
        return math.nan
    if stat == "min":
        return float(arr.min())
    return float(arr.mean())


def gate_health(path: Path, enabled: bool) -> dict[str, Any]:
    out: dict[str, Any] = {"leg_gate_enabled": enabled}
    with np.load(path, allow_pickle=True) as data:
        for key in LEG_GATE_KEYS:
            out[f"{key}_mean"] = array_stat(data, key, "mean")
            out[f"{key}_last_iter_mean"] = array_stat(data, key, "last")
        out["cem_leg_gate_min_sdf_worst_m"] = array_stat(
            data, "cem_leg_gate_min_sdf_min_m", "min"
        )
    if not enabled:
        out["leg_gate_health_applicable"] = False
        out["leg_gate_health_pass"] = None
        return out
    fallback = finite(out["cem_leg_gate_fallback_used_mean"], math.inf)
    valid_last = finite(out["cem_leg_gate_valid_frac_last_iter_mean"], -math.inf)
    selected_all = finite(out["cem_leg_gate_selected_all_valid_mean"], -math.inf)
    out["leg_gate_health_applicable"] = True
    out["leg_gate_fallback_pass"] = fallback <= LEG_GATE_FALLBACK_MAX
    out["leg_gate_valid_frac_pass"] = valid_last >= LEG_GATE_VALID_LAST_MIN
    out["leg_gate_selected_valid_pass"] = selected_all >= 1.0 - 1e-9
    out["leg_gate_health_pass"] = bool(
        out["leg_gate_fallback_pass"]
        and out["leg_gate_valid_frac_pass"]
        and out["leg_gate_selected_valid_pass"]
    )
    return out


def artifact_status(row: dict[str, str]) -> dict[str, Any]:
    required = {
        "root_npz": repo_path(row["result_npz"]),
        "outdir_npz": repo_path(row["outdir_npz"]),
        "config_act": repo_path(row["config_act"]),
        "scene_act": repo_path(row["scene_act"]),
        "trajectory": repo_path(row["trajectory"]),
        "contact_mask": repo_path(row["contact_mask"]),
    }
    out = {f"{key}_exists": value.is_file() for key, value in required.items()}
    out["video_exists"] = repo_path(row["video"]).is_file()
    out["artifact_gate_pass"] = all(out[f"{key}_exists"] for key in required)
    out["config_gate_pass"] = row.get("config_audit_status") == "pass" and out["config_act_exists"]
    return out


def review_fields(variant: str, reviews: dict[str, dict[str, str]]) -> dict[str, Any]:
    row = reviews.get(variant)
    if row is None:
        return {
            "manual_review_status": "pending",
            "manual_use_decision": "PENDING",
            "manual_quality_label": "",
            "manual_failure_taxonomy": "",
            "manual_review_note": "",
            "manual_reviewer": "",
            "manual_reviewed_at": "",
            "manual_review_source_workbook": "",
            "visual_gate_pass": None,
            "visual_review_status": "PENDING_MANUAL_REVIEW",
        }
    status = row.get("manual_review_status", "reviewed")
    decision = row.get("manual_use_decision", "")
    return {
        "manual_review_status": status,
        "manual_use_decision": decision,
        "manual_quality_label": row.get("manual_quality_label", ""),
        "manual_failure_taxonomy": row.get("manual_failure_taxonomy", ""),
        "manual_review_note": row.get("manual_review_note", ""),
        "manual_reviewer": row.get("reviewer", row.get("manual_reviewer", "")),
        "manual_reviewed_at": row.get("reviewed_at", row.get("manual_reviewed_at", "")),
        "manual_review_source_workbook": row.get("manual_review_source_workbook", "manual_review_template.tsv"),
        "visual_gate_pass": decision == "USE" if status == "reviewed" else None,
        "visual_review_status": (
            f"REVIEWED_{decision}" if status == "reviewed" else "PENDING_MANUAL_REVIEW"
        ),
    }


def apply_gates(item: dict[str, Any]) -> None:
    release_applicable = bool(item["release_gate_applicable"])
    item["tracking_gate_pass"] = bool(
        not item.get("fall_flag")
        and finite(item.get("track_pelvis_z_err_terminal_m"), math.inf)
        <= EvalConfig().track_pelvis_terminal_th_m
    )
    item["holosoma_z_gate_pass"] = (
        finite(item.get("body_z_err_p95_m"), math.inf) <= BODY_Z_P95_MAX_M
    )
    item["contact_gate_pass"] = (
        finite(item.get("hand_object_physics_contact_in_mask_frac"), -math.inf)
        >= CONTACT_MIN
    )
    item["penetration_gate_pass"] = (
        finite(item.get("hand_object_physics_penetration_3mm_frame_frac"), math.inf)
        <= PENETRATION_3MM_MAX
    )
    item["lower_body_gate_pass"] = (
        finite(item.get("leg_penetration_frac"), math.inf) <= LOWER_BODY_MAX
    )
    if release_applicable:
        item["release_gate_pass"] = (
            finite(item.get("hand_object_release_false_contact_3mm_frac"), math.inf)
            <= RELEASE_FALSE_MAX
        )
        item["release_gate_status"] = (
            "APPLICABLE_PASS" if item["release_gate_pass"] else "APPLICABLE_FAIL"
        )
    else:
        item["release_gate_pass"] = None
    failures = []
    for gate, label in (
        ("artifact_gate_pass", "artifact"),
        ("config_gate_pass", "config"),
        ("tracking_gate_pass", "tracking"),
        ("holosoma_z_gate_pass", "holosoma_z"),
        ("contact_gate_pass", "contact"),
        ("penetration_gate_pass", "hand_penetration"),
        ("lower_body_gate_pass", "lower_body"),
    ):
        if not bool(item.get(gate)):
            failures.append(label)
    if release_applicable and not bool(item.get("release_gate_pass")):
        failures.append("release_false_contact")
    item["numeric_release_pass"] = not failures
    if item.get("leg_gate_health_applicable") and not item.get("leg_gate_health_pass"):
        failures.append("leg_gate_health")
    item["e169_acceptance_pass"] = not failures
    item["failure_modes"] = ",".join(failures)
    item["release_status"] = "PASS" if not failures else "FAIL_" + "+".join(failures).upper()


def evaluate_row(
    row: dict[str, str], cfg: EvalConfig, reviews: dict[str, dict[str, str]]
) -> dict[str, Any]:
    qpos_path = repo_path(row["outdir_npz"])
    root_npz = repo_path(row["result_npz"])
    scene = repo_path(row["scene_act"])
    trajectory = repo_path(row["trajectory"])
    mask = repo_path(row["contact_mask"])
    item = evaluate_sequence(
        row=row,
        method=row["spider_method_id"],
        hand_collision_variant_id=row["hand_collision_variant_id"],
        qpos_path=qpos_path,
        scene_xml=scene,
        config=cfg,
        kin_ref_path=trajectory,
        contact_mask_path=mask,
        person_idx=source_person_idx(row),
    )
    sim_qpos, _ = npz_qpos(qpos_path)
    item.update(run_health(qpos_path, scene, cfg))
    item.update(fixed_reference_z_metrics(qpos_path, scene, trajectory))
    item.update(artifact_status(row))
    item.update(gate_health(root_npz, truth(row["g_enabled"])))
    item.update(release_window_info(mask, source_person_idx(row), len(sim_qpos)))
    for key in (
        "ordinal", "variant", "case_id", "sequence_key", "source_person",
        "object_key", "source_action", "source_obstacle_level", "preferred_pool",
        "analysis_role", "cell_id", "p_enabled", "r_enabled",
        "g_enabled", "reused_baseline", "retarget_variant_id", "target_variant_id",
        "hand_collision_variant_id", "spider_method_id", "assigned_gpu", "gpu_id",
        "status",
    ):
        item[key] = row.get(key, "")
    item.update(
        {
            "metric_standard_id": EVAL_METRIC_STANDARD_ID,
            "result_npz": rel(root_npz),
            "outdir_npz": rel(qpos_path),
            "config_act": rel(row["config_act"]),
            "video": rel(row["video"]) if repo_path(row["video"]).is_file() else rel(row["video"]),
            "trajectory": rel(trajectory),
            "contact_mask": rel(mask),
            "scene_xml": rel(scene),
        }
    )
    item.update(review_fields(row["variant"], reviews))
    item["manifest_status"] = row.get("status", "")
    item["relative_gate_status"] = (
        "BASELINE_REFERENCE" if row.get("cell_id") == "B0" else "B0_DELTA_AVAILABLE"
    )
    apply_gates(item)
    return item


def add_baseline_deltas(rows: list[dict[str, Any]]) -> None:
    baselines = {row["case_id"]: row for row in rows if row["cell_id"] == "B0"}
    for row in rows:
        baseline = baselines.get(row["case_id"])
        for metric, direction in CONTRAST_METRICS.items():
            delta = (
                finite(row.get(metric)) - finite(baseline.get(metric))
                if baseline is not None
                and math.isfinite(finite(row.get(metric)))
                and math.isfinite(finite(baseline.get(metric)))
                else math.nan
            )
            row[f"delta_b0_{metric}"] = delta
            row[f"improvement_b0_{metric}"] = delta if direction == "higher" else -delta


def contrast_sign(row: dict[str, Any], term: str) -> int:
    sign = 1
    for factor in term:
        sign *= 1 if truth(row[FACTOR_BITS[factor]]) else -1
    return sign


def factorial_contrasts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cases = sorted({row["case_id"] for row in rows})
    for case_id in [*cases, "OVERALL_CASE_MEAN"]:
        case_rows = rows if case_id == "OVERALL_CASE_MEAN" else [row for row in rows if row["case_id"] == case_id]
        if case_id != "OVERALL_CASE_MEAN" and len(case_rows) != 8:
            continue
        for metric, direction in CONTRAST_METRICS.items():
            for term in FACTOR_ORDER:
                if case_id == "OVERALL_CASE_MEAN":
                    case_effects = []
                    for actual_case in cases:
                        subset = [row for row in rows if row["case_id"] == actual_case]
                        values = [finite(row.get(metric)) for row in subset]
                        if len(subset) == 8 and all(math.isfinite(value) for value in values):
                            case_effects.append(
                                2.0 * mean([contrast_sign(row, term) * finite(row[metric]) for row in subset])
                            )
                    effect = mean(case_effects)
                    n = len(case_effects)
                else:
                    values = [finite(row.get(metric)) for row in case_rows]
                    effect = (
                        2.0 * mean([contrast_sign(row, term) * finite(row[metric]) for row in case_rows])
                        if all(math.isfinite(value) for value in values)
                        else math.nan
                    )
                    n = len(case_rows)
                out.append(
                    {
                        "case_id": case_id,
                        "metric": metric,
                        "direction": direction,
                        "term": term,
                        "raw_effect_on_minus_off": effect,
                        "improvement_effect": effect if direction == "higher" else -effect,
                        "n": n,
                    }
                )
    return out


def cell_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for cell in ("B0", "P", "R", "G", "PR", "PG", "RG", "PRG"):
        subset = [row for row in rows if row["cell_id"] == cell]
        item: dict[str, Any] = {
            "cell_id": cell,
            "n": len(subset),
            "failure_case_pass": sum(row["e169_acceptance_pass"] for row in subset if row["analysis_role"] == "failure"),
            "failure_case_total": sum(row["analysis_role"] == "failure" for row in subset),
            "control_pass": sum(row["e169_acceptance_pass"] for row in subset if row["analysis_role"] == "control"),
            "control_total": sum(row["analysis_role"] == "control" for row in subset),
            "all_acceptance_pass": all(row["e169_acceptance_pass"] for row in subset) if subset else False,
            "failed_cases": ",".join(row["case_id"] for row in subset if not row["e169_acceptance_pass"]),
            "numeric_pass": sum(bool(row.get("numeric_release_pass")) for row in subset),
            "tracking_pass": sum(bool(row.get("tracking_gate_pass")) for row in subset),
            "z_pass": sum(bool(row.get("holosoma_z_gate_pass")) for row in subset),
            "contact_pass": sum(bool(row.get("contact_gate_pass")) for row in subset),
            "release_pass": sum(bool(row.get("release_gate_pass")) for row in subset),
            "release_applicable": sum(bool(row.get("release_gate_applicable")) for row in subset),
            "release_not_applicable": sum(not bool(row.get("release_gate_applicable")) for row in subset),
            "penetration_pass": sum(bool(row.get("penetration_gate_pass")) for row in subset),
            "lower_body_pass": sum(bool(row.get("lower_body_gate_pass")) for row in subset),
            "leg_gate_health_pass": sum(bool(row.get("leg_gate_health_pass")) for row in subset),
            "leg_gate_health_applicable": sum(bool(row.get("leg_gate_health_applicable")) for row in subset),
            "manual_reviewed": sum(row.get("manual_review_status") == "reviewed" for row in subset),
            "manual_use": sum(row.get("manual_use_decision") == "USE" for row in subset),
            "manual_do_not_use": sum(row.get("manual_use_decision") == "DO_NOT_USE" for row in subset),
            "fall_count": sum(bool(row.get("fall_flag")) for row in subset),
        }
        for metric in CONTRAST_METRICS:
            item[f"{metric}_mean"] = mean([row.get(metric) for row in subset])
        out.append(item)
    return out


def metric_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = [("overall", "all", rows)]
    groups.extend(
        ("cell_id", cell, [row for row in rows if row["cell_id"] == cell])
        for cell in ("B0", "P", "R", "G", "PR", "PG", "RG", "PRG")
    )
    out = []
    for group_type, group_value, subset in groups:
        for metric, direction in CONTRAST_METRICS.items():
            valid = [finite(row.get(metric)) for row in subset]
            valid = [value for value in valid if math.isfinite(value)]
            out.append(
                {
                    "group_type": group_type,
                    "group_value": group_value,
                    "metric": metric,
                    "direction": direction,
                    "n": len(valid),
                    "mean": mean(valid),
                    "median": percentile(valid, 50),
                    "p95": percentile(valid, 95),
                    "min": min(valid) if valid else math.nan,
                    "max": max(valid) if valid else math.nan,
                }
            )
    return out


def metric_rankings(rows: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
    out = []
    for metric, direction in RANKING_METRICS.items():
        valid = [row for row in rows if math.isfinite(finite(row.get(metric)))]
        valid.sort(
            key=lambda row: finite(row.get(metric)),
            reverse=direction == "high",
        )
        for rank, row in enumerate(valid[:top_k], start=1):
            out.append(
                {
                    "metric": metric,
                    "worse_direction": direction,
                    "rank": rank,
                    "variant": row["variant"],
                    "case_id": row["case_id"],
                    "cell_id": row["cell_id"],
                    "value": row.get(metric),
                    "release_status": row.get("release_status"),
                    "failure_modes": row.get("failure_modes"),
                    "manual_use_decision": row.get("manual_use_decision"),
                    "video": row.get("video"),
                }
            )
    return out


def key_metric_rows(rows: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    fields = [
        "variant", "case_id", "analysis_role", "cell_id", "p_enabled", "r_enabled",
        "g_enabled", "reused_baseline", "release_status", "failure_modes",
        "e169_acceptance_pass", "numeric_release_pass", "tracking_gate_pass",
        "holosoma_z_gate_pass", "contact_gate_pass", "release_gate_pass",
        "penetration_gate_pass", "lower_body_gate_pass", "leg_gate_health_pass",
        "manual_review_status", "manual_use_decision", "manual_quality_label",
        *E168_SUMMARY_METRICS,
        "leg_object_physics_contact_frac",
        "cem_leg_gate_valid_frac_last_iter_mean",
        "cem_leg_gate_selected_valid_frac_mean",
        "cem_leg_gate_fallback_used_mean",
        "video",
    ]
    return fields, [{field: row.get(field, "") for field in fields} for row in rows]


def review_template(
    rows: list[dict[str, Any]], reviews: dict[str, dict[str, str]]
) -> list[dict[str, Any]]:
    return [
        {
            "variant": row["variant"],
            "case_id": row["case_id"],
            "cell_id": row["cell_id"],
            "manual_review_status": reviews.get(row["variant"], {}).get("manual_review_status", "pending"),
            "manual_use_decision": reviews.get(row["variant"], {}).get("manual_use_decision", "PENDING"),
            "manual_quality_label": reviews.get(row["variant"], {}).get("manual_quality_label", ""),
            "manual_failure_taxonomy": reviews.get(row["variant"], {}).get("manual_failure_taxonomy", ""),
            "manual_review_note": reviews.get(row["variant"], {}).get("manual_review_note", ""),
            "reviewer": reviews.get(row["variant"], {}).get("reviewer", ""),
            "reviewed_at": reviews.get(row["variant"], {}).get("reviewed_at", ""),
            "video": row["video"],
        }
        for row in rows
        if row["cell_id"] != "B0"
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("available", "full"), default="available")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--manual-review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    manifest = repo_path(args.manifest)
    out_dir = repo_path(args.out_dir)
    manifest_rows = read_tsv(manifest)
    e168_context = {
        row["case_id"]: row for row in read_tsv(E168_MANIFEST) if row.get("case_id")
    }
    for row in manifest_rows:
        source = e168_context.get(row.get("case_id", ""), {})
        for key in ("source_action", "source_obstacle_level"):
            row[key] = source.get(key, row.get(key, ""))
    review_rows = read_tsv(repo_path(args.manual_review))
    reviews = {row["variant"]: row for row in review_rows}
    required_keys = ("result_npz", "outdir_npz", "config_act", "scene_act", "trajectory", "contact_mask")
    ready, not_ready = [], []
    for row in manifest_rows:
        missing = [key for key in required_keys if not repo_path(row[key]).is_file()]
        if missing:
            not_ready.append({**row, "missing_artifacts": ",".join(missing)})
        else:
            ready.append(row)
    cfg = EvalConfig()
    metrics: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for index, row in enumerate(ready, start=1):
        print(f"[{index}/{len(ready)}] evaluate {row['variant']}", flush=True)
        try:
            metrics.append(evaluate_row(row, cfg, reviews))
        except Exception as exc:
            errors.append({"variant": row["variant"], "case_id": row["case_id"], "error": f"{type(exc).__name__}: {exc}"})
    add_baseline_deltas(metrics)
    contrasts = factorial_contrasts(metrics)
    cells = cell_summary(metrics)
    summaries = metric_summary(metrics)
    rankings = metric_rankings(metrics)
    key_fields, key_rows = key_metric_rows(metrics)
    failures = Counter()
    for row in metrics:
        failures.update(filter(None, row["failure_modes"].split(",")))
    counts = {
        "manifest_rows": len(manifest_rows),
        "evaluated": len(metrics),
        "not_ready": len(not_ready),
        "evaluation_errors": len(errors),
        "numeric_pass": sum(row["numeric_release_pass"] for row in metrics),
        "acceptance_pass": sum(row["e169_acceptance_pass"] for row in metrics),
        "new_full_evaluated": sum(not truth(row["reused_baseline"]) for row in metrics),
        "reused_baselines": sum(truth(row["reused_baseline"]) for row in metrics),
        "g_rows": sum(truth(row["g_enabled"]) for row in metrics),
        "g_health_pass": sum(bool(row.get("leg_gate_health_pass")) for row in metrics),
    }
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "stage": args.stage,
        "metric_standard_id": EVAL_METRIC_STANDARD_ID,
        "manifest": rel(manifest),
        "manifest_sha256": sha256(manifest),
        "counts": counts,
        "failure_mode_counts": dict(failures),
        "thresholds": {
            "track_pelvis_terminal_m": cfg.track_pelvis_terminal_th_m,
            "body_z_err_p95_m": BODY_Z_P95_MAX_M,
            "raw_contact_in_mask_min": CONTACT_MIN,
            "release_false_contact_3mm_max": RELEASE_FALSE_MAX,
            "hand_penetration_3mm_max": PENETRATION_3MM_MAX,
            "leg_penetration_frac_max": LOWER_BODY_MAX,
            "leg_gate_fallback_mean_max": LEG_GATE_FALLBACK_MAX,
            "leg_gate_valid_frac_last_iter_min": LEG_GATE_VALID_LAST_MIN,
            "leg_gate_selected_all_valid_mean": 1.0,
        },
    }
    metric_fields: list[str] = []
    priority = [
        "variant", "case_id", "analysis_role", "cell_id", "p_enabled", "r_enabled", "g_enabled",
        "reused_baseline", "release_status", "failure_modes", "e169_acceptance_pass",
        "numeric_release_pass", "leg_gate_health_pass", "manual_review_status",
        "manual_use_decision", "manual_quality_label", "leg_penetration_frac",
        "leg_object_physics_contact_frac", "hand_object_physics_contact_in_mask_frac",
        "hand_object_physics_penetration_3mm_frame_frac", "body_z_err_p95_m",
    ]
    for field in [*priority, *METRIC_FIELDS, *METRIC_KEYS, *HEALTH_AGGS.keys(), *LEG_GATE_KEYS]:
        if field not in metric_fields:
            metric_fields.append(field)
    for row in metrics:
        for field in row:
            if field not in metric_fields:
                metric_fields.append(field)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(out_dir / "e169_case_metrics.tsv", metrics, metric_fields)
    write_tsv(out_dir / "e169_cell_summary.tsv", cells)
    write_tsv(out_dir / "e169_metric_summary.tsv", summaries)
    write_tsv(out_dir / "e169_worst_case_rankings.tsv", rankings)
    write_tsv(out_dir / "e169_e168_key_metrics.tsv", key_rows, key_fields)
    write_tsv(out_dir / "e169_factorial_contrasts.tsv", contrasts)
    write_tsv(out_dir / "e169_not_ready.tsv", not_ready)
    write_tsv(out_dir / "e169_evaluation_errors.tsv", errors)
    write_tsv(out_dir / "evaluated_manifest_snapshot.tsv", ready)
    template_path = out_dir / "manual_review_template.tsv"
    write_tsv(template_path, review_template(metrics, reviews))
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"E169 eval: evaluated={counts['evaluated']}/{counts['manifest_rows']} "
        f"new={counts['new_full_evaluated']}/28 not_ready={counts['not_ready']} errors={len(errors)}"
    )
    if errors:
        return 1
    if args.require_all and not_ready:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
