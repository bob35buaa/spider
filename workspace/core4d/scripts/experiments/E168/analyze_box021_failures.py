#!/usr/bin/env python3
"""Analyze E168 Box021 manual failures against the available metric bank."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[5]
RUN_ROOT = REPO / "workspace/core4d/results/E168"
METRICS = RUN_ROOT / "s6_downstream/cem/eval/box021_all28_reviewed/e168_case_metrics.tsv"
REVIEW = RUN_ROOT / "s6_downstream/cem/eval/manual_review/e168_user_visual_review.tsv"
OUT_DIR = REPO / "workspace/core4d/report/E168/assets/box021_failure_analysis"

DIAGNOSTICS = {
    "qpos_frames": "lower_worse",
    "duration_s": "lower_worse",
    "body_z_err_p95_m": "higher_worse",
    "body_z_err_peak_m": "higher_worse",
    "track_root_pos_err_cm_mean": "higher_worse",
    "track_root_ori_err_deg_mean": "higher_worse",
    "track_eef_pos_err_cm_mean": "higher_worse",
    "track_eef_ori_err_deg_mean": "higher_worse",
    "track_obj_pos_err_cm_mean": "higher_worse",
    "track_obj_ori_err_deg_mean": "higher_worse",
    "hand_object_physics_contact_in_mask_frac": "lower_worse",
    "hand_object_physics_contact_3mm_in_mask_frac": "lower_worse",
    "hand_object_physics_penetration_3mm_frame_frac": "higher_worse",
    "hand_geom_penetration_2mm_frac": "higher_worse",
    "leg_penetration_frac": "higher_worse",
    "trackbody_jerk_p95": "higher_worse",
    "ankle_acc_max": "higher_worse",
    "ankle_jerk_p95": "higher_worse",
    "trackbody_speed_max": "higher_worse",
    "obj_speed_max": "higher_worse",
    "foot_slip_max_m": "higher_worse",
}

CEM_HEALTH_DIAGNOSTICS = {
    "cem_gate_fallback_used_mean": "higher_worse",
    "cem_gate_fallback_used_last_iter_mean": "higher_worse",
    "cem_gate_fallback_time_frac": "higher_worse",
    "cem_gate_valid_frac_mean": "lower_worse",
    "cem_gate_valid_frac_last_iter_mean": "lower_worse",
    "cem_hand_gate_valid_frac_mean": "lower_worse",
    "cem_hand_gate_valid_frac_last_iter_mean": "lower_worse",
    "cem_body_gate_valid_frac_mean": "lower_worse",
    "cem_body_gate_valid_frac_last_iter_mean": "lower_worse",
    "cem_posture_gate_valid_frac_mean": "lower_worse",
    "cem_posture_gate_valid_frac_last_iter_mean": "lower_worse",
    "sample_posture_violation_mean": "higher_worse",
    "sample_posture_violation_last_iter_mean": "higher_worse",
    "sample_body_gate_violation_pct_mean": "higher_worse",
    "sample_body_gate_violation_pct_last_iter_mean": "higher_worse",
}

# Manual, overlapping mechanism annotations from the 12-frame timelines and the
# representative ref/sim timelines. Primary families are exclusive only to make
# the inventory readable; secondary symptoms preserve the overlaps.
FAILURE_TAXONOMY = {
    "box021_20231011_036_p1": {
        "primary_family": "invalid_lower_body_support",
        "secondary_families": "balance_root_drift",
        "visual_evidence": "Narrow and crossed support steps; repeated leg/box overlap while carrying.",
        "likely_mechanism": "No lower-body/object exclusion or stance-foot constraint.",
    },
    "box021_20231018_028_p1": {
        "primary_family": "hard_reference_feasibility",
        "secondary_families": "temporal_chatter;invalid_lower_body_support;balance_root_drift",
        "visual_evidence": "Short, fast motion with unstable recovery and large whole-body divergence.",
        "likely_mechanism": "Local search becomes infeasible while chasing a high-dynamic reference.",
    },
    "box021_20231018_028_p2": {
        "primary_family": "hard_reference_feasibility",
        "secondary_families": "balance_root_drift;contact_loss;invalid_lower_body_support",
        "visual_evidence": "Extreme lateral reach produces a wide split stance, foot crossing, and contact loss.",
        "likely_mechanism": "Reference reach exceeds the support-preserving envelope.",
    },
    "box021_20231018_029_p1": {
        "primary_family": "invalid_lower_body_support",
        "secondary_families": "balance_root_drift;hard_reference_feasibility",
        "visual_evidence": "Feet route around/through the box and the robot ends supported on its top.",
        "likely_mechanism": "The box is available as an unpenalized support shortcut.",
    },
    "box021_20231018_030_p1": {
        "primary_family": "invalid_lower_body_support",
        "secondary_families": "balance_root_drift;contact_loss",
        "visual_evidence": "Crossed feet and severe root translation culminate in standing on the box.",
        "likely_mechanism": "Hand/object tracking wins over support and lower-body feasibility.",
    },
    "box021_20231018_030_p2": {
        "primary_family": "balance_root_collapse",
        "secondary_families": "temporal_chatter;contact_loss;hard_reference_feasibility",
        "visual_evidence": "Only full fall: backward flip/collapse near the end while the reference stays upright.",
        "likely_mechanism": "Posture-z fallback cannot enforce terminal upright balance.",
    },
    "box021_20231018_031_p2": {
        "primary_family": "invalid_lower_body_support",
        "secondary_families": "contact_loss;balance_root_drift",
        "visual_evidence": "Robot climbs/stands on the box and loses sustained hand contact.",
        "likely_mechanism": "Unpenalized object support substitutes for a valid stance.",
    },
    "box021_20231018_032_p1": {
        "primary_family": "invalid_lower_body_support",
        "secondary_families": "temporal_chatter;balance_root_drift",
        "visual_evidence": "Lower body intersects the box amid high-acceleration, unstable recovery.",
        "likely_mechanism": "Missing collision feasibility combines with an aggressive short-horizon update.",
    },
    "box021_20231018_032_p2": {
        "primary_family": "invalid_lower_body_support",
        "secondary_families": "contact_loss;balance_root_drift",
        "visual_evidence": "Foot steps onto/through the box, followed by separation from the intended carry contact.",
        "likely_mechanism": "No stance-foot anchoring or nonhand-support rejection.",
    },
    "box021_20231018_033_p1": {
        "primary_family": "invalid_lower_body_support",
        "secondary_families": "contact_loss;balance_root_drift",
        "visual_evidence": "Reference remains beside the box; simulation crosses over and stands on it.",
        "likely_mechanism": "A reward-valid but task-invalid support shortcut dominates local tracking.",
    },
    "box021_20231018_033_p2": {
        "primary_family": "invalid_lower_body_support",
        "secondary_families": "balance_root_drift;contact_loss",
        "visual_evidence": "Persistent lower-body/box intersection with extreme torso and end-effector orientation error.",
        "likely_mechanism": "Object support masks loss of whole-body balance.",
    },
    "box021_20231018_034_p2": {
        "primary_family": "temporal_chatter",
        "secondary_families": "invalid_lower_body_support;contact_loss",
        "visual_evidence": "Sparse poses look plausible, but the timeline contains unstable foot/support transitions.",
        "likely_mechanism": "No CEM smoothness term or contact-preserving postprocess.",
    },
    "box021_20231018_035_p2": {
        "primary_family": "balance_root_collapse",
        "secondary_families": "invalid_lower_body_support;temporal_chatter",
        "visual_evidence": "Severe torso lean/collapse against the box with repeated lower-body penetration.",
        "likely_mechanism": "No roll/pitch, support-polygon, or nonhand-support constraint.",
    },
    "box021_20231020_019_p1": {
        "primary_family": "balance_root_collapse",
        "secondary_families": "invalid_lower_body_support;temporal_chatter",
        "visual_evidence": "Unstable one-leg carry with large root rotation and repeated support loss.",
        "likely_mechanism": "Support phase is not represented in the local objective/gates.",
    },
    "box021_20231020_020_p2": {
        "primary_family": "invalid_lower_body_support",
        "secondary_families": "balance_root_drift",
        "visual_evidence": "A foot is placed on/inside the box during lift despite otherwise adequate hand contact.",
        "likely_mechanism": "Current safety geoms exclude the entire lower body.",
    },
}

GATES = {
    "body_z_err_p95_m": ("greater", 0.20),
    "hand_object_physics_contact_in_mask_frac": ("less", 0.50),
    "hand_object_physics_penetration_3mm_frame_frac": ("greater", 0.30),
    "leg_penetration_frac": ("greater", 0.10),
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def number(raw: str) -> float | None:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def summarize(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p25": quantile(values, 0.25),
        "p75": quantile(values, 0.75),
        "p95": quantile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def failure_auc(failed: list[float], usable: list[float], direction: str) -> float:
    wins = 0.0
    pairs = 0
    for bad in failed:
        for good in usable:
            pairs += 1
            if bad == good:
                wins += 0.5
            elif (direction == "higher_worse" and bad > good) or (direction == "lower_worse" and bad < good):
                wins += 1.0
    return wins / pairs


def gate_positive(value: float, operator: str, threshold: float) -> bool:
    return value > threshold if operator == "greater" else value < threshold


def finite_array(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    if key not in data.files:
        return np.asarray([], dtype=np.float64)
    array = np.asarray(data[key], dtype=np.float64)
    return array[np.isfinite(array)]


def array_mean(data: np.lib.npyio.NpzFile, key: str) -> float | None:
    array = finite_array(data, key)
    return float(array.mean()) if array.size else None


def last_iteration_mean(data: np.lib.npyio.NpzFile, key: str) -> float | None:
    if key not in data.files:
        return None
    array = np.asarray(data[key], dtype=np.float64)
    if array.ndim != 2 or array.shape[1] == 0:
        return None
    last = array[:, -1]
    last = last[np.isfinite(last)]
    return float(last.mean()) if last.size else None


def cem_health(row: dict[str, Any]) -> dict[str, Any]:
    result_path = REPO / str(row["result_npz"])
    with np.load(result_path, allow_pickle=True) as data:
        fallback = np.asarray(data["cem_gate_fallback_used"], dtype=np.float64)
        fallback_time_frac = float(np.mean(np.any(fallback > 0.5, axis=1)))
        return {
            "cem_gate_fallback_used_mean": array_mean(data, "cem_gate_fallback_used"),
            "cem_gate_fallback_used_last_iter_mean": last_iteration_mean(
                data, "cem_gate_fallback_used"
            ),
            "cem_gate_fallback_time_frac": fallback_time_frac,
            "cem_gate_valid_frac_mean": array_mean(data, "cem_gate_valid_frac"),
            "cem_gate_valid_frac_last_iter_mean": last_iteration_mean(
                data, "cem_gate_valid_frac"
            ),
            "cem_hand_gate_valid_frac_mean": array_mean(data, "cem_hand_gate_valid_frac"),
            "cem_hand_gate_valid_frac_last_iter_mean": last_iteration_mean(
                data, "cem_hand_gate_valid_frac"
            ),
            "cem_body_gate_valid_frac_mean": array_mean(data, "cem_body_gate_valid_frac"),
            "cem_body_gate_valid_frac_last_iter_mean": last_iteration_mean(
                data, "cem_body_gate_valid_frac"
            ),
            "cem_posture_gate_valid_frac_mean": array_mean(
                data, "cem_posture_gate_valid_frac"
            ),
            "cem_posture_gate_valid_frac_last_iter_mean": last_iteration_mean(
                data, "cem_posture_gate_valid_frac"
            ),
            "sample_posture_violation_mean": array_mean(
                data, "sample_posture_violation_mean"
            ),
            "sample_posture_violation_last_iter_mean": last_iteration_mean(
                data, "sample_posture_violation_mean"
            ),
            "sample_body_gate_violation_pct_mean": array_mean(
                data, "sample_body_gate_violation_pct_mean"
            ),
            "sample_body_gate_violation_pct_last_iter_mean": last_iteration_mean(
                data, "sample_body_gate_violation_pct_mean"
            ),
        }


def compare_groups(
    failed: list[dict[str, Any]],
    usable: list[dict[str, Any]],
    diagnostics: dict[str, str],
) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for metric, direction in diagnostics.items():
        bad = [value for row in failed if (value := number(row.get(metric, ""))) is not None]
        good = [value for row in usable if (value := number(row.get(metric, ""))) is not None]
        bad_summary = summarize(bad)
        good_summary = summarize(good)
        comparisons.append({
            "metric": metric,
            "direction": direction,
            "failed_n": bad_summary["n"],
            "failed_median": bad_summary["median"],
            "failed_p25": bad_summary["p25"],
            "failed_p75": bad_summary["p75"],
            "usable_n": good_summary["n"],
            "usable_median": good_summary["median"],
            "usable_p25": good_summary["p25"],
            "usable_p75": good_summary["p75"],
            "median_delta_failed_minus_usable": bad_summary["median"] - good_summary["median"],
            "failure_auc": failure_auc(bad, good, direction),
        })
    comparisons.sort(key=lambda item: float(item["failure_auc"]), reverse=True)
    return comparisons


def main() -> int:
    metrics = read_tsv(METRICS)
    reviews = {row["case_id"]: row for row in read_tsv(REVIEW)}
    if len(metrics) != 28 or len(reviews) != 28:
        raise SystemExit("expected 28 Box021 metric and review rows")
    for row in metrics:
        row.update({
            "manual_use_decision": reviews[row["case_id"]]["manual_use_decision"],
            "manual_quality_label": reviews[row["case_id"]]["manual_quality_label"],
        })
        row.update(cem_health(row))

    failed = [row for row in metrics if row["manual_use_decision"] == "DO_NOT_USE"]
    usable = [row for row in metrics if row["manual_use_decision"] == "USE"]
    if len(failed) != 15 or len(usable) != 13:
        raise SystemExit(f"expected failed/usable=15/13, got {len(failed)}/{len(usable)}")

    comparisons = compare_groups(failed, usable, DIAGNOSTICS)
    write_tsv(OUT_DIR / "group_comparison.tsv", comparisons, list(comparisons[0]))

    health_comparisons = compare_groups(failed, usable, CEM_HEALTH_DIAGNOSTICS)
    write_tsv(
        OUT_DIR / "cem_health_comparison.tsv",
        health_comparisons,
        list(health_comparisons[0]),
    )
    health_fields = ["case_id", "manual_use_decision"] + list(CEM_HEALTH_DIAGNOSTICS)
    write_tsv(OUT_DIR / "cem_health.tsv", metrics, health_fields)

    gate_rows: list[dict[str, Any]] = []
    for metric, (operator, threshold) in GATES.items():
        flagged = {
            row["case_id"]
            for row in metrics
            if (value := number(row.get(metric, ""))) is not None and gate_positive(value, operator, threshold)
        }
        failed_ids = {row["case_id"] for row in failed}
        usable_ids = {row["case_id"] for row in usable}
        tp = len(flagged & failed_ids)
        fp = len(flagged & usable_ids)
        fn = len(failed_ids - flagged)
        tn = len(usable_ids - flagged)
        gate_rows.append({
            "metric": metric,
            "operator": operator,
            "threshold": threshold,
            "flagged": len(flagged),
            "failed_caught_tp": tp,
            "usable_flagged_fp": fp,
            "failed_missed_fn": fn,
            "usable_clear_tn": tn,
            "failure_recall": tp / len(failed_ids),
            "failure_precision": tp / len(flagged) if flagged else 0.0,
            "usable_specificity": tn / len(usable_ids),
        })
    write_tsv(OUT_DIR / "gate_diagnostics.tsv", gate_rows, list(gate_rows[0]))

    selected_fields = [
        "case_id", "sequence_key", "source_person", "retarget_variant_id",
        "manual_use_decision", "manual_quality_label", "numeric_release_pass",
        "failure_modes", "success_tracked", "fall_flag", "qpos_frames", "duration_s",
        "body_z_err_p95_m",
        "body_z_err_peak_m", "track_root_pos_err_cm_mean", "track_root_ori_err_deg_mean",
        "track_eef_pos_err_cm_mean", "track_eef_ori_err_deg_mean",
        "track_obj_pos_err_cm_mean", "track_obj_ori_err_deg_mean",
        "hand_object_physics_contact_in_mask_frac",
        "hand_object_physics_contact_3mm_in_mask_frac",
        "hand_object_physics_penetration_3mm_frame_frac", "leg_penetration_frac",
        "trackbody_speed_max", "trackbody_jerk_p95", "ankle_acc_max", "ankle_jerk_p95", "obj_speed_max",
        "foot_slip_max_m", "video", "result_npz", "trajectory",
    ]
    write_tsv(OUT_DIR / "case_metrics_selected.tsv", metrics, selected_fields)

    taxonomy_rows = []
    failed_ids = {row["case_id"] for row in failed}
    if set(FAILURE_TAXONOMY) != failed_ids:
        raise SystemExit("failure taxonomy does not exactly cover the 15 rejected cases")
    for row in failed:
        taxonomy_rows.append({"case_id": row["case_id"], **FAILURE_TAXONOMY[row["case_id"]]})
    taxonomy_fields = [
        "case_id", "primary_family", "secondary_families", "visual_evidence", "likely_mechanism"
    ]
    write_tsv(OUT_DIR / "manual_failure_taxonomy.tsv", taxonomy_rows, taxonomy_fields)

    failure_mode_counts: Counter[str] = Counter()
    for row in failed:
        for mode in row["failure_modes"].split(","):
            if mode:
                failure_mode_counts[mode] += 1
    primary_family_counts = Counter(
        annotation["primary_family"] for annotation in FAILURE_TAXONOMY.values()
    )
    date_counts: dict[str, Counter[str]] = {}
    for row in metrics:
        date = row["case_id"].split("_")[1]
        date_counts.setdefault(date, Counter())[row["manual_use_decision"]] += 1
    summary = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scope": "E168 Box021 all 28 manually reviewed cases",
        "manual_counts": {"USE": len(usable), "DO_NOT_USE": len(failed)},
        "numeric_pass_counts": dict(Counter(row["numeric_release_pass"] for row in metrics)),
        "failed_numeric_failure_mode_counts": dict(failure_mode_counts),
        "manual_primary_failure_family_counts": dict(primary_family_counts),
        "manual_counts_by_capture_date": {
            date: dict(counts) for date, counts in sorted(date_counts.items())
        },
        "top_metric_discriminators": comparisons[:8],
        "top_cem_health_discriminators": health_comparisons[:8],
        "gate_diagnostics": gate_rows,
        "source_metrics": str(METRICS.relative_to(REPO)),
        "source_review": str(REVIEW.relative_to(REPO)),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
