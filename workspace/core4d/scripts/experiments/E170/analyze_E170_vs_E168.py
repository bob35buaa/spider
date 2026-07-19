#!/usr/bin/env python3
"""Build reproducible E168-to-E170 paired comparison summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from scipy.stats import wilcoxon


REPO = Path(__file__).resolve().parents[5]
E168_METRICS = REPO / "workspace/core4d/results/E168/s6_downstream/cem/eval/box021_all28_reviewed/e168_case_metrics.tsv"
E170_METRICS = REPO / "workspace/core4d/results/E170/s6_downstream/eval/full/e170_case_metrics.tsv"
E168_TAXONOMY = REPO / "workspace/core4d/report/E168/assets/box021_failure_analysis/manual_failure_taxonomy.tsv"
DEFAULT_OUT = REPO / "workspace/core4d/results/E170/s6_downstream/eval/full/comparison_analysis"


METRICS = [
    ("body_z_err_p95_m", "Body-z p95", "lower"),
    ("track_pelvis_z_err_terminal_m", "Pelvis-z terminal", "lower"),
    ("hand_object_physics_contact_in_mask_frac", "Raw contact in mask", "higher"),
    ("hand_object_physics_contact_3mm_in_mask_frac", "3mm contact in mask", "higher"),
    ("hand_object_release_false_contact_3mm_frac", "Release false contact", "lower"),
    ("hand_object_physics_penetration_3mm_frame_frac", "Hand penetration >3mm", "lower"),
    ("leg_penetration_frac", "Lower-body penetration", "lower"),
    ("leg_near_2cm_frac", "Lower-body near 2cm", "lower"),
    ("track_root_pos_err_cm_mean", "Root position error", "lower"),
    ("track_root_ori_err_deg_mean", "Root orientation error", "lower"),
    ("track_eef_pos_err_cm_mean", "EEF position error", "lower"),
    ("track_eef_ori_err_deg_mean", "EEF orientation error", "lower"),
    ("track_obj_pos_err_cm_mean", "Object position error", "lower"),
    ("track_obj_ori_err_deg_mean", "Object orientation error", "lower"),
    ("qpos_accel_l2_p95", "Qpos acceleration p95", "lower"),
    ("qpos_jerk_l2_p95", "Qpos jerk p95", "lower"),
    ("trackbody_jerk_p95", "Trackbody jerk p95", "lower"),
    ("ankle_jerk_p95", "Ankle jerk p95", "lower"),
    ("obj_speed_max", "Object speed max", "lower"),
    ("foot_slip_max_m", "Foot slip max", "lower"),
]

CORE_SCORE_METRICS = [
    "body_z_err_p95_m",
    "track_pelvis_z_err_terminal_m",
    "hand_object_physics_contact_in_mask_frac",
    "hand_object_physics_contact_3mm_in_mask_frac",
    "hand_object_physics_penetration_3mm_frame_frac",
    "leg_penetration_frac",
    "track_root_pos_err_cm_mean",
    "track_root_ori_err_deg_mean",
    "track_eef_pos_err_cm_mean",
    "track_eef_ori_err_deg_mean",
    "trackbody_jerk_p95",
    "ankle_jerk_p95",
]

CASE_FIELDS = [
    "case_id",
    "date",
    "e168_manual_use_decision",
    "e170_manual_use_decision",
    "manual_transition",
    "e168_numeric_release_pass",
    "e170_numeric_release_pass",
    "numeric_transition",
    "e168_failure_modes",
    "e170_failure_modes",
    "e168_primary_failure_family",
    "e168_secondary_failure_families",
    "e170_manual_failure_taxonomy",
    "core_metric_wins",
    "core_metric_ties",
    "core_metric_losses",
    "leg_penetration_e168",
    "leg_penetration_e170",
    "leg_penetration_delta",
    "raw_contact_e168",
    "raw_contact_e170",
    "raw_contact_delta",
    "contact_3mm_e168",
    "contact_3mm_e170",
    "contact_3mm_delta",
    "hand_penetration_e168",
    "hand_penetration_e170",
    "hand_penetration_delta",
    "body_z_e168",
    "body_z_e170",
    "body_z_delta",
    "root_pos_cm_e168",
    "root_pos_cm_e170",
    "root_pos_cm_delta",
    "root_ori_deg_e168",
    "root_ori_deg_e170",
    "root_ori_deg_delta",
    "eef_pos_cm_e168",
    "eef_pos_cm_e170",
    "eef_pos_cm_delta",
    "trackbody_jerk_e168",
    "trackbody_jerk_e170",
    "trackbody_jerk_delta",
    "ankle_jerk_e168",
    "ankle_jerk_e170",
    "ankle_jerk_delta",
]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def as_float(value: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def improvement(before: float, after: float, direction: str) -> float:
    return after - before if direction == "higher" else before - after


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [1.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, p_values[index] * (total - rank)))
        adjusted[index] = running
    return adjusted


def metric_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for field, label, direction in METRICS:
        pairs: list[tuple[float, float]] = []
        for row in rows:
            before = as_float(row.get(f"e168_{field}", ""))
            after = as_float(row.get(field, ""))
            if before is not None and after is not None:
                pairs.append((before, after))
        gains = [improvement(before, after, direction) for before, after in pairs]
        before_mean = statistics.fmean(before for before, _ in pairs)
        after_mean = statistics.fmean(after for _, after in pairs)
        delta = after_mean - before_mean
        relative = 100.0 * delta / abs(before_mean) if before_mean else ""
        nonzero = [value for value in gains if abs(value) > 1e-12]
        p_value = float(wilcoxon(gains, alternative="two-sided", zero_method="wilcox").pvalue) if nonzero else 1.0
        output.append({
            "metric": field,
            "label": label,
            "direction": direction,
            "n": len(pairs),
            "e168_mean": before_mean,
            "e170_mean": after_mean,
            "delta_e170_minus_e168": delta,
            "relative_change_percent": relative,
            "median_directional_improvement": statistics.median(gains),
            "improved_cases": sum(value > 1e-12 for value in gains),
            "tied_cases": sum(abs(value) <= 1e-12 for value in gains),
            "regressed_cases": sum(value < -1e-12 for value in gains),
            "wilcoxon_p_two_sided": p_value,
        })
    adjusted = holm_adjust([row["wilcoxon_p_two_sided"] for row in output])
    for row, p_adjusted in zip(output, adjusted):
        row["holm_adjusted_p_20_metrics"] = p_adjusted
        directional_mean = row["e170_mean"] - row["e168_mean"]
        mean_improved = directional_mean > 0 if row["direction"] == "higher" else directional_mean < 0
        if mean_improved and p_adjusted < 0.05:
            verdict = "robust_improvement"
        elif mean_improved:
            verdict = "mixed_or_outlier_sensitive_improvement"
        elif abs(directional_mean) <= 1e-12:
            verdict = "no_mean_change"
        else:
            verdict = "mixed_regression"
        row["verdict"] = verdict
    return output


def case_summary(
    rows: list[dict[str, str]],
    baseline: dict[str, dict[str, str]],
    taxonomy: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    direction_by_metric = {field: direction for field, _, direction in METRICS}
    output: list[dict[str, Any]] = []
    for row in rows:
        case_id = row["case_id"]
        base = baseline[case_id]
        tax = taxonomy.get(case_id, {})
        wins = ties = losses = 0
        for field in CORE_SCORE_METRICS:
            before = as_float(row.get(f"e168_{field}", ""))
            after = as_float(row.get(field, ""))
            if before is None or after is None:
                continue
            value = improvement(before, after, direction_by_metric[field])
            if value > 1e-12:
                wins += 1
            elif value < -1e-12:
                losses += 1
            else:
                ties += 1

        def triple(field: str, prefix: str) -> dict[str, Any]:
            before = as_float(row.get(f"e168_{field}", ""))
            after = as_float(row.get(field, ""))
            return {
                f"{prefix}_e168": before if before is not None else "",
                f"{prefix}_e170": after if after is not None else "",
                f"{prefix}_delta": after - before if before is not None and after is not None else "",
            }

        output.append({
            "case_id": case_id,
            "date": case_id.split("_")[1],
            "e168_manual_use_decision": row["e168_manual_use_decision"],
            "e170_manual_use_decision": row["manual_use_decision"],
            "manual_transition": f"{row['e168_manual_use_decision']}->{row['manual_use_decision']}",
            "e168_numeric_release_pass": base["numeric_release_pass"],
            "e170_numeric_release_pass": row["numeric_release_pass"],
            "numeric_transition": f"{base['numeric_release_pass']}->{row['numeric_release_pass']}",
            "e168_failure_modes": base["failure_modes"],
            "e170_failure_modes": row["numeric_failure_modes"],
            "e168_primary_failure_family": tax.get("primary_family", ""),
            "e168_secondary_failure_families": tax.get("secondary_families", ""),
            "e170_manual_failure_taxonomy": row["manual_failure_taxonomy"],
            "core_metric_wins": wins,
            "core_metric_ties": ties,
            "core_metric_losses": losses,
            **triple("leg_penetration_frac", "leg_penetration"),
            **triple("hand_object_physics_contact_in_mask_frac", "raw_contact"),
            **triple("hand_object_physics_contact_3mm_in_mask_frac", "contact_3mm"),
            **triple("hand_object_physics_penetration_3mm_frame_frac", "hand_penetration"),
            **triple("body_z_err_p95_m", "body_z"),
            **triple("track_root_pos_err_cm_mean", "root_pos_cm"),
            **triple("track_root_ori_err_deg_mean", "root_ori_deg"),
            **triple("track_eef_pos_err_cm_mean", "eef_pos_cm"),
            **triple("trackbody_jerk_p95", "trackbody_jerk"),
            **triple("ankle_jerk_p95", "ankle_jerk"),
        })
    return output


def group_metric_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    predicates = {
        "recovered": lambda row: row["e168_manual_use_decision"] == "DO_NOT_USE" and row["manual_use_decision"] == "USE",
        "unresolved": lambda row: row["e168_manual_use_decision"] == "DO_NOT_USE" and row["manual_use_decision"] == "DO_NOT_USE",
        "retained": lambda row: row["e168_manual_use_decision"] == "USE" and row["manual_use_decision"] == "USE",
    }
    selected_metrics = [
        "body_z_err_p95_m",
        "hand_object_physics_contact_in_mask_frac",
        "hand_object_physics_contact_3mm_in_mask_frac",
        "hand_object_physics_penetration_3mm_frame_frac",
        "leg_penetration_frac",
        "leg_near_2cm_frac",
        "track_root_pos_err_cm_mean",
        "track_root_ori_err_deg_mean",
        "track_eef_pos_err_cm_mean",
        "trackbody_jerk_p95",
        "ankle_jerk_p95",
    ]
    direction_by_metric = {field: direction for field, _, direction in METRICS}
    output: dict[str, Any] = {}
    for group, predicate in predicates.items():
        group_rows = [row for row in rows if predicate(row)]
        metric_payload: dict[str, Any] = {}
        for field in selected_metrics:
            pairs = []
            for row in group_rows:
                before = as_float(row.get(f"e168_{field}", ""))
                after = as_float(row.get(field, ""))
                if before is not None and after is not None:
                    pairs.append((before, after))
            gains = [improvement(before, after, direction_by_metric[field]) for before, after in pairs]
            metric_payload[field] = {
                "n": len(pairs),
                "e168_mean": statistics.fmean(before for before, _ in pairs),
                "e170_mean": statistics.fmean(after for _, after in pairs),
                "improved_cases": sum(value > 1e-12 for value in gains),
                "tied_cases": sum(abs(value) <= 1e-12 for value in gains),
                "regressed_cases": sum(value < -1e-12 for value in gains),
            }
        output[group] = {"rows": len(group_rows), "metrics": metric_payload}
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    e168_rows = read_tsv(E168_METRICS)
    e170_rows = read_tsv(E170_METRICS)
    taxonomy_rows = read_tsv(E168_TAXONOMY)
    baseline = {row["case_id"]: row for row in e168_rows}
    taxonomy = {row["case_id"]: row for row in taxonomy_rows}
    if len(e168_rows) != 28 or len(e170_rows) != 28 or set(baseline) != {row["case_id"] for row in e170_rows}:
        raise SystemExit("E168/E170 metrics must contain the same 28 unique cases")

    metrics = metric_summary(e170_rows)
    cases = case_summary(e170_rows, baseline, taxonomy)
    manual_transitions: dict[str, list[str]] = defaultdict(list)
    numeric_transitions: dict[str, list[str]] = defaultdict(list)
    date_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for case in cases:
        manual_transitions[case["manual_transition"]].append(case["case_id"])
        numeric_transitions[case["numeric_transition"]].append(case["case_id"])
        date_counts[case["date"]][f"E168_{case['e168_manual_use_decision']}"] += 1
        date_counts[case["date"]][f"E170_{case['e170_manual_use_decision']}"] += 1

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    metric_fields = list(metrics[0])
    write_tsv(out_dir / "e170_vs_e168_metric_summary.tsv", metrics, metric_fields)
    write_tsv(out_dir / "e170_vs_e168_case_summary.tsv", cases, CASE_FIELDS)
    payload = {
        "experiment": "E170",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "comparison": "E168 E167A baseline versus E170 frozen PRG",
        "rows": 28,
        "manual_transition_counts": {key: len(value) for key, value in sorted(manual_transitions.items())},
        "manual_transition_cases": {key: sorted(value) for key, value in sorted(manual_transitions.items())},
        "numeric_transition_counts": {key: len(value) for key, value in sorted(numeric_transitions.items())},
        "numeric_transition_cases": {key: sorted(value) for key, value in sorted(numeric_transitions.items())},
        "operational_use_e168": sum(row["manual_use_decision"] == "USE" for row in e168_rows),
        "operational_use_e170": sum(row["manual_use_decision"] == "USE" for row in e170_rows),
        "numeric_pass_e168": sum(row["numeric_release_pass"] == "true" for row in e168_rows),
        "numeric_pass_e170": sum(row["numeric_release_pass"] == "true" for row in e170_rows),
        "date_counts": {key: dict(value) for key, value in sorted(date_counts.items())},
        "e170_failure_taxonomy_counts": dict(Counter(
            taxonomy_item
            for row in e170_rows
            if row["manual_use_decision"] == "DO_NOT_USE"
            for taxonomy_item in row["manual_failure_taxonomy"].split(",")
            if taxonomy_item
        )),
        "group_metric_summary": group_metric_summary(e170_rows),
        "metric_summary_tsv": str(out_dir / "e170_vs_e168_metric_summary.tsv"),
        "case_summary_tsv": str(out_dir / "e170_vs_e168_case_summary.tsv"),
        "statistical_note": "Wilcoxon tests treat 28 person-cases as paired observations; Holm correction covers 20 metrics. This measures cross-case directional consistency, not optimizer seed variance or sequence-level independence.",
    }
    (out_dir / "e170_vs_e168_analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "out_dir": str(out_dir),
        "manual_transition_counts": payload["manual_transition_counts"],
        "numeric_transition_counts": payload["numeric_transition_counts"],
        "robust_metric_improvements": [row["metric"] for row in metrics if row["verdict"] == "robust_improvement"],
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
