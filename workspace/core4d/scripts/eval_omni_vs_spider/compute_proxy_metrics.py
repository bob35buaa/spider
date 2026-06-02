#!/usr/bin/env python3
"""Compute threshold-explicit proxy metrics for the fair-eval case bank."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    DEFAULT_THRESHOLDS_M,
    DEEP_PENETRATION_THRESHOLD_M,
    FALL_PELVIS_THRESHOLD_M,
    as_float,
    read_csv,
    resolve_path,
    threshold_tag,
    write_json,
    write_tsv,
)


BASE_FIELDS = [
    "bank_row_id",
    "case_id",
    "case_key",
    "object_key",
    "method",
    "method_family",
    "experiment",
    "source_kind",
    "variant",
    "fair_metric_scope",
    "fair_metric_ready",
    "thresholds_m",
    "thresholds_note",
    "cem_status",
    "rl_status",
    "pelvis_min_m",
    "fall_threshold_m",
    "fall_flag",
    "obj_err_mean_m",
    "obj_err_max_m",
    "object_tracking_fairness",
    "leg_object_interference_frac",
    "leg_object_interference_threshold_m",
    "deep_penetration_frac_2cm",
    "deep_penetration_threshold_m",
    "hand_object_penetration_frac",
    "metric_warnings",
]


def metric_fields(thresholds: tuple[float, ...]) -> list[str]:
    fields = list(BASE_FIELDS)
    for threshold in thresholds:
        tag = threshold_tag(threshold)
        fields.extend(
            [
                f"hand_near_frac_{tag}",
                f"leg_near_frac_{tag}",
                f"threshold_m_{tag}",
            ]
        )
    return fields


def load_summary_row(row: dict[str, str]) -> dict[str, str]:
    metrics_ref = resolve_path(row.get("metrics_ref"))
    variant = row.get("variant", "")
    if not metrics_ref or not metrics_ref.is_file() or metrics_ref.suffix.lower() != ".csv":
        return {}
    for item in read_csv(metrics_ref):
        if variant and item.get("variant") == variant:
            return item
        if row.get("case_id") and item.get("case_id") == row.get("case_id"):
            return item
    return {}


def load_timeseries(row: dict[str, str]) -> list[dict[str, str]]:
    path = resolve_path(row.get("legobj_timeseries_csv"))
    if not path or not path.is_file():
        return []
    return read_csv(path)


def fraction(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else math.nan


def metrics_from_timeseries(rows: list[dict[str, str]], thresholds: tuple[float, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    hand_sdf = np.asarray([as_float(r.get("hand_box_sdf_min_m")) for r in rows], dtype=np.float64)
    leg_sdf = np.asarray([as_float(r.get("leg_box_sdf_min_m")) for r in rows], dtype=np.float64)
    hand_sdf = hand_sdf[np.isfinite(hand_sdf)]
    leg_sdf = leg_sdf[np.isfinite(leg_sdf)]
    if hand_sdf.size:
        out["hand_object_penetration_frac"] = fraction(hand_sdf < 0.0)
        out["deep_penetration_frac_2cm"] = fraction(hand_sdf < DEEP_PENETRATION_THRESHOLD_M)
        for threshold in thresholds:
            out[f"hand_near_frac_{threshold_tag(threshold)}"] = fraction(hand_sdf < threshold)
    if leg_sdf.size:
        out["leg_object_interference_frac"] = fraction(leg_sdf < 0.0)
        for threshold in thresholds:
            out[f"leg_near_frac_{threshold_tag(threshold)}"] = fraction(leg_sdf < threshold)
    return out


def metrics_from_summary(summary: dict[str, str], thresholds: tuple[float, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not summary:
        return out
    out["pelvis_min_m"] = summary.get("pelvis_min_m", "")
    out["obj_err_mean_m"] = summary.get("obj_err_mean_m", summary.get("obj_pos_cm", ""))
    out["obj_err_max_m"] = summary.get("obj_err_max_m", "")
    out["leg_object_interference_frac"] = summary.get("leg_box_interference_frac", "")
    out["deep_penetration_frac_2cm"] = summary.get("deep_pen_pct", "")
    out["hand_object_penetration_frac"] = summary.get("hand_object_penetration_duration_pct", "")

    # Most current CEM summaries already define an 8cm near-contact proxy.
    if summary.get("contact_frac_either") not in (None, ""):
        out["hand_near_frac_08cm"] = summary.get("contact_frac_either", "")
    if summary.get("contact_proxy_pct") not in (None, ""):
        out["hand_near_frac_05cm"] = as_float(summary.get("contact_proxy_pct")) / 100.0
    if summary.get("leg_box_near_2cm_frac") not in (None, ""):
        out["leg_near_frac_02cm"] = summary.get("leg_box_near_2cm_frac", "")
    for threshold in thresholds:
        out[f"threshold_m_{threshold_tag(threshold)}"] = threshold
    return out


def normalize_fraction(value: Any) -> str:
    v = as_float(value)
    if not math.isfinite(v):
        return ""
    # E026 stores several metrics in percent, current CEM stores fractions.
    if v > 1.0:
        v = v / 100.0
    return f"{v:.6f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-bank", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--thresholds-m", default="0.03,0.05,0.08")
    args = parser.parse_args()

    thresholds = tuple(float(x) for x in args.thresholds_m.split(",") if x.strip())
    if len(thresholds) < 3:
        raise ValueError("--thresholds-m must contain at least three thresholds")

    case_rows = read_csv(args.case_bank, delimiter="\t")
    metric_rows: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    for row in case_rows:
        summary = load_summary_row(row)
        series = load_timeseries(row)
        metrics = metrics_from_summary(summary, thresholds)
        if series:
            metrics.update(metrics_from_timeseries(series, thresholds))

        pelvis = as_float(metrics.get("pelvis_min_m"))
        source_kind = row.get("source_kind", "")
        is_historical_e026 = source_kind == "e026_method_metrics"
        if is_historical_e026:
            fair_metric_scope = "diagnostic_only"
        elif series:
            fair_metric_scope = "geometry_timeseries_proxy"
        elif summary:
            fair_metric_scope = "summary_proxy"
        else:
            fair_metric_scope = "metadata_only"
        fair_metric_ready = (not is_historical_e026) and bool(series or summary)

        warn_parts: list[str] = []
        if not summary and not series:
            warn_parts.append("no_metric_source_found")
        if row.get("self_ref_tracking_caveat"):
            warn_parts.append(row["self_ref_tracking_caveat"])
        if is_historical_e026:
            warn_parts.append("historical_e026_metrics_not_raw_gt")

        out = {
            "bank_row_id": row.get("bank_row_id", ""),
            "case_id": row.get("case_id", ""),
            "case_key": row.get("case_key", ""),
            "object_key": row.get("object_key", ""),
            "method": row.get("method", ""),
            "method_family": row.get("method_family", ""),
            "experiment": row.get("experiment", ""),
            "source_kind": source_kind,
            "variant": row.get("variant", ""),
            "fair_metric_scope": fair_metric_scope,
            "fair_metric_ready": fair_metric_ready,
            "thresholds_m": ",".join(f"{x:.3f}" for x in thresholds),
            "thresholds_note": "Near-contact thresholds are reported independently; do not compare thresholded metrics without the threshold suffix.",
            "cem_status": row.get("cem_status", ""),
            "rl_status": row.get("rl_status", ""),
            "pelvis_min_m": metrics.get("pelvis_min_m", ""),
            "fall_threshold_m": FALL_PELVIS_THRESHOLD_M,
            "fall_flag": bool(math.isfinite(pelvis) and pelvis < FALL_PELVIS_THRESHOLD_M),
            "obj_err_mean_m": metrics.get("obj_err_mean_m", ""),
            "obj_err_max_m": metrics.get("obj_err_max_m", ""),
            "object_tracking_fairness": "not_fair_historical_method_ref"
            if is_historical_e026 and metrics.get("obj_err_mean_m", "") != ""
            else "not_fair_method_ref_until_raw_object_gt"
            if metrics.get("obj_err_mean_m", "") != ""
            else "",
            "leg_object_interference_frac": normalize_fraction(metrics.get("leg_object_interference_frac", "")),
            "leg_object_interference_threshold_m": 0.0,
            "deep_penetration_frac_2cm": normalize_fraction(metrics.get("deep_penetration_frac_2cm", "")),
            "deep_penetration_threshold_m": DEEP_PENETRATION_THRESHOLD_M,
            "hand_object_penetration_frac": normalize_fraction(metrics.get("hand_object_penetration_frac", "")),
            "metric_warnings": ";".join(dict.fromkeys(warn_parts)),
        }
        for threshold in thresholds:
            tag = threshold_tag(threshold)
            out[f"threshold_m_{tag}"] = f"{threshold:.3f}"
            out[f"hand_near_frac_{tag}"] = normalize_fraction(metrics.get(f"hand_near_frac_{tag}", ""))
            out[f"leg_near_frac_{tag}"] = normalize_fraction(metrics.get(f"leg_near_frac_{tag}", ""))
        metric_rows.append(out)
        if warn_parts:
            warnings.append(
                {
                    "bank_row_id": row.get("bank_row_id", ""),
                    "case_id": row.get("case_id", ""),
                    "method": row.get("method", ""),
                    "warnings": ";".join(dict.fromkeys(warn_parts)),
                }
            )

    fields = metric_fields(thresholds)
    write_tsv(args.out_dir / "method_case_metrics.tsv", metric_rows, fields)
    write_tsv(args.out_dir / "warnings.tsv", warnings, ["bank_row_id", "case_id", "method", "warnings"])
    write_json(
        args.out_dir / "metric_summary.json",
        {
            "num_rows": len(metric_rows),
            "num_ready": sum(str(r["fair_metric_ready"]) == "True" for r in metric_rows),
            "thresholds_m": thresholds,
            "deep_penetration_threshold_m": DEEP_PENETRATION_THRESHOLD_M,
            "fall_pelvis_threshold_m": FALL_PELVIS_THRESHOLD_M,
        },
    )
    print(f"[compute_proxy_metrics] wrote {len(metric_rows)} rows to {args.out_dir / 'method_case_metrics.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
