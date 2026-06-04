#!/usr/bin/env python3
"""E110 contact metric audit from E109 unified replay aggregate metrics.

This audit intentionally does not recompute MuJoCo replay. It consumes the
E109 method metrics table and decomposes existing hand-object aggregate
fractions into contact/penetration bands, per-case method deltas, and object
summaries. Raw-contact precision/recall is recorded as coverage status only
until E111 wires S1 raw contact artifacts into the manifests.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import REPO, as_float, read_csv, write_json, write_tsv


METHODS = ("OmniRetarget", "Spider CEM")
DEFAULT_METRICS = REPO / "workspace/core4d/results/E109/expanded_24_work_cases/expanded_24_method_metrics.tsv"
DEFAULT_COMPARISON = REPO / "workspace/core4d/results/E109/expanded_24_work_cases/expanded_24_case_comparison.tsv"
DEFAULT_OUT_DIR = REPO / "workspace/core4d/results/E110/contact_metric_audit"

BAND_FIELDS = [
    "case_id",
    "method",
    "object_key",
    "target_variant_id",
    "run_id",
    "tier",
    "strict_main",
    "qpos_frames",
    "hand_geom_penetration_frac",
    "hand_geom_deep_penetration_2cm_frac",
    "hand_geom_shallow_penetration_frac",
    "hand_geom_near_0_2cm_frac",
    "hand_geom_near_2_5cm_frac",
    "hand_geom_near_0_3cm_frac",
    "hand_geom_near_3_5cm_frac",
    "hand_geom_near_5_12cm_frac",
    "hand_geom_far_gt5cm_frac",
    "hand_geom_far_gt12cm_frac",
    "hand_geom_near_5cm_frac",
    "hand_geom_near_12cm_frac",
    "hand_object_physics_contact_frac",
    "eef_near_3cm_frac",
    "eef_near_5cm_frac",
    "eef_near_8cm_frac",
    "leg_penetration_frac",
    "body_penetration_frac",
    "pelvis_min_m",
    "raw_contact_coverage_status",
    "raw_contact_precision",
    "raw_contact_recall",
    "raw_contact_f1",
]

DELTA_FIELDS = [
    "case_id",
    "object_key",
    "target_variant_id",
    "run_id",
    "tier",
    "strict_main",
    "delta_hand_object_physics_contact",
    "delta_eef_near_3cm",
    "delta_eef_near_5cm",
    "delta_eef_near_8cm",
    "delta_hand_geom_near_5cm",
    "delta_hand_geom_near_12cm",
    "delta_hand_geom_penetration",
    "delta_hand_geom_deep_penetration_2cm",
    "delta_hand_geom_shallow_penetration",
    "delta_hand_geom_far_gt5cm",
    "delta_hand_geom_far_gt12cm",
    "delta_leg_penetration",
    "delta_body_penetration",
    "delta_pelvis_min_m",
    "safety_regression_flag",
    "secondary_flags",
    "failure_label",
    "interpretation",
]

SUMMARY_FIELDS = [
    "group",
    "method",
    "num_rows",
    "mean_hand_object_physics_contact_frac",
    "mean_eef_near_5cm_frac",
    "mean_hand_geom_near_5cm_frac",
    "mean_hand_geom_near_12cm_frac",
    "mean_hand_geom_penetration_frac",
    "mean_hand_geom_deep_penetration_2cm_frac",
    "mean_hand_geom_shallow_penetration_frac",
    "mean_hand_geom_far_gt5cm_frac",
    "mean_hand_geom_far_gt12cm_frac",
    "mean_leg_penetration_frac",
    "mean_body_penetration_frac",
    "mean_pelvis_min_m",
]


def finite(value: Any) -> float:
    return as_float(value)


def fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}" if math.isfinite(value) else ""


def frac_diff(a: float, b: float) -> float:
    if not math.isfinite(a) or not math.isfinite(b):
        return math.nan
    return a - b


def clamp_frac(value: float) -> float:
    if not math.isfinite(value):
        return math.nan
    return max(0.0, min(1.0, value))


def tier_lookup(comparison_rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    out: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in comparison_rows:
        key = (row.get("case_id", ""), row.get("target_variant_id", ""), row.get("spider_run_id", ""))
        out[key] = {
            "tier": row.get("分层", ""),
            "strict_main": row.get("是否strict主表", ""),
        }
    return out


def band_row(row: dict[str, str], tiers: dict[tuple[str, str, str], dict[str, str]]) -> dict[str, Any]:
    case_id = row.get("case_id", "")
    target_variant_id = row.get("target_variant_id", "")
    run_id = row.get("run_id", "")
    tier = tiers.get((case_id, target_variant_id, run_id), {})
    pen = finite(row.get("hand_geom_penetration_frac"))
    deep = finite(row.get("hand_geom_deep_penetration_2cm_frac"))
    near3 = finite(row.get("hand_geom_near_3cm_frac"))
    near5 = finite(row.get("hand_geom_near_5cm_frac"))
    near12 = finite(row.get("hand_geom_near_12cm_frac"))
    shallow = clamp_frac(pen - deep) if math.isfinite(pen) and math.isfinite(deep) else math.nan
    near_0_3 = clamp_frac(near3 - pen) if math.isfinite(near3) and math.isfinite(pen) else math.nan
    near_3_5 = clamp_frac(near5 - near3) if math.isfinite(near5) and math.isfinite(near3) else math.nan
    near_5_12 = clamp_frac(near12 - near5) if math.isfinite(near12) and math.isfinite(near5) else math.nan
    far_gt5 = clamp_frac(1.0 - near5) if math.isfinite(near5) else math.nan
    far_gt12 = clamp_frac(1.0 - near12) if math.isfinite(near12) else math.nan
    out: dict[str, Any] = {
        "case_id": case_id,
        "method": row.get("method", ""),
        "object_key": row.get("object_key", ""),
        "target_variant_id": target_variant_id,
        "run_id": run_id,
        "tier": tier.get("tier", ""),
        "strict_main": tier.get("strict_main", ""),
        "qpos_frames": row.get("qpos_frames", ""),
        "hand_geom_penetration_frac": fmt(pen),
        "hand_geom_deep_penetration_2cm_frac": fmt(deep),
        "hand_geom_shallow_penetration_frac": fmt(shallow),
        # E109 aggregate replay did not output a 2cm near-contact threshold.
        # Keep the planned columns explicit and blank until E111 per-frame SDF
        # alignment can provide them without interpolation.
        "hand_geom_near_0_2cm_frac": "",
        "hand_geom_near_2_5cm_frac": "",
        "hand_geom_near_0_3cm_frac": fmt(near_0_3),
        "hand_geom_near_3_5cm_frac": fmt(near_3_5),
        "hand_geom_near_5_12cm_frac": fmt(near_5_12),
        "hand_geom_far_gt5cm_frac": fmt(far_gt5),
        "hand_geom_far_gt12cm_frac": fmt(far_gt12),
        "hand_geom_near_5cm_frac": row.get("hand_geom_near_5cm_frac", ""),
        "hand_geom_near_12cm_frac": row.get("hand_geom_near_12cm_frac", ""),
        "hand_object_physics_contact_frac": row.get("hand_object_physics_contact_frac", ""),
        "eef_near_3cm_frac": row.get("eef_near_3cm_frac", ""),
        "eef_near_5cm_frac": row.get("eef_near_5cm_frac", ""),
        "eef_near_8cm_frac": row.get("eef_near_8cm_frac", ""),
        "leg_penetration_frac": row.get("leg_penetration_frac", ""),
        "body_penetration_frac": row.get("body_penetration_frac", ""),
        "pelvis_min_m": row.get("pelvis_min_m", ""),
        "raw_contact_coverage_status": "missing_artifacts",
        "raw_contact_precision": "",
        "raw_contact_recall": "",
        "raw_contact_f1": "",
    }
    return out


def by_case_method(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, dict[str, str]]]:
    out: dict[tuple[str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        key = (row.get("case_id", ""), row.get("target_variant_id", ""), row.get("run_id", ""))
        out[key][row.get("method", "")] = row
    return out


def classify_delta(omni: dict[str, str], spider: dict[str, str]) -> tuple[str, str]:
    d_phys = frac_diff(finite(spider.get("hand_object_physics_contact_frac")), finite(omni.get("hand_object_physics_contact_frac")))
    d_deep = frac_diff(finite(spider.get("hand_geom_deep_penetration_2cm_frac")), finite(omni.get("hand_geom_deep_penetration_2cm_frac")))
    d_near12 = frac_diff(finite(spider.get("hand_geom_near_12cm_frac")), finite(omni.get("hand_geom_near_12cm_frac")))
    d_near5 = frac_diff(finite(spider.get("hand_geom_near_5cm_frac")), finite(omni.get("hand_geom_near_5cm_frac")))
    d_leg = frac_diff(finite(spider.get("leg_penetration_frac")), finite(omni.get("leg_penetration_frac")))
    d_pelvis = frac_diff(finite(spider.get("pelvis_min_m")), finite(omni.get("pelvis_min_m")))

    if math.isfinite(d_phys) and d_phys <= -0.05 and math.isfinite(d_deep) and d_deep <= -0.05:
        if math.isfinite(d_near12) and d_near12 >= -0.05 and math.isfinite(d_near5) and d_near5 <= -0.05:
            return (
                "safe_gap_near_contact",
                "deep penetration was removed while broad 12cm proximity stayed comparable, but 5cm/physics contact dropped.",
            )
        return (
            "penetration_removed_contact_not_recovered",
            "deep penetration was removed, but physics contact dropped and no near-band compensation is evident.",
        )
    if (
        math.isfinite(d_phys)
        and d_phys <= -0.05
        and ((math.isfinite(d_leg) and d_leg > 0.02) or (math.isfinite(d_pelvis) and d_pelvis < -0.03))
    ):
        return (
            "contact_regression_with_safety_regression",
            "contact dropped together with lower-body or pelvis degradation.",
        )
    if math.isfinite(d_phys) and d_phys >= -0.05 and (not math.isfinite(d_deep) or d_deep <= 0.02):
        return (
            "contact_preserved_or_improved",
            "physics contact is preserved within 5pp while deep penetration is not worse.",
        )
    return ("neutral", "no dominant E110 aggregate failure label.")


def safety_regression(omni: dict[str, str], spider: dict[str, str]) -> tuple[bool, list[str]]:
    flags = []
    d_leg = frac_diff(finite(spider.get("leg_penetration_frac")), finite(omni.get("leg_penetration_frac")))
    d_body = frac_diff(finite(spider.get("body_penetration_frac")), finite(omni.get("body_penetration_frac")))
    d_pelvis = frac_diff(finite(spider.get("pelvis_min_m")), finite(omni.get("pelvis_min_m")))
    if math.isfinite(d_leg) and d_leg > 0.02:
        flags.append("leg_penetration_regression")
    if math.isfinite(d_body) and d_body > 0.02:
        flags.append("body_penetration_regression")
    if math.isfinite(d_pelvis) and d_pelvis < -0.03:
        flags.append("pelvis_height_regression")
    return bool(flags), flags


def delta_rows(metrics: list[dict[str, str]], bands_by_key_method: dict[tuple[str, str, str], dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for key, methods in by_case_method(metrics).items():
        omni = methods.get("OmniRetarget")
        spider = methods.get("Spider CEM")
        if not omni or not spider:
            continue
        band_methods = bands_by_key_method.get(key, {})
        spider_band = band_methods.get("Spider CEM", {})
        omni_band = band_methods.get("OmniRetarget", {})
        label, interpretation = classify_delta(omni, spider)
        safety_flag, safety_flags = safety_regression(omni, spider)

        def d(field: str) -> str:
            return fmt(frac_diff(finite(spider.get(field)), finite(omni.get(field))))

        def dband(field: str) -> str:
            return fmt(frac_diff(finite(spider_band.get(field)), finite(omni_band.get(field))))

        rows.append(
            {
                "case_id": key[0],
                "object_key": spider.get("object_key", ""),
                "target_variant_id": key[1],
                "run_id": key[2],
                "tier": spider_band.get("tier", ""),
                "strict_main": spider_band.get("strict_main", ""),
                "delta_hand_object_physics_contact": d("hand_object_physics_contact_frac"),
                "delta_eef_near_3cm": d("eef_near_3cm_frac"),
                "delta_eef_near_5cm": d("eef_near_5cm_frac"),
                "delta_eef_near_8cm": d("eef_near_8cm_frac"),
                "delta_hand_geom_near_5cm": d("hand_geom_near_5cm_frac"),
                "delta_hand_geom_near_12cm": d("hand_geom_near_12cm_frac"),
                "delta_hand_geom_penetration": d("hand_geom_penetration_frac"),
                "delta_hand_geom_deep_penetration_2cm": d("hand_geom_deep_penetration_2cm_frac"),
                "delta_hand_geom_shallow_penetration": dband("hand_geom_shallow_penetration_frac"),
                "delta_hand_geom_far_gt5cm": dband("hand_geom_far_gt5cm_frac"),
                "delta_hand_geom_far_gt12cm": dband("hand_geom_far_gt12cm_frac"),
                "delta_leg_penetration": d("leg_penetration_frac"),
                "delta_body_penetration": d("body_penetration_frac"),
                "delta_pelvis_min_m": d("pelvis_min_m"),
                "safety_regression_flag": "true" if safety_flag else "false",
                "secondary_flags": ",".join(safety_flags),
                "failure_label": label,
                "interpretation": interpretation,
            }
        )
    return rows


def mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else math.nan


def summarize(rows: list[dict[str, Any]], group_key: str | None = None) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group = row.get(group_key, "all") if group_key else "all"
        grouped[(str(group), str(row.get("method", "")))].append(row)
    out = []
    for (group, method), rs in sorted(grouped.items()):
        out.append(
            {
                "group": group,
                "method": method,
                "num_rows": len(rs),
                "mean_hand_object_physics_contact_frac": fmt(mean([finite(r.get("hand_object_physics_contact_frac")) for r in rs])),
                "mean_eef_near_5cm_frac": fmt(mean([finite(r.get("eef_near_5cm_frac")) for r in rs])),
                "mean_hand_geom_near_5cm_frac": fmt(mean([finite(r.get("hand_geom_near_5cm_frac")) for r in rs])),
                "mean_hand_geom_near_12cm_frac": fmt(mean([finite(r.get("hand_geom_near_12cm_frac")) for r in rs])),
                "mean_hand_geom_penetration_frac": fmt(mean([finite(r.get("hand_geom_penetration_frac")) for r in rs])),
                "mean_hand_geom_deep_penetration_2cm_frac": fmt(mean([finite(r.get("hand_geom_deep_penetration_2cm_frac")) for r in rs])),
                "mean_hand_geom_shallow_penetration_frac": fmt(mean([finite(r.get("hand_geom_shallow_penetration_frac")) for r in rs])),
                "mean_hand_geom_far_gt5cm_frac": fmt(mean([finite(r.get("hand_geom_far_gt5cm_frac")) for r in rs])),
                "mean_hand_geom_far_gt12cm_frac": fmt(mean([finite(r.get("hand_geom_far_gt12cm_frac")) for r in rs])),
                "mean_leg_penetration_frac": fmt(mean([finite(r.get("leg_penetration_frac")) for r in rs])),
                "mean_body_penetration_frac": fmt(mean([finite(r.get("body_penetration_frac")) for r in rs])),
                "mean_pelvis_min_m": fmt(mean([finite(r.get("pelvis_min_m")) for r in rs])),
            }
        )
    return out


def write_markdown(path: Path, *, method_summary: list[dict[str, Any]], object_summary: list[dict[str, Any]], deltas: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = [
        "# E110 Contact Metric Audit",
        "",
        "## Summary",
        "",
        f"- method rows: `{summary['method_rows']}`",
        f"- paired cases: `{summary['paired_cases']}`",
        f"- raw-contact coverage: `{summary['raw_contact_coverage_status']}`",
        "",
        "## Method Summary",
        "",
        "| method | N | physics contact | eef5 | hand5 | hand12 | hand pen | hand deep | shallow pen | far>5 | far>12 | leg pen | pelvis |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in method_summary:
        if row["group"] != "all":
            continue
        lines.append(
            f"| {row['method']} | {row['num_rows']} | {row['mean_hand_object_physics_contact_frac']} | "
            f"{row['mean_eef_near_5cm_frac']} | {row['mean_hand_geom_near_5cm_frac']} | "
            f"{row['mean_hand_geom_near_12cm_frac']} | {row['mean_hand_geom_penetration_frac']} | "
            f"{row['mean_hand_geom_deep_penetration_2cm_frac']} | {row['mean_hand_geom_shallow_penetration_frac']} | "
            f"{row['mean_hand_geom_far_gt5cm_frac']} | {row['mean_hand_geom_far_gt12cm_frac']} | "
            f"{row['mean_leg_penetration_frac']} | {row['mean_pelvis_min_m']} |"
        )
    lines.extend(
        [
            "",
            "## Failure Labels",
            "",
            "| label | count |",
            "|---|---:|",
        ]
    )
    for label, count in summary["failure_label_counts"].items():
        lines.append(f"| `{label}` | {count} |")
    lines.extend(
        [
            "",
            f"- safety regression rows: `{summary['safety_regression_count']}`",
        ]
    )

    lines.extend(
        [
            "",
            "## Top Physics Contact Regressions",
            "",
            "| case | object | delta physics | delta eef5 | delta deep pen | label |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    top = sorted(deltas, key=lambda r: finite(r.get("delta_hand_object_physics_contact")))[:10]
    for row in top:
        lines.append(
            f"| `{row['case_id']}` | {row['object_key']} | {row['delta_hand_object_physics_contact']} | "
            f"{row['delta_eef_near_5cm']} | {row['delta_hand_geom_deep_penetration_2cm']} | `{row['failure_label']}` |"
        )

    lines.extend(
        [
            "",
            "## Object Summary",
            "",
            "| object | method | N | physics contact | hand5 | hand12 | deep pen | far>5 | far>12 | leg pen |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in object_summary:
        lines.append(
            f"| {row['group']} | {row['method']} | {row['num_rows']} | {row['mean_hand_object_physics_contact_frac']} | "
            f"{row['mean_hand_geom_near_5cm_frac']} | {row['mean_hand_geom_near_12cm_frac']} | "
            f"{row['mean_hand_geom_deep_penetration_2cm_frac']} | {row['mean_hand_geom_far_gt5cm_frac']} | "
            f"{row['mean_hand_geom_far_gt12cm_frac']} | "
            f"{row['mean_leg_penetration_frac']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- E110 uses aggregate replay fractions from E109; it does not claim raw-contact precision/recall because raw S1 artifacts are not yet wired into this table.",
            "- 2cm near-contact bands are intentionally blank in this aggregate audit because E109 did not export `hand_geom_near_2cm`; E111 should provide exact per-frame 2cm bands.",
            "- `safe_gap_near_contact` means Spider removes deep penetration and keeps broad 12cm proximity, but loses 5cm/physics contact.",
            "- `safety_regression_flag` is separate from the primary contact failure label so safety regressions are not hidden by single-label contact taxonomy.",
            "- E111 should implement per-frame S6 contact alignment using raw masks to turn this aggregate diagnosis into PR/F1 and run-length metrics.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-metrics-tsv", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--case-comparison-tsv", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    metrics = read_csv(args.method_metrics_tsv, delimiter="\t")
    comparison = read_csv(args.case_comparison_tsv, delimiter="\t")
    if not metrics:
        raise FileNotFoundError(f"missing or empty method metrics: {args.method_metrics_tsv}")
    if not comparison:
        raise FileNotFoundError(f"missing or empty comparison table: {args.case_comparison_tsv}")

    tiers = tier_lookup(comparison)
    bands = [band_row(row, tiers) for row in metrics if row.get("method") in METHODS]
    band_map: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in bands:
        band_map[(row["case_id"], row["target_variant_id"], row["run_id"])][row["method"]] = row

    deltas = delta_rows([row for row in metrics if row.get("method") in METHODS], band_map)
    method_summary = summarize(bands)
    object_summary = summarize(bands, group_key="object_key")
    label_counts: dict[str, int] = defaultdict(int)
    for row in deltas:
        label_counts[str(row.get("failure_label", ""))] += 1
    safety_count = sum(1 for row in deltas if row.get("safety_regression_flag") == "true")
    top_physics = sorted(deltas, key=lambda r: finite(r.get("delta_hand_object_physics_contact")))[:10]

    summary = {
        "method_metrics_tsv": str(args.method_metrics_tsv),
        "case_comparison_tsv": str(args.case_comparison_tsv),
        "out_dir": str(args.out_dir),
        "method_rows": len(bands),
        "paired_cases": len(deltas),
        "raw_contact_coverage_status": "missing_artifacts",
        "failure_label_counts": dict(sorted(label_counts.items())),
        "safety_regression_count": safety_count,
        "output_files": {
            "contact_band_metrics": str(args.out_dir / "contact_band_metrics.tsv"),
            "contact_case_delta": str(args.out_dir / "contact_case_delta.tsv"),
            "contact_method_summary": str(args.out_dir / "contact_method_summary.tsv"),
            "contact_object_summary": str(args.out_dir / "contact_object_summary.tsv"),
            "summary_md": str(args.out_dir / "contact_metric_audit_summary.md"),
        },
        "method_summary": method_summary,
        "top_physics_contact_regressions": top_physics,
        "notes": "E110 aggregate audit only; raw-contact PR/F1 deferred to E111 S6 evaluator.",
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(args.out_dir / "contact_band_metrics.tsv", bands, BAND_FIELDS)
    write_tsv(args.out_dir / "contact_case_delta.tsv", deltas, DELTA_FIELDS)
    write_tsv(args.out_dir / "contact_method_summary.tsv", method_summary, SUMMARY_FIELDS)
    write_tsv(args.out_dir / "contact_object_summary.tsv", object_summary, SUMMARY_FIELDS)
    write_json(args.out_dir / "contact_metric_audit_summary.json", summary)
    write_markdown(
        args.out_dir / "contact_metric_audit_summary.md",
        method_summary=method_summary,
        object_summary=object_summary,
        deltas=deltas,
        summary=summary,
    )
    print(f"[E110] wrote contact audit to {args.out_dir}")
    print(f"[E110] method_rows={len(bands)} paired_cases={len(deltas)} raw_contact_coverage=missing_artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
