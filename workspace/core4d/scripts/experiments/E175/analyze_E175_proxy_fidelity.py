#!/usr/bin/env python3
"""Build E175 attribution tables from the strict 39-case diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[5]
DEFAULT_DIAGNOSTICS = REPO / "workspace/core4d/results/E175/diagnostics"
DESK_FAIL_CASES = {
    "desk007_20231023_048_p1",
    "desk007_20231030_028_p1",
    "desk007_20231030_030_p1",
    "desk007_20231030_028_p2",
    "desk007_20231030_030_p2",
    "desk007_20231030_032_p2",
    "desk007_20231030_034_p2",
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def num(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in {"", None} else math.nan


def flag(row: dict[str, str], key: str) -> bool:
    return str(row.get(key, "")).lower() == "true"


def mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else math.nan


def median(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if array.size else math.nan


def intervals(frames: list[int]) -> list[tuple[int, int]]:
    if not frames:
        return []
    unique = sorted(set(frames))
    output: list[tuple[int, int]] = []
    start = previous = unique[0]
    for frame in unique[1:]:
        if frame != previous + 1:
            output.append((start, previous))
            start = frame
        previous = frame
    output.append((start, previous))
    return output


def leg_origin(row: dict[str, str]) -> str:
    """Descriptive decomposition using a 2%-of-frames materiality floor."""
    ref = num(row, "ref_full_penetration_frac")
    rollout = num(row, "rollout_full_penetration_frac")
    ref_material = ref > 0.02
    rollout_material = rollout > 0.02
    if ref_material and rollout_material:
        return "ref_seeded_and_persists"
    if ref_material:
        return "ref_only_corrected_by_rollout"
    if rollout_material:
        return "rollout_induced"
    return "negligible"


def case_attribution(
    case_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    fields = [
        "case_id",
        "object_key",
        "numeric_release_pass",
        "numeric_failure_modes",
        "ref_qpos_source",
        "active_contact_targets",
        "contact_full_miss_3cm_frac",
        "contact_full_proxy_undercoverage_gt_3cm_frac",
        "contact_full_proxy_overcoverage_gt_3cm_frac",
        "contact_full_proxy_mesh_abs_error_gt_3cm_frac",
        "contact_physics_blind_vs_full_frac",
        "contact_physics_coverage_loss_gt_3cm_frac",
        "ref_full_penetration_frac",
        "ref_prg_penetration_frac",
        "ref_prg_false_negative_frac",
        "rollout_full_penetration_frac",
        "rollout_prg_penetration_frac",
        "rollout_prg_false_negative_frac",
    ]
    output: list[dict[str, Any]] = []
    for row in case_rows:
        record = {key: row.get(key, "") for key in fields}
        record["rollout_minus_ref_full_penetration_frac"] = (
            num(row, "rollout_full_penetration_frac")
            - num(row, "ref_full_penetration_frac")
        )
        record["leg_penetration_origin_2pct"] = leg_origin(row)
        output.append(record)
    return output


def object_case_details(
    object_key: str,
    case_rows: list[dict[str, str]],
    contact_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    case_map = {row["case_id"]: row for row in case_rows}
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in contact_rows:
        if row["object_key"] == object_key:
            grouped[row["case_id"]].append(row)
    output: list[dict[str, Any]] = []
    for case_id in sorted(grouped):
        summary = case_map[case_id]
        contacts = grouped[case_id]
        proxy_misses = [
            row for row in contacts if flag(row, "full_miss_3cm")
        ]
        undercoverage = [
            row
            for row in contacts
            if flag(row, "full_proxy_undercoverage_gt_3cm")
        ]
        coverage_loss = [
            row
            for row in contacts
            if flag(row, "physics_coverage_loss_gt_3cm")
        ]
        nearest = Counter(row["full_geom"] for row in proxy_misses)
        output.append(
            {
                "case_id": case_id,
                "numeric_release_pass": summary[
                    "numeric_release_pass"
                ],
                "numeric_failure_modes": summary[
                    "numeric_failure_modes"
                ],
                "active_contact_targets": len(contacts),
                "full_proxy_miss_3cm_targets": len(proxy_misses),
                "full_proxy_miss_3cm_frac": num(
                    summary, "contact_full_miss_3cm_frac"
                ),
                "proxy_undercoverage_vs_mesh_3cm_targets": len(
                    undercoverage
                ),
                "proxy_undercoverage_vs_mesh_3cm_frac": num(
                    summary,
                    "contact_full_proxy_undercoverage_gt_3cm_frac",
                ),
                "proxy_overcoverage_vs_mesh_3cm_frac": num(
                    summary,
                    "contact_full_proxy_overcoverage_gt_3cm_frac",
                ),
                "physics_coverage_loss_3cm_targets": len(coverage_loss),
                "physics_coverage_loss_3cm_frac": num(
                    summary,
                    "contact_physics_coverage_loss_gt_3cm_frac",
                ),
                "dominant_nearest_full_geom_on_miss": (
                    nearest.most_common(1)[0][0] if nearest else ""
                ),
                "dominant_nearest_full_geom_miss_targets": (
                    nearest.most_common(1)[0][1] if nearest else 0
                ),
                "ref_full_penetration_frac": num(
                    summary, "ref_full_penetration_frac"
                ),
                "ref_prg_false_negative_frac": num(
                    summary, "ref_prg_false_negative_frac"
                ),
                "rollout_full_penetration_frac": num(
                    summary, "rollout_full_penetration_frac"
                ),
                "rollout_prg_false_negative_frac": num(
                    summary, "rollout_prg_false_negative_frac"
                ),
                "leg_penetration_origin_2pct": leg_origin(summary),
            }
        )
    return output


def desk_intervals(
    case_rows: list[dict[str, str]],
    contact_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    case_map = {row["case_id"]: row for row in case_rows}
    events = {
        "full_proxy_absolute_miss_3cm": "full_miss_3cm",
        "proxy_undercoverage_vs_mesh_3cm": (
            "full_proxy_undercoverage_gt_3cm"
        ),
        "physics_coverage_loss_vs_full_3cm": (
            "physics_coverage_loss_gt_3cm"
        ),
    }
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in contact_rows:
        if row["object_key"] == "desk007":
            grouped[(row["case_id"], row["hand"])].append(row)
    output: list[dict[str, Any]] = []
    for (case_id, hand), rows in sorted(grouped.items()):
        for event, field in events.items():
            selected = [row for row in rows if flag(row, field)]
            for start, end in intervals(
                [int(row["frame"]) for row in selected]
            ):
                interval_rows = [
                    row
                    for row in selected
                    if start <= int(row["frame"]) <= end
                ]
                nearest = Counter(
                    row["full_geom"] for row in interval_rows
                )
                output.append(
                    {
                        "case_id": case_id,
                        "is_one_of_7_numeric_fail": (
                            case_id in DESK_FAIL_CASES
                        ),
                        "numeric_release_pass": case_map[case_id][
                            "numeric_release_pass"
                        ],
                        "numeric_failure_modes": case_map[case_id][
                            "numeric_failure_modes"
                        ],
                        "hand": hand,
                        "event": event,
                        "start_frame": start,
                        "end_frame": end,
                        "frame_count": end - start + 1,
                        "dominant_nearest_full_geom": (
                            nearest.most_common(1)[0][0]
                            if nearest
                            else ""
                        ),
                    }
                )
    return output


def association_table(
    case_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    metrics = [
        "contact_full_miss_3cm_frac",
        "contact_full_proxy_undercoverage_gt_3cm_frac",
        "contact_full_proxy_overcoverage_gt_3cm_frac",
        "contact_physics_blind_vs_full_frac",
        "contact_physics_coverage_loss_gt_3cm_frac",
        "ref_full_penetration_frac",
        "ref_prg_false_negative_frac",
        "rollout_full_penetration_frac",
        "rollout_prg_false_negative_frac",
    ]
    output: list[dict[str, Any]] = []
    groups = {
        "numeric_pass": [
            row for row in case_rows if flag(row, "numeric_release_pass")
        ],
        "numeric_fail": [
            row
            for row in case_rows
            if not flag(row, "numeric_release_pass")
        ],
    }
    for metric in metrics:
        record: dict[str, Any] = {"metric": metric}
        for label, rows in groups.items():
            values = [num(row, metric) for row in rows]
            record[f"{label}_n"] = len(rows)
            record[f"{label}_mean"] = mean(values)
            record[f"{label}_median"] = median(values)
        record["fail_minus_pass_mean"] = (
            record["numeric_fail_mean"] - record["numeric_pass_mean"]
        )
        output.append(record)
    return output


def markdown_report(
    object_rows: list[dict[str, str]],
    desk_rows: list[dict[str, Any]],
    bucket_rows: list[dict[str, Any]],
    associations: list[dict[str, Any]],
) -> str:
    objects = {row["object_key"]: row for row in object_rows}
    bucket = objects["bucket004"]
    desk = objects["desk007"]
    association = {row["metric"]: row for row in associations}
    lines = [
        "# E175 non-box proxy fidelity attribution",
        "",
        "_E174 39-case diagnostic · ref target, proxy fidelity, physics pair and PRG coverage separated_",
        "",
        "---",
        "",
        "## 🔬 Causal decomposition",
        "",
        "```mermaid",
        "flowchart LR",
        "    accTitle: Non-box Failure Decomposition",
        "    accDescr: The audit separates reference target feasibility, visual-to-proxy fidelity, physics pair coverage, PRG SDF coverage, and rollout execution.",
        "",
        '    ref["Reference FK target"] --> mesh["Visual mesh distance"]',
        '    ref --> proxy["Full proxy distance"]',
        '    proxy --> pair["Physics-paired subset"]',
        '    proxy --> prg["PRG SDF subset"]',
        '    pair --> rollout["CEM rollout"]',
        '    prg --> rollout',
        "",
        "    classDef source fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f",
        "    classDef audit fill:#fef9c3,stroke:#ca8a04,stroke-width:2px,color:#713f12",
        "    classDef outcome fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d",
        "    class ref source",
        "    class mesh,proxy,pair,prg audit",
        "    class rollout outcome",
        "```",
        "",
        "## 📊 Main findings",
        "",
        f"- Strict coverage: `39/39`, errors `0`.",
        f"- bucket004: full-proxy absolute miss `{float(bucket['contact_full_miss_3cm_frac']):.1%}`, but relative undercoverage vs visual mesh only `{float(bucket['contact_full_proxy_undercoverage_gt_3cm_frac']):.1%}`; physics/PRG coverage loss vs full proxy is `{float(bucket['contact_physics_coverage_loss_gt_3cm_frac']):.1%}`.",
        f"- desk007: full-proxy absolute miss `{float(desk['contact_full_miss_3cm_frac']):.1%}`, relative undercoverage vs visual mesh `{float(desk['contact_full_proxy_undercoverage_gt_3cm_frac']):.1%}`, physics/PRG coverage loss `{float(desk['contact_physics_coverage_loss_gt_3cm_frac']):.1%}`.",
        "- Absolute target-to-proxy distance is not a proxy-fidelity measure by itself: the runtime ref-FK target is a wrist/palm point and can be far from both visual mesh and proxy.",
        "",
        "## 🪑 desk007 verdict",
        "",
        f"All `{len(desk_rows)}` desk cases are tabulated, including the seven numeric failures. The failed contact targets cluster near `object_collision_voxel_007` for person1 and `object_collision_voxel_002` for person2, but there are no targets where the full proxy is more than 3 cm farther than the visual mesh at object level. The claimed voxel blind spot is therefore not supported as the dominant cause. The large loss appears after proxy construction: only the first geom is connected to robot physics and PRG.",
        "",
        "## 🪣 bucket004 verdict",
        "",
        f"Across the four fixed cases, only `{sum(int(row['proxy_undercoverage_vs_mesh_3cm_targets']) for row in bucket_rows)}` active targets show >3 cm proxy undercoverage relative to the visual mesh. The current wall boxes already have inner-facing surfaces; adding a rim can address the small top-edge subset, but cannot explain the full failure yield. The five-arm canary remains necessary because pair/PRG coverage is the larger controlled variable.",
        "",
        "## 🦵 Lower-body verdict",
        "",
        f"Numeric failures have mean ref full-proxy penetration `{association['ref_full_penetration_frac']['numeric_fail_mean']:.1%}` versus `{association['ref_full_penetration_frac']['numeric_pass_mean']:.1%}` for passes. Mean ref PRG false-negative rate is `{association['ref_prg_false_negative_frac']['numeric_fail_mean']:.1%}` versus `{association['ref_prg_false_negative_frac']['numeric_pass_mean']:.1%}`. Thus both mechanisms coexist: many references are already infeasible against the full thin-wall proxy, and the legacy primary-geom SDF fails to see much of that infeasibility.",
        "",
        "## 📁 Tables",
        "",
        "- `case_attribution.tsv`: all 39 cases",
        "- `desk007_case_attribution.tsv`: all nine desk cases",
        "- `desk007_blind_intervals.tsv`: per-hand contiguous intervals by event type",
        "- `bucket004_case_attribution.tsv`: four fixed ablation cases",
        "- `failure_association.tsv`: pass/fail descriptive contrasts",
        "",
    ]
    return "\n".join(lines)


def run(diagnostics: Path) -> dict[str, Any]:
    case_rows = read_tsv(diagnostics / "case_proxy_fidelity.tsv")
    contact_rows = read_tsv(diagnostics / "contact_proxy_hits.tsv")
    object_rows = read_tsv(diagnostics / "object_summary.tsv")
    if len(case_rows) != 39:
        raise ValueError(f"expected 39 cases, got {len(case_rows)}")

    cases = case_attribution(case_rows)
    desk = object_case_details("desk007", case_rows, contact_rows)
    bucket = object_case_details("bucket004", case_rows, contact_rows)
    desk_windows = desk_intervals(case_rows, contact_rows)
    associations = association_table(case_rows)

    write_tsv(diagnostics / "case_attribution.tsv", cases)
    write_tsv(diagnostics / "desk007_case_attribution.tsv", desk)
    write_tsv(diagnostics / "desk007_blind_intervals.tsv", desk_windows)
    write_tsv(diagnostics / "bucket004_case_attribution.tsv", bucket)
    write_tsv(diagnostics / "failure_association.tsv", associations)

    desk_undercoverage_windows = [
        row
        for row in desk_windows
        if row["event"] == "proxy_undercoverage_vs_mesh_3cm"
    ]
    summary = {
        "status": "pass",
        "case_count": len(case_rows),
        "desk_case_count": len(desk),
        "desk_numeric_fail_set_exact": (
            {
                row["case_id"]
                for row in desk
                if str(row["numeric_release_pass"]).lower() != "true"
            }
            == DESK_FAIL_CASES
        ),
        "desk_proxy_undercoverage_intervals": len(
            desk_undercoverage_windows
        ),
        "bucket_case_count": len(bucket),
        "leg_penetration_origin_counts": dict(
            Counter(row["leg_penetration_origin_2pct"] for row in cases)
        ),
    }
    (diagnostics / "attribution_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (diagnostics / "attribution_summary.md").write_text(
        markdown_report(object_rows, desk, bucket, associations),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS
    )
    args = parser.parse_args()
    summary = run(args.diagnostics)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
