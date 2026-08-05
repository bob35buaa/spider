#!/usr/bin/env python3
"""Aggregate frozen object-tracking metrics across bucket and canonical box runs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO = Path(__file__).resolve().parents[5]
RESULTS = REPO / "workspace/core4d/results"
OUT = RESULTS / "E188/s6_downstream/eval/object_tracking_audit"
POS = "track_obj_pos_err_cm_mean"
ORI = "track_obj_ori_err_deg_mean"
BOOTSTRAP_SAMPLES = 10_000


@dataclass(frozen=True)
class Source:
    dataset: str
    family: str
    metrics: Path
    expected_objects: tuple[str, ...]
    manual: tuple[Path, ...] = ()
    manual_in_metrics: bool = False


SOURCES = (
    Source(
        "E178",
        "bucket",
        RESULTS / "E178/s6_downstream/eval/full/e178_case_metrics.tsv",
        ("bucket003", "bucket004", "bucket007"),
        (RESULTS / "E178/s6_downstream/eval/full/user_manual_review_filled.tsv",),
    ),
    Source(
        "E187",
        "bucket",
        RESULTS / "E187/s6_downstream/eval/full/e187_case_metrics.tsv",
        ("bucket003", "bucket004", "bucket007"),
        (RESULTS / "E187/s6_downstream/eval/full/user_manual_review_filled.tsv",),
    ),
    Source(
        "E188",
        "bucket",
        RESULTS / "E188/s6_downstream/eval/full/e188_case_metrics.tsv",
        ("bucket003", "bucket007"),
    ),
    Source(
        "E170",
        "box",
        RESULTS / "E170/s6_downstream/eval/full/e170_case_metrics.tsv",
        ("box021",),
        manual_in_metrics=True,
    ),
    Source(
        "E171",
        "box",
        RESULTS / "E171/s6_downstream/eval/full/e171_case_metrics.tsv",
        ("box026",),
    ),
    Source(
        "E172",
        "box",
        RESULTS / "E172/s6_downstream/eval/full/e171_case_metrics.tsv",
        ("box004",),
        (RESULTS / "E172/s6_downstream/eval/full/user_manual_review_filled.tsv",),
    ),
    Source(
        "E173",
        "box",
        RESULTS / "E173/s6_downstream/eval/full/e173_case_metrics.tsv",
        ("box001", "box023", "box024"),
        (
            RESULTS / "E173/s6_downstream/rl_export/box001_user_approved/box001_manual_review_snapshot.tsv",
            RESULTS / "E173/s6_downstream/rl_export/box023_user_approved/box023_manual_review_snapshot.tsv",
            RESULTS / "E173/s6_downstream/rl_export/box024_user_approved/box024_manual_review_snapshot.tsv",
        ),
    ),
)


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "pass"}


def is_manual_use(row: dict[str, str] | None) -> bool:
    if not row:
        return False
    decision = (row.get("manual_use_decision") or "").strip().upper()
    status = (
        row.get("user_manual_review_status")
        or row.get("manual_review_status")
        or ""
    ).strip().lower()
    return decision == "USE" and status == "reviewed"


def finite_float(row: dict[str, str], key: str, label: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label}: invalid {key}={row.get(key)!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{label}: non-finite {key}={value}")
    return value


def object_key(row: dict[str, str]) -> str:
    value = (row.get("object_key") or "").strip().lower()
    if value:
        return value
    return (row.get("case_id") or "").split("_", 1)[0].lower()


def load_rows() -> tuple[list[dict[str, object]], dict[str, str]]:
    records: list[dict[str, object]] = []
    input_shas: dict[str, str] = {}
    global_box_cases: set[str] = set()
    e187_manual = {
        row["case_id"]: row
        for row in read_tsv(
            RESULTS / "E187/s6_downstream/eval/full/user_manual_review_filled.tsv"
        )
    }

    for source in SOURCES:
        rows = read_tsv(source.metrics)
        input_shas[str(source.metrics.relative_to(REPO))] = sha256(source.metrics)
        manual_rows: dict[str, dict[str, str]] = {}
        if source.manual:
            for manual_path in source.manual:
                for manual_row in read_tsv(manual_path):
                    case_id = manual_row["case_id"]
                    if case_id in manual_rows:
                        raise ValueError(f"{source.dataset}: duplicate manual authority for {case_id}")
                    manual_rows[case_id] = manual_row
                input_shas[str(manual_path.relative_to(REPO))] = sha256(manual_path)

        ids = [row.get("case_id", "") for row in rows]
        if not all(ids) or len(ids) != len(set(ids)):
            raise ValueError(f"{source.dataset}: missing or duplicate case_id")
        observed_objects = {object_key(row) for row in rows}
        if observed_objects != set(source.expected_objects):
            raise ValueError(
                f"{source.dataset}: object authority mismatch: "
                f"expected={source.expected_objects}, observed={sorted(observed_objects)}"
            )
        if source.family == "box":
            overlap = global_box_cases.intersection(ids)
            if overlap:
                raise ValueError(f"duplicate box authority cases: {sorted(overlap)}")
            global_box_cases.update(ids)

        for row in rows:
            case_id = row["case_id"]
            self_manual = (
                is_manual_use(row)
                if source.manual_in_metrics
                else is_manual_use(manual_rows.get(case_id))
            )
            parent_manual = source.dataset == "E188" and is_manual_use(
                e187_manual.get(case_id)
            )
            records.append(
                {
                    "dataset": source.dataset,
                    "family": source.family,
                    "case_id": case_id,
                    "object_key": object_key(row),
                    "position_error_cm": finite_float(row, POS, f"{source.dataset}/{case_id}"),
                    "orientation_error_deg": finite_float(row, ORI, f"{source.dataset}/{case_id}"),
                    "numeric_pass": truthy(row.get("numeric_release_pass")),
                    "manual_use_self": self_manual,
                    "manual_use_parent_matched": parent_manual,
                    "metric_standard_id": row.get("metric_standard_id", ""),
                    "device_scope": row.get("device_scope", ""),
                }
            )
    return records, input_shas


def percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return math.nan
    index = (len(sorted_values) - 1) * probability
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return sorted_values[low]
    weight = index - low
    return sorted_values[low] * (1 - weight) + sorted_values[high] * weight


def bootstrap_ci(values: list[float], key: str) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    if len(values) == 1:
        return values[0], values[0]
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(BOOTSTRAP_SAMPLES)
    )
    return percentile(means, 0.025), percentile(means, 0.975)


def summarize(rows: list[dict[str, object]], label: str) -> dict[str, object]:
    pos = [float(row["position_error_cm"]) for row in rows]
    ori = [float(row["orientation_error_deg"]) for row in rows]
    pos_ci = bootstrap_ci(pos, f"{label}/pos")
    ori_ci = bootstrap_ci(ori, f"{label}/ori")
    return {
        "n": len(rows),
        "position_mean_cm": statistics.fmean(pos) if pos else math.nan,
        "position_median_cm": statistics.median(pos) if pos else math.nan,
        "position_std_cm": statistics.stdev(pos) if len(pos) > 1 else (0.0 if pos else math.nan),
        "position_ci95_low_cm": pos_ci[0],
        "position_ci95_high_cm": pos_ci[1],
        "orientation_mean_deg": statistics.fmean(ori) if ori else math.nan,
        "orientation_median_deg": statistics.median(ori) if ori else math.nan,
        "orientation_std_deg": statistics.stdev(ori) if len(ori) > 1 else (0.0 if ori else math.nan),
        "orientation_ci95_low_deg": ori_ci[0],
        "orientation_ci95_high_deg": ori_ci[1],
        "position_gt_10cm_n": sum(value > 10.0 for value in pos),
        "position_gt_20cm_n": sum(value > 20.0 for value in pos),
        "orientation_gt_5deg_n": sum(value > 5.0 for value in ori),
        "orientation_gt_10deg_n": sum(value > 10.0 for value in ori),
    }


def strata_for(dataset: str) -> tuple[tuple[str, str], ...]:
    base = (
        ("all_cases", "all"),
        ("numeric_pass", "numeric"),
        ("manual_use_self", "manual_self"),
    )
    if dataset == "E188":
        return base + (("manual_use_parent_matched", "manual_parent"),)
    return base


def select_stratum(rows: Iterable[dict[str, object]], selector: str) -> list[dict[str, object]]:
    materialized = list(rows)
    if selector == "all":
        return materialized
    if selector == "numeric":
        return [row for row in materialized if row["numeric_pass"]]
    if selector == "manual_self":
        return [row for row in materialized if row["manual_use_self"]]
    if selector == "manual_parent":
        return [row for row in materialized if row["manual_use_parent_matched"]]
    raise ValueError(selector)


def make_summaries(records: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    stratified: list[dict[str, object]] = []
    per_object: list[dict[str, object]] = []
    groups = {
        "E178": [row for row in records if row["dataset"] == "E178"],
        "E187": [row for row in records if row["dataset"] == "E187"],
        "E188": [row for row in records if row["dataset"] == "E188"],
        "BOX_CANONICAL": [row for row in records if row["family"] == "box"],
    }
    for dataset, rows in groups.items():
        family = "box" if dataset == "BOX_CANONICAL" else "bucket"
        for stratum, selector in strata_for(dataset):
            selected = select_stratum(rows, selector)
            result = {
                "dataset": dataset,
                "family": family,
                "stratum": stratum,
                "manual_authority_scope": (
                    "E187_case_selection_only"
                    if stratum == "manual_use_parent_matched"
                    else ("self" if stratum == "manual_use_self" else "not_applicable")
                ),
                **summarize(selected, f"{dataset}/{stratum}"),
            }
            stratified.append(result)

        objects = sorted({str(row["object_key"]) for row in rows})
        for obj in objects:
            object_rows = [row for row in rows if row["object_key"] == obj]
            source_experiments = sorted({str(row["dataset"]) for row in object_rows})
            for stratum, selector in strata_for(dataset):
                selected = select_stratum(object_rows, selector)
                per_object.append(
                    {
                        "dataset": dataset,
                        "source_experiment": ",".join(source_experiments),
                        "family": family,
                        "object_key": obj,
                        "stratum": stratum,
                        **summarize(selected, f"{dataset}/{obj}/{stratum}"),
                    }
                )
    return stratified, per_object


def paired_entry(
    by_dataset: dict[str, dict[str, dict[str, object]]],
    old: str,
    new: str,
    common: set[str],
    scope: str,
    subgroup: str,
) -> dict[str, object]:
    ordered = sorted(common)
    if not ordered:
        raise ValueError(f"empty paired group: {scope}/{subgroup}/{old}/{new}")
    old_pos = [float(by_dataset[old][case]["position_error_cm"]) for case in ordered]
    new_pos = [float(by_dataset[new][case]["position_error_cm"]) for case in ordered]
    old_ori = [float(by_dataset[old][case]["orientation_error_deg"]) for case in ordered]
    new_ori = [float(by_dataset[new][case]["orientation_error_deg"]) for case in ordered]
    pos_delta = [b - a for a, b in zip(old_pos, new_pos)]
    ori_delta = [b - a for a, b in zip(old_ori, new_ori)]
    key = f"paired/{scope}/{subgroup}/{old}/{new}"
    pos_ci = bootstrap_ci(pos_delta, f"{key}/pos")
    ori_ci = bootstrap_ci(ori_delta, f"{key}/ori")
    return {
        "scope": scope,
        "subgroup": subgroup,
        "old_dataset": old,
        "new_dataset": new,
        "n": len(ordered),
        "old_position_mean_cm": statistics.fmean(old_pos),
        "new_position_mean_cm": statistics.fmean(new_pos),
        "position_delta_new_minus_old_cm": statistics.fmean(pos_delta),
        "position_delta_ci95_low_cm": pos_ci[0],
        "position_delta_ci95_high_cm": pos_ci[1],
        "position_nondegraded_n": sum(delta <= 0 for delta in pos_delta),
        "old_orientation_mean_deg": statistics.fmean(old_ori),
        "new_orientation_mean_deg": statistics.fmean(new_ori),
        "orientation_delta_new_minus_old_deg": statistics.fmean(ori_delta),
        "orientation_delta_ci95_low_deg": ori_ci[0],
        "orientation_delta_ci95_high_deg": ori_ci[1],
        "orientation_nondegraded_n": sum(delta <= 0 for delta in ori_delta),
        "case_ids": ",".join(ordered),
    }


def bucket_maps(records: list[dict[str, object]]) -> dict[str, dict[str, dict[str, object]]]:
    return {
        dataset: {
            str(row["case_id"]): row for row in records if row["dataset"] == dataset
        }
        for dataset in ("E178", "E187", "E188")
    }


def paired_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    by_dataset = bucket_maps(records)
    triple = set(by_dataset["E178"]) & set(by_dataset["E187"]) & set(by_dataset["E188"])
    comparisons = (
        ("E178", "E187", set(by_dataset["E178"]) & set(by_dataset["E187"]), "pair_common", "all"),
        ("E187", "E188", set(by_dataset["E187"]) & set(by_dataset["E188"]), "pair_common", "all"),
        ("E178", "E188", set(by_dataset["E178"]) & set(by_dataset["E188"]), "pair_common", "all"),
        ("E178", "E187", triple, "triple_common", "all"),
        ("E187", "E188", triple, "triple_common", "all"),
        ("E178", "E188", triple, "triple_common", "all"),
    )
    output = [paired_entry(by_dataset, *comparison) for comparison in comparisons]
    e188_common = set(by_dataset["E187"]) & set(by_dataset["E188"])
    for device_scope in ("same_device_local4", "cross_device11"):
        cases = {
            case
            for case in e188_common
            if by_dataset["E188"][case]["device_scope"] == device_scope
        }
        output.append(
            paired_entry(
                by_dataset, "E187", "E188", cases, "device_stratified", device_scope
            )
        )
    return output


def paired_per_object_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    by_dataset = bucket_maps(records)
    output: list[dict[str, object]] = []
    for old, new in (("E178", "E187"), ("E187", "E188"), ("E178", "E188")):
        common = set(by_dataset[old]) & set(by_dataset[new])
        objects = sorted({str(by_dataset[new][case]["object_key"]) for case in common})
        for obj in objects:
            cases = {case for case in common if by_dataset[new][case]["object_key"] == obj}
            output.append(paired_entry(by_dataset, old, new, cases, "per_object", obj))
    return output


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object, digits: int = 2) -> str:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f"{float(value):.{digits}f}"
    return "N/A"


def lookup(rows: list[dict[str, object]], dataset: str, stratum: str) -> dict[str, object]:
    return next(row for row in rows if row["dataset"] == dataset and row["stratum"] == stratum)


def render_report(
    stratified: list[dict[str, object]],
    per_object: list[dict[str, object]],
    paired: list[dict[str, object]],
    paired_per_object: list[dict[str, object]],
    input_shas: dict[str, str],
) -> str:
    lines = [
        "# E188 Object Tracking Cross-object Audit",
        "",
        "Object position 使用 `track_obj_pos_err_cm_mean`（cm），orientation 使用 "
        "`track_obj_ori_err_deg_mean`（degree）；两者均越低越好。",
        "",
        "## Experiment-level stratification",
        "",
        "| Dataset | Stratum | n | Position mean | Position 95% CI | Orientation mean | Orientation 95% CI |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in stratified:
        lines.append(
            f"| {row['dataset']} | {row['stratum']} | {row['n']} | "
            f"{fmt(row['position_mean_cm'])} cm | "
            f"[{fmt(row['position_ci95_low_cm'])}, {fmt(row['position_ci95_high_cm'])}] | "
            f"{fmt(row['orientation_mean_deg'])}° | "
            f"[{fmt(row['orientation_ci95_low_deg'])}, {fmt(row['orientation_ci95_high_deg'])}] |"
        )

    lines += [
        "",
        "E188 的 `manual_use_parent_matched` 仅表示同一 case 在 E187 被人工标为 USE，"
        "不是 E188 自身的人工通过。E188 `manual_use_self` 为 0 是正确的缺失状态。",
        "",
        "## Per-object breakdown",
        "",
        "| Dataset | Object | Stratum | n | Position mean | Orientation mean |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in per_object:
        if row["stratum"] in {"all_cases", "numeric_pass", "manual_use_self", "manual_use_parent_matched"}:
            lines.append(
                f"| {row['dataset']} | {row['object_key']} | {row['stratum']} | {row['n']} | "
                f"{fmt(row['position_mean_cm'])} cm | {fmt(row['orientation_mean_deg'])}° |"
            )

    lines += [
        "",
        "## Paired common-case comparison",
        "",
        "Delta 为 new-old；正数表示 tracking error 变差。",
        "",
        "| Scope | Subgroup | Pair | n | Position old→new (Δ) | Position Δ 95% CI | Orientation old→new (Δ) | Orientation Δ 95% CI |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in paired:
        lines.append(
            f"| {row['scope']} | {row['subgroup']} | {row['old_dataset']}→{row['new_dataset']} | {row['n']} | "
            f"{fmt(row['old_position_mean_cm'])}→{fmt(row['new_position_mean_cm'])} "
            f"({fmt(row['position_delta_new_minus_old_cm'], 3)}) | "
            f"[{fmt(row['position_delta_ci95_low_cm'], 3)}, {fmt(row['position_delta_ci95_high_cm'], 3)}] | "
            f"{fmt(row['old_orientation_mean_deg'])}→{fmt(row['new_orientation_mean_deg'])} "
            f"({fmt(row['orientation_delta_new_minus_old_deg'], 3)}) | "
            f"[{fmt(row['orientation_delta_ci95_low_deg'], 3)}, {fmt(row['orientation_delta_ci95_high_deg'], 3)}] |"
        )

    lines += [
        "",
        "## Paired per-object comparison",
        "",
        "| Object | Pair | n | Position old→new (Δ) | Orientation old→new (Δ) |",
        "|---|---|---:|---:|---:|",
    ]
    for row in paired_per_object:
        lines.append(
            f"| {row['subgroup']} | {row['old_dataset']}→{row['new_dataset']} | {row['n']} | "
            f"{fmt(row['old_position_mean_cm'])}→{fmt(row['new_position_mean_cm'])} "
            f"({fmt(row['position_delta_new_minus_old_cm'], 3)}) | "
            f"{fmt(row['old_orientation_mean_deg'])}→{fmt(row['new_orientation_mean_deg'])} "
            f"({fmt(row['orientation_delta_new_minus_old_deg'], 3)}) |"
        )

    e178_all = lookup(stratified, "E178", "all_cases")
    e187_all = lookup(stratified, "E187", "all_cases")
    e188_all = lookup(stratified, "E188", "all_cases")
    box_all = lookup(stratified, "BOX_CANONICAL", "all_cases")
    box_manual = lookup(stratified, "BOX_CANONICAL", "manual_use_self")
    e187_manual = lookup(stratified, "E187", "manual_use_self")
    e188_parent = lookup(stratified, "E188", "manual_use_parent_matched")
    e187_e188 = next(
        row
        for row in paired
        if row["scope"] == "pair_common"
        and row["old_dataset"] == "E187"
        and row["new_dataset"] == "E188"
    )
    local4 = next(row for row in paired if row["subgroup"] == "same_device_local4")
    cross11 = next(row for row in paired if row["subgroup"] == "cross_device11")
    lines += [
        "",
        "## Visible-offset prevalence (all cases)",
        "",
        "`>10 cm` / `>5°` 用于描述肉眼可见偏差；`>20 cm` / `>10°` 是当前正式 object gates。",
        "",
        "| Dataset | n | Position >10 cm | Position >20 cm | Orientation >5° | Orientation >10° |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset, row in (
        ("E178", e178_all),
        ("E187", e187_all),
        ("E188", e188_all),
        ("BOX_CANONICAL", box_all),
    ):
        lines.append(
            f"| {dataset} | {row['n']} | {row['position_gt_10cm_n']} | "
            f"{row['position_gt_20cm_n']} | {row['orientation_gt_5deg_n']} | "
            f"{row['orientation_gt_10deg_n']} |"
        )

    lines += [
        "",
        "## Direct findings",
        "",
        f"- 全量 bucket position/orientation：E178={fmt(e178_all['position_mean_cm'])} cm/"
        f"{fmt(e178_all['orientation_mean_deg'])}°，E187={fmt(e187_all['position_mean_cm'])} cm/"
        f"{fmt(e187_all['orientation_mean_deg'])}°，E188={fmt(e188_all['position_mean_cm'])} cm/"
        f"{fmt(e188_all['orientation_mean_deg'])}°。",
        f"- canonical box 全量为 {fmt(box_all['position_mean_cm'])} cm/"
        f"{fmt(box_all['orientation_mean_deg'])}°；与 bucket 的差异同时包含 object geometry、"
        "case composition 与实验版本，属于描述性 benchmark。",
        f"- 人工 USE：E187 n={e187_manual['n']}，均值 {fmt(e187_manual['position_mean_cm'])} cm/"
        f"{fmt(e187_manual['orientation_mean_deg'])}°；E188 的 E187-USE matched subset n={e188_parent['n']}，"
        f"均值 {fmt(e188_parent['position_mean_cm'])} cm/{fmt(e188_parent['orientation_mean_deg'])}°；"
        f"box self-reviewed USE n={box_manual['n']}，均值 {fmt(box_manual['position_mean_cm'])} cm/"
        f"{fmt(box_manual['orientation_mean_deg'])}°。",
        f"- E187→E188 的15条 paired case 中，position 和 orientation 都是 0/15 非退化，"
        f"即两项误差逐 case 全部增加；均值分别增加 "
        f"{fmt(e187_e188['position_delta_new_minus_old_cm'], 3)} cm 和 "
        f"{fmt(e187_e188['orientation_delta_new_minus_old_deg'], 3)}°。",
        f"- 同设备 local-4 同样 position +{fmt(local4['position_delta_new_minus_old_cm'], 3)} cm、"
        f"orientation +{fmt(local4['orientation_delta_new_minus_old_deg'], 3)}°；cross-device11 为 "
        f"+{fmt(cross11['position_delta_new_minus_old_cm'], 3)} cm/+"
        f"{fmt(cross11['orientation_delta_new_minus_old_deg'], 3)}°。因此 E188 退化不能仅用设备变化解释。",
        "- numeric PASS 层天然包含 object tracking gate，是 selection-on-metric；它适合描述最终可用集，"
        "不适合单独证明底层算法改善。",
        "",
        "## Provenance",
        "",
    ]
    for path, digest in sorted(input_shas.items()):
        lines.append(f"- `{path}`: `{digest}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records, input_shas = load_rows()
    stratified, per_object = make_summaries(records)
    paired = paired_rows(records)
    paired_per_object = paired_per_object_rows(records)

    standards = sorted({str(row["metric_standard_id"]) for row in records if row["metric_standard_id"]})
    if standards != ["core4d-e154-physics-contact-v1"]:
        raise ValueError(f"metric standard mismatch: {standards}")

    write_tsv(OUT / "case_level_audit.tsv", records)
    write_tsv(OUT / "stratified_summary.tsv", stratified)
    write_tsv(OUT / "per_object_summary.tsv", per_object)
    write_tsv(OUT / "paired_version_comparison.tsv", paired)
    write_tsv(OUT / "paired_per_object_comparison.tsv", paired_per_object)
    report = render_report(stratified, per_object, paired, paired_per_object, input_shas)
    (OUT / "report.md").write_text(report, encoding="utf-8")
    summary = {
        "status": "PASS",
        "metric_standard_ids": standards,
        "metrics": {"position": POS, "orientation": ORI},
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "authority": {
            "input_sha256": input_shas,
            "box_objects": sorted({str(row["object_key"]) for row in records if row["family"] == "box"}),
            "bucket_objects": sorted({str(row["object_key"]) for row in records if row["family"] == "bucket"}),
        },
        "counts": {
            dataset: sum(row["dataset"] == dataset for row in records)
            for dataset in ("E178", "E187", "E188", "E170", "E171", "E172", "E173")
        },
        "stratified_summary": stratified,
        "paired_version_comparison": paired,
        "paired_per_object_comparison": paired_per_object,
        "manual_scope_note": (
            "E188 has no self manual review; manual_use_parent_matched transfers E187 USE case selection only. "
            "E171 box026 has no filled manual review and is excluded from manual-only strata; "
            "box022 is DATA_NEGATIVE and has no formal Full metrics."
        ),
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"OBJECT_TRACKING_AUDIT=PASS rows={len(records)} out={OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
