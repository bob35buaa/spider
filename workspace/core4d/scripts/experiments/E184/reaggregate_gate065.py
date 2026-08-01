#!/usr/bin/env python3
"""Re-aggregate frozen E183 static-P confusions at 0.65 without re-querying SDF."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[5]
EXPERIMENT_ID = "E184"
E183_ROOT = REPO_ROOT / "workspace/core4d/results/E183/full27_static_p"
SOURCE_CASE_PATH = E183_ROOT / "case_candidate_metrics.tsv"
SOURCE_SUMMARY_PATH = E183_ROOT / "candidate_summary.tsv"
EXPECTED_CASE_SHA256 = (
    "28a3c8118fe10767856b5d6bcdb93430dc9ba629d7f4bb58eead9924168a949c"
)
EXPECTED_SUMMARY_SHA256 = (
    "08b8927ccc0513b8ff6dc20bf36f83526c9cf265472fd6bf441107aa21766093"
)
OUTPUT_ROOT = REPO_ROOT / "workspace/core4d/results/E184/static_p_gate065"
PROTOCOL_PATH_NAME = "protocol_manifest.json"
OBJECT_KEYS = ("bucket003", "bucket004", "bucket007")
OLD_FLOOR = 0.70
NEW_FLOOR = 0.65
EXPECTED_CASE_ROWS = 540
EXPECTED_CANDIDATES = 60
EXPECTED_CASES = 27
EXPECTED_POSES = 14_542
EXPECTED_ORACLE_CONTACTS = 2_198


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write stable indented JSON atomically."""
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def atomic_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a non-empty list of like-shaped mappings as TSV atomically."""
    if not rows:
        raise ValueError("cannot write an empty TSV")
    fields = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, delimiter="\t", fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _artifact(path: Path) -> dict[str, str]:
    absolute = path.absolute()
    try:
        display_path = str(absolute.relative_to(REPO_ROOT))
    except ValueError:
        display_path = str(absolute)
    return {
        "path": display_path,
        "sha256": sha256_file(path),
    }


def _float(value: float) -> str:
    return f"{value:.9f}"


def _status(value: bool) -> str:
    return "PASS" if value else "FAIL"


def phantom_limit(true_positive: int, floor: float) -> int:
    """Largest integer phantom count satisfying TP/(TP+phantom) >= floor."""
    if true_positive < 0 or not 0.0 < floor <= 1.0:
        raise ValueError("invalid TP or floor")
    return math.floor(true_positive * (1.0 - floor) / floor + 1e-12)


def metrics_from_confusion(
    true_positive: int,
    phantom: int,
    missed: int,
) -> dict[str, float | int | bool]:
    """Recompute precision/recall and identify a positive-oracle case."""
    if min(true_positive, phantom, missed) < 0:
        raise ValueError("confusion counts cannot be negative")
    predicted = true_positive + phantom
    oracle = true_positive + missed
    return {
        "oracle_contact_count": oracle,
        "oracle_positive": oracle > 0,
        "precision": true_positive / predicted if predicted else 0.0,
        "recall": true_positive / oracle if oracle else 0.0,
    }


def gate_from_confusion(
    true_positive: int,
    phantom: int,
    missed: int,
    floor: float,
    *,
    zero_aware: bool,
) -> bool:
    """Apply a P/R floor, with an optional explicit no-contact case contract."""
    metrics = metrics_from_confusion(true_positive, phantom, missed)
    if not metrics["oracle_positive"]:
        return phantom == 0 if zero_aware else False
    return bool(metrics["precision"] >= floor and metrics["recall"] >= floor)


def load_sources(
    case_path: Path = SOURCE_CASE_PATH,
    summary_path: Path = SOURCE_SUMMARY_PATH,
    *,
    expected_case_sha: str = EXPECTED_CASE_SHA256,
    expected_summary_sha: str = EXPECTED_SUMMARY_SHA256,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Load the two exact E183 authority tables after digest and closure checks."""
    if sha256_file(case_path) != expected_case_sha:
        raise RuntimeError("E183 case table SHA changed")
    if sha256_file(summary_path) != expected_summary_sha:
        raise RuntimeError("E183 candidate summary SHA changed")
    cases = _read_tsv(case_path)
    summaries = _read_tsv(summary_path)
    if len(cases) != EXPECTED_CASE_ROWS or len(summaries) != EXPECTED_CANDIDATES:
        raise RuntimeError("E183 source row closure changed")
    pairs = {(row["candidate_key"], row["case_id"]) for row in cases}
    candidates = {row["candidate_key"] for row in cases}
    case_ids = {row["case_id"] for row in cases}
    if len(pairs) != len(cases) or len(candidates) != EXPECTED_CANDIDATES:
        raise RuntimeError("E183 case/candidate uniqueness changed")
    if len(case_ids) != EXPECTED_CASES:
        raise RuntimeError("E183 case authority changed")
    if {row["candidate_key"] for row in summaries} != candidates:
        raise RuntimeError("E183 case/summary candidate sets differ")
    return cases, summaries


def protocol_payload() -> dict[str, Any]:
    """Describe the result-independent E184 re-aggregation contract."""
    runner = Path(__file__).resolve()
    wrapper = REPO_ROOT / "workspace/core4d/scripts/eval/wrappers/eval_E184_gate065.sh"
    return {
        "experiment_id": EXPERIMENT_ID,
        "status": "FROZEN",
        "scope": "OFFLINE_REAGGREGATION_ONLY",
        "source_case_table": {
            "path": str(SOURCE_CASE_PATH.absolute().relative_to(REPO_ROOT)),
            "sha256": EXPECTED_CASE_SHA256,
            "rows": EXPECTED_CASE_ROWS,
        },
        "source_candidate_summary": {
            "path": str(SOURCE_SUMMARY_PATH.absolute().relative_to(REPO_ROOT)),
            "sha256": EXPECTED_SUMMARY_SHA256,
            "rows": EXPECTED_CANDIDATES,
        },
        "thresholds": {"baseline": OLD_FLOOR, "sensitivity": NEW_FLOOR},
        "positive_case_gate": "precision>=floor AND recall>=floor",
        "phantom_limit": "floor(TP*(1-floor)/floor)",
        "zero_oracle_gate": "PASS iff phantom==0",
        "macro_contract": "positive-oracle cases only; legacy all-case macro retained separately",
        "all_case_contract": "pooled gate AND positive-only macro gate AND every case gate",
        "isolation": {
            "sdf_queries": 0,
            "coacd_builds": 0,
            "gpu_access": 0,
            "full_cem_runs": 0,
        },
        "runner": _artifact(runner),
        "wrapper": _artifact(wrapper),
    }


def freeze_protocol(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    """Freeze before aggregate outputs exist; validate exact resume otherwise."""
    load_sources()
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / PROTOCOL_PATH_NAME
    payload = protocol_payload()
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError("frozen E184 protocol differs from current contract")
        return existing
    forbidden = [
        output_root / "case_candidate_metrics_gate065.tsv",
        output_root / "candidate_summary_gate065.tsv",
        output_root / "aggregate.json",
        output_root / "threshold_comparison.png",
        output_root / "validation.json",
    ]
    if any(path.exists() for path in forbidden):
        raise RuntimeError("result exists before E184 protocol freeze")
    atomic_json(path, payload)
    return payload


def load_protocol(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    """Require the current runner and source inputs to match the frozen protocol."""
    path = output_root / PROTOCOL_PATH_NAME
    if not path.is_file():
        raise RuntimeError("E184 protocol is not frozen")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload != protocol_payload():
        raise RuntimeError("E184 protocol or implementation changed after freeze")
    load_sources()
    return payload


def _case_output(row: Mapping[str, str]) -> dict[str, str]:
    tp = int(row["true_positive_count"])
    phantom = int(row["phantom_contact_count"])
    missed = int(row["missed_contact_count"])
    metrics = metrics_from_confusion(tp, phantom, missed)
    precision = float(metrics["precision"])
    recall = float(metrics["recall"])
    if abs(float(row["precision"]) - precision) > 1e-9:
        raise RuntimeError("source precision does not match confusion counts")
    if abs(float(row["recall"]) - recall) > 1e-9:
        raise RuntimeError("source recall does not match confusion counts")
    legacy_070 = gate_from_confusion(tp, phantom, missed, OLD_FLOOR, zero_aware=False)
    if _status(legacy_070) != row["gate_status"]:
        raise RuntimeError("source 0.70 case gate regression")
    positive = bool(metrics["oracle_positive"])
    return {
        **row,
        "oracle_contact_count": str(metrics["oracle_contact_count"]),
        "oracle_positive": str(positive).lower(),
        "phantom_limit_070": str(phantom_limit(tp, OLD_FLOOR)) if positive else "",
        "phantom_limit_065": str(phantom_limit(tp, NEW_FLOOR)) if positive else "",
        "legacy_gate_070": _status(legacy_070),
        "legacy_gate_065": _status(
            gate_from_confusion(tp, phantom, missed, NEW_FLOOR, zero_aware=False)
        ),
        "zero_aware_gate_070": _status(
            gate_from_confusion(tp, phantom, missed, OLD_FLOOR, zero_aware=True)
        ),
        "zero_aware_gate_065": _status(
            gate_from_confusion(tp, phantom, missed, NEW_FLOOR, zero_aware=True)
        ),
    }


def _gate_pair(precision: float, recall: float, floor: float) -> bool:
    return precision >= floor and recall >= floor


def _mean(rows: Iterable[Mapping[str, str]], field: str) -> float:
    values = [float(row[field]) for row in rows]
    if not values:
        raise RuntimeError(f"no values for macro field {field}")
    return sum(values) / len(values)


def _sum(rows: Iterable[Mapping[str, str]], field: str) -> int:
    return sum(int(row[field]) for row in rows)


def _candidate_output(
    cases: Sequence[Mapping[str, str]], source: Mapping[str, str]
) -> dict[str, str]:
    positive = [row for row in cases if row["oracle_positive"] == "true"]
    zero = [row for row in cases if row["oracle_positive"] == "false"]
    pooled_tp = _sum(cases, "true_positive_count")
    pooled_phantom = _sum(cases, "phantom_contact_count")
    pooled_missed = _sum(cases, "missed_contact_count")
    pooled = metrics_from_confusion(pooled_tp, pooled_phantom, pooled_missed)
    pooled_precision = float(pooled["precision"])
    pooled_recall = float(pooled["recall"])
    macro_all_precision = _mean(cases, "precision")
    macro_all_recall = _mean(cases, "recall")
    macro_pos_precision = _mean(positive, "precision")
    macro_pos_recall = _mean(positive, "recall")
    numeric_regressions = (
        ("pooled_precision", pooled_precision),
        ("pooled_recall", pooled_recall),
        ("macro_precision", macro_all_precision),
        ("macro_recall", macro_all_recall),
    )
    for field, actual in numeric_regressions:
        if abs(float(source[field]) - actual) > 1e-9:
            raise RuntimeError(f"source candidate regression: {field}")

    statuses: dict[str, bool] = {}
    for suffix, floor in (("070", OLD_FLOOR), ("065", NEW_FLOOR)):
        statuses[f"pooled_{suffix}"] = _gate_pair(
            pooled_precision, pooled_recall, floor
        )
        statuses[f"macro_legacy_{suffix}"] = _gate_pair(
            macro_all_precision, macro_all_recall, floor
        )
        statuses[f"macro_positive_{suffix}"] = _gate_pair(
            macro_pos_precision, macro_pos_recall, floor
        )
        statuses[f"legacy_all_{suffix}"] = (
            statuses[f"pooled_{suffix}"]
            and statuses[f"macro_legacy_{suffix}"]
            and all(row[f"legacy_gate_{suffix}"] == "PASS" for row in cases)
        )
        statuses[f"zero_aware_all_{suffix}"] = (
            statuses[f"pooled_{suffix}"]
            and statuses[f"macro_positive_{suffix}"]
            and all(row[f"zero_aware_gate_{suffix}"] == "PASS" for row in cases)
        )
    if _status(statuses["pooled_070"]) != source["pooled_gate_status"]:
        raise RuntimeError("source pooled gate regression")
    if _status(statuses["macro_legacy_070"]) != source["macro_gate_status"]:
        raise RuntimeError("source macro gate regression")
    if _status(statuses["legacy_all_070"]) != source["all_case_coverage_status"]:
        raise RuntimeError("source all-case gate regression")

    common_fields = (
        "candidate_key",
        "source",
        "object_key",
        "candidate_id",
        "threshold_m",
        "max_hulls",
        "actual_hulls",
        "max_vertices",
        "case_count",
    )
    output = {field: source[field] for field in common_fields}
    output.update(
        {
            "positive_case_count": str(len(positive)),
            "zero_oracle_case_count": str(len(zero)),
            "pooled_tp": str(pooled_tp),
            "pooled_phantom": str(pooled_phantom),
            "pooled_missed": str(pooled_missed),
            "pooled_precision": _float(pooled_precision),
            "pooled_recall": _float(pooled_recall),
            "macro_all_precision": _float(macro_all_precision),
            "macro_all_recall": _float(macro_all_recall),
            "macro_positive_precision": _float(macro_pos_precision),
            "macro_positive_recall": _float(macro_pos_recall),
        }
    )
    for suffix in ("070", "065"):
        output.update(
            {
                f"legacy_case_pass_count_{suffix}": str(
                    sum(row[f"legacy_gate_{suffix}"] == "PASS" for row in cases)
                ),
                f"zero_aware_case_pass_count_{suffix}": str(
                    sum(row[f"zero_aware_gate_{suffix}"] == "PASS" for row in cases)
                ),
                f"pooled_gate_{suffix}": _status(statuses[f"pooled_{suffix}"]),
                f"macro_legacy_gate_{suffix}": _status(
                    statuses[f"macro_legacy_{suffix}"]
                ),
                f"macro_positive_gate_{suffix}": _status(
                    statuses[f"macro_positive_{suffix}"]
                ),
                f"legacy_all_case_gate_{suffix}": _status(
                    statuses[f"legacy_all_{suffix}"]
                ),
                f"zero_aware_all_case_gate_{suffix}": _status(
                    statuses[f"zero_aware_all_{suffix}"]
                ),
            }
        )
    output["threshold_migration"] = (
        "PROMOTED"
        if not statuses["zero_aware_all_070"] and statuses["zero_aware_all_065"]
        else "STABLE_PASS"
        if statuses["zero_aware_all_070"]
        else "STABLE_FAIL"
    )
    output["zero_semantics_effect_070"] = (
        "PROMOTED"
        if not statuses["legacy_all_070"] and statuses["zero_aware_all_070"]
        else "UNCHANGED"
    )
    output["zero_semantics_effect_065"] = (
        "PROMOTED"
        if not statuses["legacy_all_065"] and statuses["zero_aware_all_065"]
        else "UNCHANGED"
    )
    return output


def _count(rows: Iterable[Mapping[str, str]], field: str) -> int:
    return sum(row[field] == "PASS" for row in rows)


def _aggregate_group(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    return {
        "candidate_count": len(rows),
        "pooled_pass_070": _count(rows, "pooled_gate_070"),
        "pooled_pass_065": _count(rows, "pooled_gate_065"),
        "macro_legacy_pass_070": _count(rows, "macro_legacy_gate_070"),
        "macro_legacy_pass_065": _count(rows, "macro_legacy_gate_065"),
        "macro_positive_pass_070": _count(rows, "macro_positive_gate_070"),
        "macro_positive_pass_065": _count(rows, "macro_positive_gate_065"),
        "legacy_all_case_pass_070": _count(rows, "legacy_all_case_gate_070"),
        "legacy_all_case_pass_065": _count(rows, "legacy_all_case_gate_065"),
        "zero_aware_all_case_pass_070": _count(rows, "zero_aware_all_case_gate_070"),
        "zero_aware_all_case_pass_065": _count(rows, "zero_aware_all_case_gate_065"),
        "threshold_promoted": sum(
            row["threshold_migration"] == "PROMOTED" for row in rows
        ),
    }


def _rank_key(row: Mapping[str, str]) -> tuple[Any, ...]:
    return (
        row["zero_aware_all_case_gate_065"] != "PASS",
        -int(row["zero_aware_case_pass_count_065"]),
        -min(
            float(row["macro_positive_precision"]),
            float(row["macro_positive_recall"]),
        ),
        -min(float(row["pooled_precision"]), float(row["pooled_recall"])),
        int(row["actual_hulls"]),
        row["candidate_key"],
    )


def aggregate(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    """Recompute all requested gates from E183 confusion counts and write tables."""
    protocol = load_protocol(output_root)
    source_cases, source_summaries = load_sources()
    case_rows = [_case_output(row) for row in source_cases]
    by_candidate: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in case_rows:
        by_candidate[row["candidate_key"]].append(row)
    summary_lookup = {row["candidate_key"]: row for row in source_summaries}
    summary_rows = [
        _candidate_output(by_candidate[key], summary_lookup[key])
        for key in sorted(by_candidate)
    ]
    unique_case_pose = {row["case_id"]: int(row["pose_count"]) for row in source_cases}
    unique_case_oracle = {
        row["case_id"]: int(row["oracle_contact_count"]) for row in case_rows
    }
    if sum(unique_case_pose.values()) != EXPECTED_POSES:
        raise RuntimeError("E183 unique pose closure changed")
    if sum(unique_case_oracle.values()) != EXPECTED_ORACLE_CONTACTS:
        raise RuntimeError("E183 unique oracle-contact closure changed")
    atomic_tsv(output_root / "case_candidate_metrics_gate065.tsv", case_rows)
    atomic_tsv(output_root / "candidate_summary_gate065.tsv", summary_rows)

    overall = _aggregate_group(summary_rows)
    if (
        overall["pooled_pass_070"],
        overall["macro_legacy_pass_070"],
        overall["legacy_all_case_pass_070"],
    ) != (23, 19, 3):
        raise RuntimeError("E183 aggregate regression failed")
    by_object = {}
    best_by_object = {}
    for object_key in OBJECT_KEYS:
        group = [row for row in summary_rows if row["object_key"] == object_key]
        by_object[object_key] = _aggregate_group(group)
        best_by_object[object_key] = sorted(group, key=_rank_key)[0]["candidate_key"]
    v9 = [row for row in summary_rows if row["source"] == "E182_V9"]
    promoted = [
        row["candidate_key"]
        for row in summary_rows
        if row["threshold_migration"] == "PROMOTED"
    ]
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "stage": "static_P_gate065_reaggregation",
        "status": "COMPLETE",
        "selection_contract": "SENSITIVITY_ONLY_NO_CANDIDATE_RETUNING",
        "case_count": EXPECTED_CASES,
        "candidate_count": len(summary_rows),
        "case_candidate_row_count": len(case_rows),
        "pose_count": sum(unique_case_pose.values()),
        "oracle_contact_count": sum(unique_case_oracle.values()),
        "zero_oracle_case_ids": sorted(
            case_id for case_id, count in unique_case_oracle.items() if count == 0
        ),
        "thresholds": {"baseline": OLD_FLOOR, "sensitivity": NEW_FLOOR},
        "overall": overall,
        "by_object": by_object,
        "best_by_object_gate065": best_by_object,
        "newly_promoted_zero_aware_all_case_candidates": promoted,
        "e182_v9": {
            **_aggregate_group(v9),
            "candidate_keys": [row["candidate_key"] for row in v9],
        },
        "isolation": protocol["isolation"],
        "protocol": _artifact(output_root / PROTOCOL_PATH_NAME),
        "source_case_table": _artifact(SOURCE_CASE_PATH),
        "source_candidate_summary": _artifact(SOURCE_SUMMARY_PATH),
        "case_table": _artifact(output_root / "case_candidate_metrics_gate065.tsv"),
        "candidate_table": _artifact(output_root / "candidate_summary_gate065.tsv"),
    }
    atomic_json(output_root / "aggregate.json", payload)
    return payload


def render_comparison(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    """Render 0.70→0.65 pass-count and best case-coverage comparisons."""
    load_protocol(output_root)
    aggregate_payload = json.loads((output_root / "aggregate.json").read_text())
    summaries = _read_tsv(output_root / "candidate_summary_gate065.tsv")
    categories = ("Pooled", "Macro+", "All-case\nzero-aware")
    old_fields = (
        "pooled_pass_070",
        "macro_positive_pass_070",
        "zero_aware_all_case_pass_070",
    )
    new_fields = (
        "pooled_pass_065",
        "macro_positive_pass_065",
        "zero_aware_all_case_pass_065",
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), dpi=160)
    for axis, object_key in zip(axes.flat[:3], OBJECT_KEYS, strict=True):
        group = aggregate_payload["by_object"][object_key]
        x = list(range(len(categories)))
        old = [group[field] for field in old_fields]
        new = [group[field] for field in new_fields]
        axis.bar(
            [value - 0.19 for value in x], old, 0.38, label="0.70", color="#6c8ebf"
        )
        axis.bar(
            [value + 0.19 for value in x], new, 0.38, label="0.65", color="#ed7d31"
        )
        for index, value in enumerate(old):
            axis.text(index - 0.19, value + 0.2, str(value), ha="center", fontsize=9)
        for index, value in enumerate(new):
            axis.text(index + 0.19, value + 0.2, str(value), ha="center", fontsize=9)
        axis.set_title(f"{object_key} ({group['candidate_count']} candidates)")
        axis.set_xticks(x, categories)
        axis.set_ylim(0, group["candidate_count"] + 2)
        axis.set_ylabel("candidate PASS count")
        axis.grid(axis="y", alpha=0.22)
        axis.legend(loc="upper left")

    coverage_axis = axes.flat[3]
    x = list(range(len(OBJECT_KEYS)))
    old_coverage = []
    new_coverage = []
    case_counts = []
    for object_key in OBJECT_KEYS:
        group = [row for row in summaries if row["object_key"] == object_key]
        old_coverage.append(
            max(int(row["zero_aware_case_pass_count_070"]) for row in group)
        )
        new_coverage.append(
            max(int(row["zero_aware_case_pass_count_065"]) for row in group)
        )
        case_counts.append(int(group[0]["case_count"]))
    coverage_axis.bar(
        [value - 0.19 for value in x], old_coverage, 0.38, label="0.70", color="#6c8ebf"
    )
    coverage_axis.bar(
        [value + 0.19 for value in x], new_coverage, 0.38, label="0.65", color="#ed7d31"
    )
    for index, (old, new, total) in enumerate(
        zip(old_coverage, new_coverage, case_counts, strict=True)
    ):
        coverage_axis.text(
            index - 0.19, old + 0.15, f"{old}/{total}", ha="center", fontsize=9
        )
        coverage_axis.text(
            index + 0.19, new + 0.15, f"{new}/{total}", ha="center", fontsize=9
        )
    coverage_axis.set_title("Best per-candidate case coverage")
    coverage_axis.set_xticks(x, OBJECT_KEYS)
    coverage_axis.set_ylim(0, max(case_counts) + 2)
    coverage_axis.set_ylabel("case PASS count")
    coverage_axis.grid(axis="y", alpha=0.22)
    coverage_axis.legend(loc="upper left")
    fig.suptitle(
        "E184 static-P threshold sensitivity (positive-only macro; zero-aware cases)"
    )
    fig.tight_layout()
    path = output_root / "threshold_comparison.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "stage": "threshold_comparison_visual",
        "status": "COMPLETE",
        "artifact": _artifact(path),
        "old_best_case_coverage": dict(zip(OBJECT_KEYS, old_coverage, strict=True)),
        "new_best_case_coverage": dict(zip(OBJECT_KEYS, new_coverage, strict=True)),
    }
    atomic_json(output_root / "visual_manifest.json", payload)
    return payload


def validate(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    """Validate source immutability, row closure, regressions, outputs, and isolation."""
    protocol = load_protocol(output_root)
    aggregate_payload = json.loads((output_root / "aggregate.json").read_text())
    case_rows = _read_tsv(output_root / "case_candidate_metrics_gate065.tsv")
    summary_rows = _read_tsv(output_root / "candidate_summary_gate065.tsv")
    visual = json.loads((output_root / "visual_manifest.json").read_text())
    checks = {
        "source_case_sha": sha256_file(SOURCE_CASE_PATH) == EXPECTED_CASE_SHA256,
        "source_summary_sha": sha256_file(SOURCE_SUMMARY_PATH)
        == EXPECTED_SUMMARY_SHA256,
        "case_rows_540": len(case_rows) == EXPECTED_CASE_ROWS,
        "candidate_rows_60": len(summary_rows) == EXPECTED_CANDIDATES,
        "unique_candidate_case_rows": len(
            {(row["candidate_key"], row["case_id"]) for row in case_rows}
        )
        == EXPECTED_CASE_ROWS,
        "legacy_070_exact": (
            aggregate_payload["overall"]["pooled_pass_070"],
            aggregate_payload["overall"]["macro_legacy_pass_070"],
            aggregate_payload["overall"]["legacy_all_case_pass_070"],
        )
        == (23, 19, 3),
        "threshold_monotone": all(
            int(row["zero_aware_case_pass_count_065"])
            >= int(row["zero_aware_case_pass_count_070"])
            for row in summary_rows
        ),
        "zero_oracle_contract": all(
            row["zero_aware_gate_065"]
            == _status(int(row["phantom_contact_count"]) == 0)
            for row in case_rows
            if row["oracle_positive"] == "false"
        ),
        "visual_exists": (output_root / "threshold_comparison.png").is_file()
        and visual["status"] == "COMPLETE",
        "gpu_sdf_coacd_full_zero": all(
            value == 0 for value in protocol["isolation"].values()
        ),
    }
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "stage": "static_P_gate065_validation",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "artifacts": {
            "protocol": _artifact(output_root / PROTOCOL_PATH_NAME),
            "case_table": _artifact(output_root / "case_candidate_metrics_gate065.tsv"),
            "candidate_table": _artifact(output_root / "candidate_summary_gate065.tsv"),
            "aggregate": _artifact(output_root / "aggregate.json"),
            "comparison": _artifact(output_root / "threshold_comparison.png"),
        },
    }
    atomic_json(output_root / "validation.json", payload)
    if payload["status"] != "PASS":
        raise RuntimeError(f"E184 validation failed: {checks}")
    return payload


def parse_args() -> argparse.Namespace:
    """Parse one explicit resume-safe offline E184 stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("protocol", "aggregate", "visual", "validate", "all")
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def main() -> int:
    """Run the requested E184 stage and print a compact status."""
    args = parse_args()
    if args.stage in {"protocol", "all"}:
        payload = freeze_protocol(args.output_root)
        print(f"E184_PROTOCOL={payload['status']}")
    if args.stage in {"aggregate", "all"}:
        payload = aggregate(args.output_root)
        print(
            "E184_AGGREGATE=COMPLETE "
            f"zeroaware_allcase_065={payload['overall']['zero_aware_all_case_pass_065']}"
        )
    if args.stage in {"visual", "all"}:
        payload = render_comparison(args.output_root)
        print(f"E184_VISUAL={payload['status']}")
    if args.stage in {"validate", "all"}:
        payload = validate(args.output_root)
        print(f"E184_VALIDATE={payload['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
