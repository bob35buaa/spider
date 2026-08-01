#!/usr/bin/env python3
"""Re-aggregate frozen E183 static-P confusions at 0.70, 0.65, and 0.60."""

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
EXPERIMENT_ID = "E185"
E183_ROOT = REPO_ROOT / "workspace/core4d/results/E183/full27_static_p"
E184_ROOT = REPO_ROOT / "workspace/core4d/results/E184/static_p_gate065"
SOURCE_CASE_PATH = E183_ROOT / "case_candidate_metrics.tsv"
SOURCE_SUMMARY_PATH = E183_ROOT / "candidate_summary.tsv"
E184_AGGREGATE_PATH = E184_ROOT / "aggregate.json"
EXPECTED_CASE_SHA256 = (
    "28a3c8118fe10767856b5d6bcdb93430dc9ba629d7f4bb58eead9924168a949c"
)
EXPECTED_SUMMARY_SHA256 = (
    "08b8927ccc0513b8ff6dc20bf36f83526c9cf265472fd6bf441107aa21766093"
)
EXPECTED_E184_AGGREGATE_SHA256 = (
    "e448c960ac723c0bab0de8b6582ddf9d0f8dfda285f70011b939de1dd47f91d1"
)
OUTPUT_ROOT = REPO_ROOT / "workspace/core4d/results/E185/static_p_gate060"
PROTOCOL_NAME = "protocol_manifest.json"
OBJECT_KEYS = ("bucket003", "bucket004", "bucket007")
FLOORS = {"070": 0.70, "065": 0.65, "060": 0.60}
EXPECTED_CASE_ROWS = 540
EXPECTED_CANDIDATES = 60
EXPECTED_CASES = 27
EXPECTED_POSES = 14_542
EXPECTED_ORACLE_CONTACTS = 2_198
EXPECTED_E184_OVERALL = {
    "pooled_pass_070": 23,
    "pooled_pass_065": 27,
    "macro_positive_pass_070": 26,
    "macro_positive_pass_065": 30,
    "zero_aware_all_case_pass_070": 3,
    "zero_aware_all_case_pass_065": 4,
}


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
    """Write stable JSON atomically."""
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def atomic_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a non-empty sequence of mappings as TSV atomically."""
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


def _display_path(path: Path) -> str:
    absolute = path.absolute()
    try:
        return str(absolute.relative_to(REPO_ROOT))
    except ValueError:
        return str(absolute)


def _artifact(path: Path) -> dict[str, str]:
    return {"path": _display_path(path), "sha256": sha256_file(path)}


def _status(value: bool) -> str:
    return "PASS" if value else "FAIL"


def _float(value: float) -> str:
    return f"{value:.9f}"


def phantom_limit(true_positive: int, floor: float) -> int:
    """Return the largest phantom count satisfying a precision floor."""
    if true_positive < 0 or not 0.0 < floor <= 1.0:
        raise ValueError("invalid TP or floor")
    return math.floor(true_positive * (1.0 - floor) / floor + 1e-12)


def confusion_metrics(tp: int, phantom: int, missed: int) -> dict[str, Any]:
    """Compute P/R and the oracle-positive flag from immutable counts."""
    if min(tp, phantom, missed) < 0:
        raise ValueError("confusion counts cannot be negative")
    predicted = tp + phantom
    oracle = tp + missed
    return {
        "oracle_contact_count": oracle,
        "oracle_positive": oracle > 0,
        "precision": tp / predicted if predicted else 0.0,
        "recall": tp / oracle if oracle else 0.0,
    }


def gate_from_confusion(
    tp: int, phantom: int, missed: int, floor: float, *, zero_aware: bool
) -> bool:
    """Apply P/R floor or the explicit no-contact contract."""
    metrics = confusion_metrics(tp, phantom, missed)
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
    """Load exact E183 authority after digest and row-closure checks."""
    if sha256_file(case_path) != expected_case_sha:
        raise RuntimeError("E183 case table SHA changed")
    if sha256_file(summary_path) != expected_summary_sha:
        raise RuntimeError("E183 candidate summary SHA changed")
    cases = _read_tsv(case_path)
    summaries = _read_tsv(summary_path)
    pairs = {(row["candidate_key"], row["case_id"]) for row in cases}
    candidate_keys = {row["candidate_key"] for row in cases}
    if len(cases) != EXPECTED_CASE_ROWS or len(pairs) != EXPECTED_CASE_ROWS:
        raise RuntimeError("E183 candidate-case closure changed")
    if (
        len(summaries) != EXPECTED_CANDIDATES
        or len(candidate_keys) != EXPECTED_CANDIDATES
    ):
        raise RuntimeError("E183 candidate closure changed")
    if len({row["case_id"] for row in cases}) != EXPECTED_CASES:
        raise RuntimeError("E183 case authority changed")
    if {row["candidate_key"] for row in summaries} != candidate_keys:
        raise RuntimeError("E183 case/summary candidate sets differ")
    return cases, summaries


def protocol_payload() -> dict[str, Any]:
    """Build the result-independent E185 protocol payload."""
    runner = Path(__file__).resolve()
    wrapper = REPO_ROOT / "workspace/core4d/scripts/eval/wrappers/eval_E185_gate060.sh"
    return {
        "experiment_id": EXPERIMENT_ID,
        "status": "FROZEN",
        "scope": "OFFLINE_REAGGREGATION_ONLY",
        "thresholds": FLOORS,
        "positive_case_gate": "precision>=floor AND recall>=floor",
        "phantom_limit_060": "floor(TP*0.40/0.60)=floor(2TP/3)",
        "zero_oracle_gate": "PASS iff phantom==0",
        "macro_contract": "positive-oracle cases only",
        "all_case_contract": "pooled AND positive-only macro AND every case",
        "source_case_table": {
            "path": _display_path(SOURCE_CASE_PATH),
            "sha256": EXPECTED_CASE_SHA256,
            "rows": EXPECTED_CASE_ROWS,
        },
        "source_candidate_summary": {
            "path": _display_path(SOURCE_SUMMARY_PATH),
            "sha256": EXPECTED_SUMMARY_SHA256,
            "rows": EXPECTED_CANDIDATES,
        },
        "e184_comparison_authority": {
            "path": _display_path(E184_AGGREGATE_PATH),
            "sha256": EXPECTED_E184_AGGREGATE_SHA256,
        },
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
    """Freeze the protocol before any E185 aggregate result exists."""
    load_sources()
    if sha256_file(E184_AGGREGATE_PATH) != EXPECTED_E184_AGGREGATE_SHA256:
        raise RuntimeError("E184 comparison authority SHA changed")
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / PROTOCOL_NAME
    payload = protocol_payload()
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise RuntimeError("frozen E185 protocol differs from current contract")
        return payload
    result_names = (
        "case_candidate_metrics_gate060.tsv",
        "candidate_summary_gate060.tsv",
        "aggregate.json",
        "threshold_comparison_070_065_060.png",
        "visual_manifest.json",
        "validation.json",
    )
    if any((output_root / name).exists() for name in result_names):
        raise RuntimeError("E185 result exists before protocol freeze")
    atomic_json(path, payload)
    return payload


def load_protocol(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    """Require frozen sources, runner, and wrapper to remain exact."""
    path = output_root / PROTOCOL_NAME
    if not path.is_file():
        raise RuntimeError("E185 protocol is not frozen")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload != protocol_payload():
        raise RuntimeError("E185 protocol or implementation changed after freeze")
    load_sources()
    if sha256_file(E184_AGGREGATE_PATH) != EXPECTED_E184_AGGREGATE_SHA256:
        raise RuntimeError("E184 comparison authority SHA changed")
    return payload


def _case_output(row: Mapping[str, str]) -> dict[str, str]:
    tp = int(row["true_positive_count"])
    phantom = int(row["phantom_contact_count"])
    missed = int(row["missed_contact_count"])
    metrics = confusion_metrics(tp, phantom, missed)
    if abs(float(row["precision"]) - metrics["precision"]) > 1e-9:
        raise RuntimeError("source precision regression")
    if abs(float(row["recall"]) - metrics["recall"]) > 1e-9:
        raise RuntimeError("source recall regression")
    result = {
        **row,
        "oracle_contact_count": str(metrics["oracle_contact_count"]),
        "oracle_positive": str(metrics["oracle_positive"]).lower(),
    }
    for suffix, floor in FLOORS.items():
        result[f"phantom_limit_{suffix}"] = (
            str(phantom_limit(tp, floor)) if metrics["oracle_positive"] else ""
        )
        result[f"zero_aware_gate_{suffix}"] = _status(
            gate_from_confusion(tp, phantom, missed, floor, zero_aware=True)
        )
    if (
        result["zero_aware_gate_070"] != row["gate_status"]
        and metrics["oracle_positive"]
    ):
        raise RuntimeError("source positive-case 0.70 gate regression")
    return result


def _mean(rows: Iterable[Mapping[str, str]], field: str) -> float:
    values = [float(row[field]) for row in rows]
    if not values:
        raise RuntimeError(f"no macro values for {field}")
    return sum(values) / len(values)


def _sum(rows: Iterable[Mapping[str, str]], field: str) -> int:
    return sum(int(row[field]) for row in rows)


def _pair_pass(precision: float, recall: float, floor: float) -> bool:
    return precision >= floor and recall >= floor


def _candidate_output(
    cases: Sequence[Mapping[str, str]], source: Mapping[str, str]
) -> dict[str, str]:
    positive = [row for row in cases if row["oracle_positive"] == "true"]
    zero = [row for row in cases if row["oracle_positive"] == "false"]
    tp = _sum(cases, "true_positive_count")
    phantom = _sum(cases, "phantom_contact_count")
    missed = _sum(cases, "missed_contact_count")
    pooled = confusion_metrics(tp, phantom, missed)
    macro_precision = _mean(positive, "precision")
    macro_recall = _mean(positive, "recall")
    if abs(float(source["pooled_precision"]) - pooled["precision"]) > 1e-9:
        raise RuntimeError("source pooled precision regression")
    if abs(float(source["pooled_recall"]) - pooled["recall"]) > 1e-9:
        raise RuntimeError("source pooled recall regression")
    common = (
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
    result = {field: source[field] for field in common}
    result.update(
        {
            "positive_case_count": str(len(positive)),
            "zero_oracle_case_count": str(len(zero)),
            "pooled_tp": str(tp),
            "pooled_phantom": str(phantom),
            "pooled_missed": str(missed),
            "pooled_precision": _float(pooled["precision"]),
            "pooled_recall": _float(pooled["recall"]),
            "macro_positive_precision": _float(macro_precision),
            "macro_positive_recall": _float(macro_recall),
        }
    )
    for suffix, floor in FLOORS.items():
        pooled_pass = _pair_pass(pooled["precision"], pooled["recall"], floor)
        macro_pass = _pair_pass(macro_precision, macro_recall, floor)
        case_pass = sum(row[f"zero_aware_gate_{suffix}"] == "PASS" for row in cases)
        all_pass = pooled_pass and macro_pass and case_pass == len(cases)
        result.update(
            {
                f"case_pass_count_{suffix}": str(case_pass),
                f"pooled_gate_{suffix}": _status(pooled_pass),
                f"macro_positive_gate_{suffix}": _status(macro_pass),
                f"all_case_gate_{suffix}": _status(all_pass),
            }
        )
    result["migration_065_to_060"] = (
        "PROMOTED"
        if result["all_case_gate_065"] == "FAIL"
        and result["all_case_gate_060"] == "PASS"
        else "STABLE_PASS"
        if result["all_case_gate_065"] == "PASS"
        else "STABLE_FAIL"
    )
    return result


def _count(rows: Iterable[Mapping[str, str]], field: str) -> int:
    return sum(row[field] == "PASS" for row in rows)


def _group_metrics(rows: Sequence[Mapping[str, str]]) -> dict[str, int]:
    payload: dict[str, int] = {"candidate_count": len(rows)}
    for suffix in FLOORS:
        payload[f"pooled_pass_{suffix}"] = _count(rows, f"pooled_gate_{suffix}")
        payload[f"macro_positive_pass_{suffix}"] = _count(
            rows, f"macro_positive_gate_{suffix}"
        )
        payload[f"zero_aware_all_case_pass_{suffix}"] = _count(
            rows, f"all_case_gate_{suffix}"
        )
        payload[f"best_case_coverage_{suffix}"] = max(
            int(row[f"case_pass_count_{suffix}"]) for row in rows
        )
    payload["promoted_065_to_060"] = sum(
        row["migration_065_to_060"] == "PROMOTED" for row in rows
    )
    return payload


def _rank_key(row: Mapping[str, str]) -> tuple[Any, ...]:
    return (
        row["all_case_gate_060"] != "PASS",
        -int(row["case_pass_count_060"]),
        -min(
            float(row["macro_positive_precision"]),
            float(row["macro_positive_recall"]),
        ),
        -min(float(row["pooled_precision"]), float(row["pooled_recall"])),
        int(row["actual_hulls"]),
        row["candidate_key"],
    )


def aggregate(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    """Generate three-threshold case/candidate tables and aggregate JSON."""
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
    unique_pose = {row["case_id"]: int(row["pose_count"]) for row in case_rows}
    unique_oracle = {
        row["case_id"]: int(row["oracle_contact_count"]) for row in case_rows
    }
    if sum(unique_pose.values()) != EXPECTED_POSES:
        raise RuntimeError("unique pose closure changed")
    if sum(unique_oracle.values()) != EXPECTED_ORACLE_CONTACTS:
        raise RuntimeError("oracle contact closure changed")
    atomic_tsv(output_root / "case_candidate_metrics_gate060.tsv", case_rows)
    atomic_tsv(output_root / "candidate_summary_gate060.tsv", summary_rows)
    overall = _group_metrics(summary_rows)
    for key, expected in EXPECTED_E184_OVERALL.items():
        if overall[key] != expected:
            raise RuntimeError(f"E184 0.70/0.65 regression failed: {key}")
    by_object = {}
    best_by_object = {}
    for object_key in OBJECT_KEYS:
        group = [row for row in summary_rows if row["object_key"] == object_key]
        by_object[object_key] = _group_metrics(group)
        best_by_object[object_key] = sorted(group, key=_rank_key)[0]["candidate_key"]
    v9 = [row for row in summary_rows if row["source"] == "E182_V9"]
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "stage": "static_P_gate060_reaggregation",
        "status": "COMPLETE",
        "selection_contract": "SENSITIVITY_ONLY_NO_CANDIDATE_RETUNING",
        "thresholds": FLOORS,
        "case_count": EXPECTED_CASES,
        "candidate_count": len(summary_rows),
        "case_candidate_row_count": len(case_rows),
        "pose_count": sum(unique_pose.values()),
        "oracle_contact_count": sum(unique_oracle.values()),
        "overall": overall,
        "by_object": by_object,
        "best_by_object_gate060": best_by_object,
        "newly_promoted_065_to_060": [
            row["candidate_key"]
            for row in summary_rows
            if row["migration_065_to_060"] == "PROMOTED"
        ],
        "e182_v9": {
            **_group_metrics(v9),
            "candidate_keys": [row["candidate_key"] for row in v9],
        },
        "zero_oracle_case_ids": sorted(
            case_id for case_id, count in unique_oracle.items() if count == 0
        ),
        "isolation": protocol["isolation"],
        "protocol": _artifact(output_root / PROTOCOL_NAME),
        "source_case_table": _artifact(SOURCE_CASE_PATH),
        "source_candidate_summary": _artifact(SOURCE_SUMMARY_PATH),
        "e184_comparison_authority": _artifact(E184_AGGREGATE_PATH),
        "case_table": _artifact(output_root / "case_candidate_metrics_gate060.tsv"),
        "candidate_table": _artifact(output_root / "candidate_summary_gate060.tsv"),
    }
    atomic_json(output_root / "aggregate.json", payload)
    return payload


def render_comparison(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    """Render 0.70/0.65/0.60 candidate and coverage comparisons."""
    load_protocol(output_root)
    aggregate_payload = json.loads((output_root / "aggregate.json").read_text())
    categories = ("Pooled", "Macro+", "All-case")
    suffixes = ("070", "065", "060")
    labels = ("0.70", "0.65", "0.60")
    colors = ("#6c8ebf", "#ed7d31", "#70ad47")
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), dpi=160)
    for axis, object_key in zip(axes.flat[:3], OBJECT_KEYS, strict=True):
        group = aggregate_payload["by_object"][object_key]
        x = list(range(len(categories)))
        for threshold_index, (suffix, label, color) in enumerate(
            zip(suffixes, labels, colors, strict=True)
        ):
            offset = (threshold_index - 1) * 0.25
            values = [
                group[f"pooled_pass_{suffix}"],
                group[f"macro_positive_pass_{suffix}"],
                group[f"zero_aware_all_case_pass_{suffix}"],
            ]
            axis.bar(
                [value + offset for value in x], values, 0.24, label=label, color=color
            )
            for index, value in enumerate(values):
                axis.text(
                    index + offset, value + 0.18, str(value), ha="center", fontsize=8
                )
        axis.set_title(f"{object_key} ({group['candidate_count']} candidates)")
        axis.set_xticks(x, categories)
        axis.set_ylim(0, group["candidate_count"] + 2)
        axis.set_ylabel("candidate PASS count")
        axis.grid(axis="y", alpha=0.22)
        axis.legend(loc="upper left")
    coverage_axis = axes.flat[3]
    x = list(range(len(OBJECT_KEYS)))
    coverage_payload: dict[str, dict[str, int]] = {}
    case_counts = {"bucket003": 9, "bucket004": 4, "bucket007": 14}
    for threshold_index, (suffix, label, color) in enumerate(
        zip(suffixes, labels, colors, strict=True)
    ):
        offset = (threshold_index - 1) * 0.25
        values = [
            aggregate_payload["by_object"][key][f"best_case_coverage_{suffix}"]
            for key in OBJECT_KEYS
        ]
        coverage_payload[label] = dict(zip(OBJECT_KEYS, values, strict=True))
        coverage_axis.bar(
            [value + offset for value in x], values, 0.24, label=label, color=color
        )
        for index, (value, object_key) in enumerate(
            zip(values, OBJECT_KEYS, strict=True)
        ):
            coverage_axis.text(
                index + offset,
                value + 0.12,
                f"{value}/{case_counts[object_key]}",
                ha="center",
                fontsize=8,
            )
    coverage_axis.set_title("Best per-candidate case coverage")
    coverage_axis.set_xticks(x, OBJECT_KEYS)
    coverage_axis.set_ylim(0, 16)
    coverage_axis.set_ylabel("case PASS count")
    coverage_axis.grid(axis="y", alpha=0.22)
    coverage_axis.legend(loc="upper left")
    fig.suptitle("E185 static-P sensitivity (positive-only macro; zero-aware cases)")
    fig.tight_layout()
    path = output_root / "threshold_comparison_070_065_060.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "stage": "threshold_comparison_visual",
        "status": "COMPLETE",
        "artifact": _artifact(path),
        "best_case_coverage": coverage_payload,
    }
    atomic_json(output_root / "visual_manifest.json", payload)
    return payload


def validate(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    """Validate sources, regressions, threshold monotonicity, visual, and isolation."""
    protocol = load_protocol(output_root)
    aggregate_payload = json.loads((output_root / "aggregate.json").read_text())
    case_rows = _read_tsv(output_root / "case_candidate_metrics_gate060.tsv")
    summary_rows = _read_tsv(output_root / "candidate_summary_gate060.tsv")
    visual = json.loads((output_root / "visual_manifest.json").read_text())
    checks = {
        "source_case_sha": sha256_file(SOURCE_CASE_PATH) == EXPECTED_CASE_SHA256,
        "source_summary_sha": sha256_file(SOURCE_SUMMARY_PATH)
        == EXPECTED_SUMMARY_SHA256,
        "e184_authority_sha": sha256_file(E184_AGGREGATE_PATH)
        == EXPECTED_E184_AGGREGATE_SHA256,
        "case_rows_540": len(case_rows) == EXPECTED_CASE_ROWS,
        "candidate_rows_60": len(summary_rows) == EXPECTED_CANDIDATES,
        "unique_candidate_case_rows": len(
            {(row["candidate_key"], row["case_id"]) for row in case_rows}
        )
        == EXPECTED_CASE_ROWS,
        "e184_070_065_exact": all(
            aggregate_payload["overall"][key] == value
            for key, value in EXPECTED_E184_OVERALL.items()
        ),
        "case_threshold_monotone": all(
            int(row["case_pass_count_070"])
            <= int(row["case_pass_count_065"])
            <= int(row["case_pass_count_060"])
            for row in summary_rows
        ),
        "zero_oracle_contract": all(
            row["zero_aware_gate_060"]
            == _status(int(row["phantom_contact_count"]) == 0)
            for row in case_rows
            if row["oracle_positive"] == "false"
        ),
        "visual_exists": (
            output_root / "threshold_comparison_070_065_060.png"
        ).is_file()
        and visual["status"] == "COMPLETE",
        "gpu_sdf_coacd_full_zero": all(
            value == 0 for value in protocol["isolation"].values()
        ),
    }
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "stage": "static_P_gate060_validation",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "artifacts": {
            "protocol": _artifact(output_root / PROTOCOL_NAME),
            "case_table": _artifact(output_root / "case_candidate_metrics_gate060.tsv"),
            "candidate_table": _artifact(output_root / "candidate_summary_gate060.tsv"),
            "aggregate": _artifact(output_root / "aggregate.json"),
            "comparison": _artifact(
                output_root / "threshold_comparison_070_065_060.png"
            ),
        },
    }
    atomic_json(output_root / "validation.json", payload)
    if payload["status"] != "PASS":
        raise RuntimeError(f"E185 validation failed: {checks}")
    return payload


def parse_args() -> argparse.Namespace:
    """Parse one explicit E185 offline stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("protocol", "aggregate", "visual", "validate", "all")
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def main() -> int:
    """Run the requested E185 stage and print a compact status."""
    args = parse_args()
    if args.stage in {"protocol", "all"}:
        payload = freeze_protocol(args.output_root)
        print(f"E185_PROTOCOL={payload['status']}")
    if args.stage in {"aggregate", "all"}:
        payload = aggregate(args.output_root)
        print(
            "E185_AGGREGATE=COMPLETE "
            f"allcase060={payload['overall']['zero_aware_all_case_pass_060']}"
        )
    if args.stage in {"visual", "all"}:
        payload = render_comparison(args.output_root)
        print(f"E185_VISUAL={payload['status']}")
    if args.stage in {"validate", "all"}:
        payload = validate(args.output_root)
        print(f"E185_VALIDATE={payload['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
