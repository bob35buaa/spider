#!/usr/bin/env python3
"""Validate E176 canary planning-time throughput before Full CEM."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[5]
DEFAULT_MANIFEST = (
    REPO
    / "workspace/core4d/results/E176/s6_downstream/manifests/"
    "lowgeom_canary_manifest.tsv"
)
DEFAULT_OUTPUT = (
    REPO
    / "workspace/core4d/results/E176/s6_downstream/cem/canary/"
    "canary_throughput_gate.json"
)
PLAN_TIME_RE = re.compile(
    r"plan time:\s*([0-9]+(?:\.[0-9]+)?)s,.*?opt_steps:\s*(\d+)"
)
MAX_MEDIAN_PLAN_TIME_S = 3.0


def repo_path(raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPO / path


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def plan_times(path: Path) -> list[float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [
        float(match.group(1))
        for match in PLAN_TIME_RE.finditer(text)
        if int(match.group(2)) > 0
    ]


def run(
    manifest: Path,
    output: Path,
    *,
    threshold_s: float,
    expected_rows: int = 6,
    experiment_id: str = "E176",
) -> dict[str, Any]:
    rows = read_tsv(manifest)
    errors: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    if (
        len(rows) != expected_rows
        or len({row["case_id"] for row in rows}) != expected_rows
    ):
        errors.append(
            {
                "scope": "manifest",
                "failure": (
                    "canary_manifest_not_expected_unique_rows:"
                    f"{expected_rows}"
                ),
            }
        )
    for row in rows:
        case_id = row["case_id"]
        log_path = repo_path(row["log"])
        failures: list[str] = []
        values: list[float] = []
        if not log_path.is_file():
            failures.append("missing_case_log")
        else:
            values = plan_times(log_path)
            if not values:
                failures.append("no_optimized_plan_time_records")
        median_s = statistics.median(values) if values else float("nan")
        if values and median_s > threshold_s:
            failures.append(
                f"median_plan_time_s:{median_s:.6f}>{threshold_s:.6f}"
            )
        if int(row["object_geom_count"]) > 9:
            failures.append("object_geom_count_gt_9")
        if int(row["compiled_robot_object_pair_count"]) > 162:
            failures.append("compiled_pair_count_gt_162")
        metric = {
            "case_id": case_id,
            "object_key": row["object_key"],
            "object_geom_count": int(row["object_geom_count"]),
            "compiled_robot_object_pair_count": int(
                row["compiled_robot_object_pair_count"]
            ),
            "plan_time_records": len(values),
            "median_plan_time_s": median_s,
            "p90_plan_time_s": (
                sorted(values)[min(len(values) - 1, int(0.9 * len(values)))]
                if values
                else float("nan")
            ),
            "max_plan_time_s": max(values) if values else float("nan"),
            "threshold_s": threshold_s,
            "status": "pass" if not failures else "fail",
            "failures": failures,
            "log": str(log_path.relative_to(REPO))
            if log_path.is_relative_to(REPO)
            else str(log_path),
        }
        metrics.append(metric)
        if failures:
            errors.append(
                {
                    "scope": "case",
                    "case_id": case_id,
                    "failures": failures,
                }
            )
    passed_rows = sum(row["status"] == "pass" for row in metrics)
    status = (
        "pass"
        if not errors and passed_rows == expected_rows
        else "fail"
    )
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "experiment_id": experiment_id,
        "status": status,
        "metric": "median optimized-record plan time",
        "threshold_s": threshold_s,
        "expected_rows": expected_rows,
        "passed_rows": passed_rows,
        "failed_rows": expected_rows - passed_rows,
        "manifest": str(manifest.relative_to(REPO))
        if manifest.is_relative_to(REPO)
        else str(manifest),
        "rows": metrics,
        "errors": errors,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": status,
                "passed_rows": passed_rows,
                "expected_rows": expected_rows,
                "threshold_s": threshold_s,
                "output": str(output.relative_to(REPO))
                if output.is_relative_to(REPO)
                else str(output),
            },
            sort_keys=True,
        )
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--threshold-s",
        type=float,
        default=MAX_MEDIAN_PLAN_TIME_S,
    )
    parser.add_argument("--expected-rows", type=int, default=6)
    parser.add_argument("--experiment-id", default="E176")
    args = parser.parse_args()
    payload = run(
        repo_path(args.manifest),
        repo_path(args.output),
        threshold_s=args.threshold_s,
        expected_rows=args.expected_rows,
        experiment_id=args.experiment_id,
    )
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
