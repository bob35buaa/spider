#!/usr/bin/env python3
"""Build a race-safe local/A100 suffix allocation for the active E178 Full run."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


PLAN_TIME_RE = re.compile(
    r"plan time:\s*([0-9]+(?:\.[0-9]+)?)s,"
    r".*?sim_steps:\s*(\d+)/(\d+),\s*opt_steps:\s*(\d+)"
)


def read_tsv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        return list(reader), list(reader.fieldnames or [])


def write_tsv(
    path: Path,
    rows: list[dict[str, Any]],
    fields: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_plan_times(path: Path) -> tuple[list[float], dict[str, int] | None]:
    if not path.is_file():
        return [], None
    values: list[float] = []
    last: dict[str, int] | None = None
    for match in PLAN_TIME_RE.finditer(
        path.read_text(encoding="utf-8", errors="replace")
    ):
        steps = int(match.group(4))
        last = {
            "sim_steps": int(match.group(2)),
            "sim_total": int(match.group(3)),
            "opt_steps": steps,
        }
        if steps > 0:
            values.append(float(match.group(1)))
    return values, last


def object_rate(
    object_key: str,
    *,
    five_geom_rate: float,
    one_geom_rate: float,
) -> float:
    return one_geom_rate if object_key == "bucket004" else five_geom_rate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-manifest", type=Path, required=True)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--case-log-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--local-five-geom-seconds-per-record",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--local-safety-factor",
        type=float,
        default=1.2,
    )
    parser.add_argument(
        "--minimum-deadline-margin-s",
        type=float,
        default=300.0,
    )
    parser.add_argument(
        "--minimum-improvement-pct",
        type=float,
        default=5.0,
    )
    args = parser.parse_args()

    full_rows, full_fields = read_tsv(args.full_manifest)
    full_by_case = {row["case_id"]: row for row in full_rows}
    if len(full_rows) != 27 or len(full_by_case) != 27:
        raise ValueError("E178 Full authority must be 27 unique cases")

    frame_rows, _ = read_tsv(args.queue_dir / "queue_order.tsv")
    frame_by_gpu_case = {
        (row["gpu_id"], row["case_id"]): int(row["trajectory_frames"])
        for row in frame_rows
    }

    queues: dict[str, list[dict[str, Any]]] = {}
    running_medians: list[tuple[str, str, float]] = []
    for shard in sorted(args.queue_dir.glob("gpu*.tsv")):
        gpu = shard.stem.removeprefix("gpu")
        shard_rows, _ = read_tsv(shard)
        enriched: list[dict[str, Any]] = []
        for position, row in enumerate(shard_rows, start=1):
            case_id = row["case_id"]
            if case_id not in full_by_case:
                raise ValueError(f"shard case missing from Full authority: {case_id}")
            frames = frame_by_gpu_case[(gpu, case_id)]
            records_total = max(0, frames - 6)
            values, last = parse_plan_times(
                args.case_log_dir / Path(row["log"]).name
            )
            median = statistics.median(values) if values else None
            if row["status"] in {
                "running",
                "run_complete_pending_eval",
            } and median is not None:
                running_medians.append((gpu, row["object_key"], median))
            enriched.append(
                {
                    "gpu": gpu,
                    "position": position,
                    "row": row,
                    "frames": frames,
                    "records_total": records_total,
                    "records_observed": len(values),
                    "records_remaining": max(0, records_total - len(values)),
                    "median_plan_time_s": median,
                    "last": last,
                }
            )
        queues[gpu] = enriched

    five_values = [
        value
        for _, object_key, value in running_medians
        if object_key != "bucket004"
    ]
    one_values = [
        value
        for _, object_key, value in running_medians
        if object_key == "bucket004"
    ]
    if not five_values or not one_values:
        raise ValueError(
            "need live running medians for both five-geom and one-geom rows"
        )
    remote_five = statistics.median(five_values)
    remote_one = statistics.median(one_values)
    local_five = args.local_five_geom_seconds_per_record
    local_one = local_five * remote_one / remote_five

    def remote_rate(object_key: str) -> float:
        return object_rate(
            object_key,
            five_geom_rate=remote_five,
            one_geom_rate=remote_one,
        )

    def local_rate(object_key: str) -> float:
        return object_rate(
            object_key,
            five_geom_rate=local_five,
            one_geom_rate=local_one,
        )

    base_remote_seconds: dict[str, float] = {}
    pending: dict[str, list[dict[str, Any]]] = {}
    fixed_seconds: dict[str, float] = {}
    for gpu, rows in queues.items():
        fixed = 0.0
        not_run: list[dict[str, Any]] = []
        for item in rows:
            status = item["row"]["status"]
            object_key = item["row"]["object_key"]
            if status == "running":
                fixed += item["records_remaining"] * remote_rate(object_key)
            elif status in {"run_complete_pending_eval"}:
                continue
            elif status == "not_run":
                remote_seconds = item["records_total"] * remote_rate(object_key)
                local_seconds = item["records_total"] * local_rate(object_key)
                value = dict(item)
                value["remote_seconds"] = remote_seconds
                value["local_seconds"] = local_seconds
                not_run.append(value)
            else:
                raise ValueError(
                    f"unsupported live status {status} for {item['row']['case_id']}"
                )
        fixed_seconds[gpu] = fixed
        pending[gpu] = not_run
        base_remote_seconds[gpu] = fixed + sum(
            item["remote_seconds"] for item in not_run
        )

    gpu_order = sorted(queues, key=int)
    baseline_makespan = max(base_remote_seconds.values())
    candidates: list[dict[str, Any]] = []
    for tail_counts in itertools.product(
        *(range(len(pending[gpu]) + 1) for gpu in gpu_order)
    ):
        selected_by_gpu: dict[str, list[dict[str, Any]]] = {}
        remote_seconds = dict(base_remote_seconds)
        groups: list[dict[str, Any]] = []
        for gpu, count in zip(gpu_order, tail_counts):
            chosen = pending[gpu][-count:] if count else []
            selected_by_gpu[gpu] = chosen
            remote_seconds[gpu] -= sum(
                item["remote_seconds"] for item in chosen
            )
            if chosen:
                unselected_prefix = pending[gpu][:-count]
                deadline = fixed_seconds[gpu] + sum(
                    item["remote_seconds"] for item in unselected_prefix
                )
                groups.append(
                    {
                        "gpu": gpu,
                        "deadline_s": deadline,
                        "rows": chosen,
                        "local_seconds": sum(
                            item["local_seconds"] for item in chosen
                        )
                        * args.local_safety_factor,
                    }
                )

        local_seconds = sum(group["local_seconds"] for group in groups)
        makespan = max([local_seconds, *remote_seconds.values()])
        improvement_pct = (1.0 - makespan / baseline_makespan) * 100.0

        cumulative = 0.0
        safe = True
        deadline_margins: dict[str, float] = {}
        for group in sorted(groups, key=lambda value: value["deadline_s"]):
            cumulative += group["local_seconds"]
            margin = group["deadline_s"] - cumulative
            deadline_margins[group["gpu"]] = margin
            if margin < args.minimum_deadline_margin_s:
                safe = False
        if safe and improvement_pct >= args.minimum_improvement_pct:
            candidates.append(
                {
                    "tail_counts": dict(zip(gpu_order, tail_counts)),
                    "selected_by_gpu": selected_by_gpu,
                    "groups": groups,
                    "remote_seconds": remote_seconds,
                    "local_seconds": local_seconds,
                    "makespan": makespan,
                    "improvement_pct": improvement_pct,
                    "deadline_margins": deadline_margins,
                }
            )
    if not candidates:
        raise ValueError("no race-safe allocation passes the improvement gate")
    best = min(candidates, key=lambda value: value["makespan"])

    local_order: list[dict[str, Any]] = []
    for group in sorted(best["groups"], key=lambda value: value["deadline_s"]):
        # Longest first within an equal-deadline suffix reduces tail risk.
        for item in sorted(
            group["rows"],
            key=lambda value: (-value["local_seconds"], value["row"]["case_id"]),
        ):
            value = dict(item)
            value["deadline_s"] = group["deadline_s"]
            local_order.append(value)

    local_case_ids = {
        item["row"]["case_id"] for item in local_order
    }
    if len(local_case_ids) != len(local_order):
        raise ValueError("duplicate local case authority")

    allocation_rows: list[dict[str, Any]] = []
    order_lookup = {
        item["row"]["case_id"]: index
        for index, item in enumerate(local_order, start=1)
    }
    for gpu in gpu_order:
        for item in queues[gpu]:
            case_id = item["row"]["case_id"]
            owner = (
                "local"
                if case_id in local_case_ids
                else (
                    "remote_fixed"
                    if item["row"]["status"] != "not_run"
                    else "remote"
                )
            )
            allocation_rows.append(
                {
                    "case_id": case_id,
                    "object_key": item["row"]["object_key"],
                    "owner": owner,
                    "local_order": order_lookup.get(case_id, ""),
                    "remote_gpu": gpu,
                    "remote_queue_position": item["position"],
                    "status_at_allocation": item["row"]["status"],
                    "trajectory_frames": item["frames"],
                    "optimized_records_estimate": item["records_total"],
                    "predicted_remote_seconds": (
                        item["records_total"]
                        * remote_rate(item["row"]["object_key"])
                    ),
                    "predicted_local_seconds": (
                        item["records_total"]
                        * local_rate(item["row"]["object_key"])
                    ),
                    "remote_group_deadline_seconds": (
                        next(
                            (
                                group["deadline_s"]
                                for group in best["groups"]
                                if group["gpu"] == gpu
                            ),
                            "",
                        )
                        if owner == "local"
                        else ""
                    ),
                    "source_shard": f"gpu{gpu}.tsv",
                }
            )

    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)
    allocation_fields = list(allocation_rows[0])
    write_tsv(output / "allocation.tsv", allocation_rows, allocation_fields)

    row_manifest_root = output / "row_manifests"
    row_manifest_root.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(local_order, start=1):
        row = dict(full_by_case[item["row"]["case_id"]])
        row.update(
            {
                "status": "not_run",
                "failure_mode": "",
                "blocker_detail": "",
                "gpu_id": "0",
                "updated_at": datetime.now().astimezone().isoformat(
                    timespec="seconds"
                ),
            }
        )
        write_tsv(
            row_manifest_root
            / f"{index:02d}_{row['case_id']}.tsv",
            [row],
            full_fields,
        )

    summary = {
        "generated_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "status": "pass",
        "authority_rows": len(full_rows),
        "authority_object_counts": dict(
            Counter(row["object_key"] for row in full_rows)
        ),
        "remote_running_medians": [
            {
                "gpu": gpu,
                "object_key": object_key,
                "median_plan_time_s": value,
            }
            for gpu, object_key, value in running_medians
        ],
        "rates": {
            "remote_five_geom_seconds_per_record": remote_five,
            "remote_one_geom_seconds_per_record": remote_one,
            "local_five_geom_seconds_per_record": local_five,
            "local_one_geom_seconds_per_record": local_one,
            "local_safety_factor": args.local_safety_factor,
        },
        "baseline_remote_seconds_by_gpu": base_remote_seconds,
        "baseline_remote_makespan_s": baseline_makespan,
        "hybrid_remote_seconds_by_gpu": best["remote_seconds"],
        "hybrid_local_seconds": best["local_seconds"],
        "hybrid_makespan_s": best["makespan"],
        "predicted_improvement_pct": best["improvement_pct"],
        "tail_counts": best["tail_counts"],
        "deadline_margins_s": best["deadline_margins"],
        "local_case_ids_in_execution_order": [
            item["row"]["case_id"] for item in local_order
        ],
        "local_rows": len(local_order),
        "remote_rows": len(full_rows) - len(local_order),
        "minimum_deadline_margin_s": args.minimum_deadline_margin_s,
        "minimum_improvement_pct": args.minimum_improvement_pct,
    }
    (output / "allocation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "local_rows": summary["local_rows"],
                "remote_rows": summary["remote_rows"],
                "tail_counts": summary["tail_counts"],
                "predicted_improvement_pct": round(
                    summary["predicted_improvement_pct"], 2
                ),
                "baseline_makespan_h": round(
                    summary["baseline_remote_makespan_s"] / 3600, 3
                ),
                "hybrid_makespan_h": round(
                    summary["hybrid_makespan_s"] / 3600, 3
                ),
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
