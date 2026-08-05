#!/usr/bin/env python3
"""Freeze the user-authorized E188 speed-rebalanced remaining queue (v2)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import CANARY_BY_WORKER, PHYSICAL_GPU, REPO, RESULTS, WORKERS, read_tsv, relative, sha256, write_json, write_tsv

V1_ROOT = RESULTS / "s5_full/queue"
V1_MANIFEST = V1_ROOT / "queue_manifest.json"
OUTPUT_ROOT = RESULTS / "s5_full/queue_speed_rebalanced_v2"
OUTPUT = OUTPUT_ROOT / "queue_manifest.json"
E187_ROWS = RESULTS.parent / "E187/s5_full/rows"
CANARY_ROWS = RESULTS / "s4_canary/rows"
PLAN_SECONDS = {"local-0": 27.9845, "a100-4": 73.05815, "a100-5": 69.1504}
REMAINING = {
    "local-0": (
        "bucket007_20231023_073_p1",
        "bucket007_20231018_021_p2",
        "bucket003_20231020_064_p1",
        "bucket007_20231023_075_p2",
        "bucket007_20231018_021_p1",
        "bucket007_20231018_019_p2",
    ),
    "a100-4": (
        "bucket007_20231023_075_p1",
        "bucket007_20231018_019_p1",
        "bucket007_20231003_2_023_p1",
    ),
    "a100-5": (
        "bucket007_20231003_1_021_p2",
        "bucket007_20231020_059_p1",
        "bucket007_20231003_1_021_p1",
    ),
}
PRESERVED_LOCAL4 = {
    "bucket003_20231018_003_p1",
    "bucket007_20231018_021_p2",
    "bucket007_20231018_021_p1",
    "bucket007_20231018_019_p2",
}


def plan_time_count(case_id: str) -> int:
    canary = CANARY_ROWS / case_id / "manifest.json"
    source = canary if canary.is_file() else E187_ROWS / case_id / "manifest.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("case_id") != case_id or int(payload["plan_time_count"]) <= 0:
        raise RuntimeError(f"invalid timing authority: {source}")
    return int(payload["plan_time_count"])


def build_payload() -> dict[str, Any]:
    v1 = json.loads(V1_MANIFEST.read_text(encoding="utf-8"))
    source_rows = {
        row["case_id"]: row
        for worker in WORKERS
        for row in v1["queues"][worker]
    }
    queues: dict[str, list[dict[str, Any]]] = {}
    for worker in WORKERS:
        case_ids = (CANARY_BY_WORKER[worker], *REMAINING[worker])
        rows = []
        for position, case_id in enumerate(case_ids, 1):
            row = dict(source_rows[case_id])
            count = plan_time_count(case_id)
            row.update(
                {
                    "queue_position": position,
                    "worker": worker,
                    "physical_gpu": PHYSICAL_GPU[worker],
                    "initial_status": "CANARY_NOT_RUN" if position == 1 else "NOT_RUN",
                    "canary": position == 1,
                    "predicted_plan_time_count": count,
                    "device_plan_time_seconds": PLAN_SECONDS[worker],
                    "predicted_e188_wall_seconds": count * PLAN_SECONDS[worker],
                }
            )
            rows.append(row)
        queues[worker] = rows
    flattened = [row["case_id"] for worker in WORKERS for row in queues[worker]]
    if len(flattened) != len(set(flattened)) != 15:
        raise RuntimeError("unreachable queue closure")
    if len(flattened) != 15 or len(set(flattened)) != 15 or set(flattened) != set(source_rows):
        raise RuntimeError("v2 queue is not the same 15-case authority")
    if not PRESERVED_LOCAL4 <= {row["case_id"] for row in queues["local-0"]}:
        raise RuntimeError("v2 queue lost the four same-device local evidence rows")
    predicted = {
        worker: sum(float(row["predicted_e188_wall_seconds"]) for row in queues[worker][1:])
        for worker in WORKERS
    }
    return {
        "schema": "e188_mass5kg_speed_rebalanced_full_queue_v2",
        "experiment_id": "E188",
        "status": "FROZEN",
        "queue_revision": 2,
        "worker_order": list(WORKERS),
        "physical_gpu_by_worker": PHYSICAL_GPU,
        "canary_by_worker": CANARY_BY_WORKER,
        "row_count": 15,
        "canary_count": 3,
        "remaining_count": 12,
        "remaining_counts_by_worker": {worker: len(REMAINING[worker]) for worker in WORKERS},
        "predicted_remaining_wall_seconds": predicted,
        "speed_authority_seconds_per_plan_step": PLAN_SECONDS,
        "scheduling_objective": "minimize_makespan_with_original_local4_preserved",
        "gpu_overlap_policy": "USER_AUTHORIZED_NO_MEMORY_OR_COMPUTE_PROCESS_GATE",
        "source_v1_queue": {"path": relative(V1_MANIFEST), "sha256": sha256(V1_MANIFEST)},
        "canary_gate": {"path": relative(RESULTS / "s4_canary/canary_gate.json"), "sha256": sha256(RESULTS / "s4_canary/canary_gate.json")},
        "promotion": {"path": relative(RESULTS / "s5_full/promotion_manifest.json"), "sha256": sha256(RESULTS / "s5_full/promotion_manifest.json")},
        "queues": queues,
    }


def freeze(payload: dict[str, Any]) -> None:
    fields = list(payload["queues"][WORKERS[0]][0])
    for worker in WORKERS:
        write_tsv(OUTPUT_ROOT / f"{worker}.tsv", [{field: row[field] for field in fields} for row in payload["queues"][worker]])
    payload["queue_tsv_sha256"] = {worker: sha256(OUTPUT_ROOT / f"{worker}.tsv") for worker in WORKERS}
    write_json(OUTPUT, payload, immutable=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "freeze"))
    args = parser.parse_args()
    payload = build_payload()
    if args.mode == "freeze":
        freeze(payload)
    print(json.dumps({"status": "PASS", "counts": {worker: len(payload["queues"][worker]) for worker in WORKERS}, "predicted_hours": {worker: payload["predicted_remaining_wall_seconds"][worker] / 3600 for worker in WORKERS}, "written": args.mode == "freeze"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
