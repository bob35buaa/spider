#!/usr/bin/env python3
"""Freeze plan212's fixed 4/6/5 E188 worker queues."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import (
    CANARY_BY_WORKER,
    PHYSICAL_GPU,
    QUEUE_CASES,
    RESULTS,
    S0,
    WORKERS,
    queue_case_set,
    read_tsv,
    relative,
    sha256,
    write_json,
    write_tsv,
)

AUTHORITY = S0 / "authority_manifest.tsv"
OVERRIDES = S0 / "overrides_manifest.json"
OUTPUT_ROOT = RESULTS / "s5_full/queue"
OUTPUT = OUTPUT_ROOT / "queue_manifest.json"


def build_payload() -> dict[str, Any]:
    authority_rows = read_tsv(AUTHORITY)
    authority = {row["case_id"]: row for row in authority_rows}
    overrides_payload = json.loads(OVERRIDES.read_text(encoding="utf-8"))
    overrides = {row["case_id"]: row for row in overrides_payload["rows"]}
    expected = queue_case_set()
    if set(authority) != expected or set(overrides) != expected:
        raise RuntimeError("authority/override sets differ from plan212 queue set")
    queues: dict[str, list[dict[str, Any]]] = {}
    for worker in WORKERS:
        rows = []
        for position, case_id in enumerate(QUEUE_CASES[worker], 1):
            source = authority[case_id]
            override = overrides[case_id]
            rows.append(
                {
                    "queue_position": position,
                    "ordinal": int(source["ordinal"]),
                    "e187_ordinal": int(source["e187_ordinal"]),
                    "case_id": case_id,
                    "object_key": source["object_key"],
                    "worker": worker,
                    "physical_gpu": PHYSICAL_GPU[worker],
                    "initial_status": "CANARY_NOT_RUN" if position == 1 else "NOT_RUN",
                    "canary": position == 1,
                    "predicted_wall_seconds": float(source["e187_predicted_wall_seconds"]),
                    "override_id": override["override_id"],
                    "override_path": override["override_path"],
                    "override_sha256": override["override_sha256"],
                    "target_task": source["target_task"],
                    "scene_act": source["scene_act"],
                    "scene_sha256": source["scene_sha256"],
                    "trajectory": source["trajectory"],
                    "trajectory_sha256": source["trajectory_sha256"],
                    "contact_mask": source["contact_mask"],
                    "contact_mask_sha256": source["contact_mask_sha256"],
                    "grid_manifest": source["grid_manifest"],
                    "grid_manifest_sha256": source["grid_manifest_sha256"],
                    "epsilon_grid_m": float(source["epsilon_grid_m"]),
                }
            )
        queues[worker] = rows
    flattened = [row["case_id"] for worker in WORKERS for row in queues[worker]]
    if len(flattened) != 15 or len(set(flattened)) != 15:
        raise RuntimeError("queue closure failure")
    return {
        "schema": "e188_mass5kg_full_queue_v1",
        "experiment_id": "E188",
        "status": "FROZEN",
        "worker_order": list(WORKERS),
        "physical_gpu_by_worker": PHYSICAL_GPU,
        "canary_by_worker": CANARY_BY_WORKER,
        "row_count": 15,
        "canary_count": 3,
        "remaining_count": 12,
        "source_authority": {"path": relative(AUTHORITY), "sha256": sha256(AUTHORITY)},
        "overrides": {"path": relative(OVERRIDES), "sha256": sha256(OVERRIDES)},
        "queues": queues,
    }


def freeze(payload: dict[str, Any]) -> None:
    fields = list(payload["queues"][WORKERS[0]][0])
    for worker in WORKERS:
        write_tsv(OUTPUT_ROOT / f"{worker}.tsv", [{field: row[field] for field in fields} for row in payload["queues"][worker]])
    payload["queue_tsv_sha256"] = {
        worker: sha256(OUTPUT_ROOT / f"{worker}.tsv") for worker in WORKERS
    }
    write_json(OUTPUT, payload, immutable=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "freeze"))
    args = parser.parse_args()
    payload = build_payload()
    if args.mode == "freeze":
        freeze(payload)
    print(json.dumps({"status": "PASS", "rows": payload["row_count"], "counts": {worker: len(payload["queues"][worker]) for worker in WORKERS}, "written": args.mode == "freeze"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
