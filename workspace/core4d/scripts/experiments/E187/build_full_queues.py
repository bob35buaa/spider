#!/usr/bin/env python3
"""Build immutable three-worker LPT queues after the E187 A3 lock passes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[5]
RESULTS = REPO / "workspace/core4d/results/E187"
A3_LOCK = RESULTS / "s4_canary/canary_gate_lock.json"
OVERRIDES = RESULTS / "s3_prg_audit/production_overrides.json"
SCENES = (
    REPO
    / "workspace/core4d/results/E186/s2_compound_physics/compound_scene_manifest.tsv"
)
OUTPUT_ROOT = RESULTS / "s5_full/queue"
OUTPUT_MANIFEST = OUTPUT_ROOT / "queue_manifest.json"
WORKERS = ("local-0", "remote-0", "remote-1")
CANARY_BY_WORKER = {
    "local-0": "bucket003_20231018_003_p1",
    "remote-0": "bucket004_20231002_021_p1",
    "remote-1": "bucket007_20231020_055_p1",
}


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    """Read one tab-separated authority file."""
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def trajectory_frames(path: Path) -> int:
    """Return the qpos frame count from one frozen trajectory."""
    with np.load(path, allow_pickle=False) as payload:
        frames = int(payload["qpos"].shape[0])
    if frames < 1:
        raise RuntimeError(f"empty trajectory: {path}")
    return frames


def lpt_assign(
    rows: list[dict[str, Any]],
    canary_by_worker: dict[str, str],
    representative_wall: dict[str, float],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, float]]:
    """Assign remaining rows by LPT after fixing canaries as queue position one."""
    by_case = {row["case_id"]: row for row in rows}
    queues: dict[str, list[dict[str, Any]]] = {worker: [] for worker in WORKERS}
    loads = {worker: float(representative_wall[worker]) for worker in WORKERS}
    canary_cases = set(canary_by_worker.values())
    for worker in WORKERS:
        case_id = canary_by_worker[worker]
        row = dict(by_case[case_id])
        row.update(
            {
                "queue_position": 1,
                "worker": worker,
                "predicted_wall_seconds": representative_wall[worker],
                "initial_status": "PROMOTED_CANARY_PENDING_ATOMIC_REGISTRATION",
                "promoted_canary": True,
            }
        )
        queues[worker].append(row)
    remaining = sorted(
        (row for row in rows if row["case_id"] not in canary_cases),
        key=lambda row: (-float(row["predicted_wall_seconds"]), int(row["ordinal"])),
    )
    worker_order = {worker: index for index, worker in enumerate(WORKERS)}
    for source in remaining:
        worker = min(WORKERS, key=lambda value: (loads[value], worker_order[value]))
        row = dict(source)
        row.update(
            {
                "queue_position": len(queues[worker]) + 1,
                "worker": worker,
                "initial_status": "NOT_RUN",
                "promoted_canary": False,
            }
        )
        queues[worker].append(row)
        loads[worker] += float(row["predicted_wall_seconds"])
    return queues, loads


def authority_rows(root: Path = REPO) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build all 22 queue rows from A2 authority and frozen trajectories."""
    overrides_path = root / OVERRIDES.relative_to(REPO)
    scenes_path = root / SCENES.relative_to(REPO)
    overrides = json.loads(overrides_path.read_text(encoding="utf-8"))
    if overrides.get("status") != "PASS" or overrides.get("row_count") != 22:
        raise RuntimeError("production override authority is not exactly 22 PASS rows")
    scenes = {row["case_id"]: row for row in read_tsv(scenes_path)}
    rows: list[dict[str, Any]] = []
    for override in overrides["rows"]:
        case_id = override["case_id"]
        scene = scenes[case_id]
        trajectory = root / scene["trajectory"]
        if sha256(trajectory) != scene["trajectory_sha256"]:
            raise RuntimeError(f"{case_id}: trajectory SHA changed before LPT")
        rows.append(
            {
                "ordinal": int(override["ordinal"]),
                "case_id": case_id,
                "object_key": override["object_key"],
                "override_id": override["override_id"],
                "override_path": override["override_path"],
                "override_sha256": override["override_sha256"],
                "target_task": scene["target_task"],
                "scene_sha256": scene["effective_scene_sha256"],
                "trajectory": scene["trajectory"],
                "trajectory_sha256": scene["trajectory_sha256"],
                "trajectory_frames": trajectory_frames(trajectory),
                "grid_manifest": override["grid_manifest"],
                "grid_manifest_sha256": override["grid_manifest_sha256"],
                "epsilon_grid_m": override["epsilon_grid_m"],
            }
        )
    if [row["ordinal"] for row in rows] != list(range(1, 23)):
        raise RuntimeError("production row order is not the frozen 1..22 authority")
    return rows, overrides


def build_payload(root: Path = REPO) -> dict[str, Any]:
    """Build the immutable queue payload only after A3 is frozen."""
    a3_path = root / A3_LOCK.relative_to(REPO)
    a3 = json.loads(a3_path.read_text(encoding="utf-8"))
    if (
        a3.get("status") != "FROZEN"
        or not a3.get("c8_pass")
        or not a3.get("c9_progression_allowed")
        or a3.get("full_cem_started_rows") != 0
    ):
        raise RuntimeError("A3 lock did not authorize Full queue generation")
    canary_by_case = {row["case_id"]: row for row in a3["canaries"]}
    if set(canary_by_case) != set(CANARY_BY_WORKER.values()):
        raise RuntimeError("A3 canary set changed before LPT")
    rows, overrides = authority_rows(root)
    representative_wall = {
        worker: float(canary_by_case[case_id]["wall_seconds"])
        for worker, case_id in CANARY_BY_WORKER.items()
    }
    representative_frames = {
        row["object_key"]: int(row["trajectory_frames"])
        for row in rows
        if row["case_id"] in canary_by_case
    }
    representative_wall_by_object = {
        canary_by_case[case_id]["case_id"].split("_", 1)[0]: representative_wall[worker]
        for worker, case_id in CANARY_BY_WORKER.items()
    }
    for row in rows:
        row["predicted_wall_seconds"] = representative_wall_by_object[
            row["object_key"]
        ] * (int(row["trajectory_frames"]) / representative_frames[row["object_key"]])
    queues, predicted_loads = lpt_assign(rows, CANARY_BY_WORKER, representative_wall)
    all_cases = [row["case_id"] for queue in queues.values() for row in queue]
    if len(all_cases) != 22 or len(set(all_cases)) != 22:
        raise RuntimeError("LPT queues do not contain 22 unique rows")
    return {
        "schema": "e187_keep22_full_queue_v1",
        "experiment_id": "E187",
        "stage": "A4_S5_KEEP22_FULL_CEM_QUEUE",
        "status": "FROZEN",
        "gate0_technical_status": "FAIL",
        "progression_authority": "USER_WAIVED",
        "a3_lock": {
            "path": A3_LOCK.relative_to(REPO).as_posix(),
            "sha256": sha256(a3_path),
        },
        "production_overrides": {
            "path": OVERRIDES.relative_to(REPO).as_posix(),
            "sha256": sha256(root / OVERRIDES.relative_to(REPO)),
            "row_count": overrides["row_count"],
        },
        "algorithm": "LPT_REPRESENTATIVE_WALL_SCALED_BY_TRAJECTORY_FRAMES",
        "worker_order": list(WORKERS),
        "canary_by_worker": CANARY_BY_WORKER,
        "representative_wall_seconds": representative_wall,
        "predicted_load_seconds": predicted_loads,
        "row_count": 22,
        "promoted_canary_count": 3,
        "not_run_count": 19,
        "queues": queues,
        "full_cem_started_rows": 0,
    }


def immutable_text(path: Path, value: str) -> None:
    """Write one queue artifact atomically without replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.read_text(encoding="utf-8") != value:
            raise RuntimeError(f"refusing to replace frozen queue artifact: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def write_payload(payload: dict[str, Any], output: Path = OUTPUT_MANIFEST) -> None:
    """Write worker TSVs first and the authority manifest last."""
    fields = (
        "queue_position",
        "ordinal",
        "case_id",
        "object_key",
        "worker",
        "initial_status",
        "promoted_canary",
        "predicted_wall_seconds",
        "trajectory_frames",
        "override_id",
        "override_path",
        "override_sha256",
        "target_task",
        "scene_sha256",
        "trajectory",
        "trajectory_sha256",
        "grid_manifest",
        "grid_manifest_sha256",
        "epsilon_grid_m",
    )
    for worker, rows in payload["queues"].items():
        lines = ["\t".join(fields)]
        for row in rows:
            lines.append("\t".join(str(row[field]) for field in fields))
        immutable_text(output.parent / f"{worker}.tsv", "\n".join(lines) + "\n")
    immutable_text(output, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    """Preflight 22-row authority or freeze LPT queues after A3."""
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "freeze"))
    parser.add_argument("--output", type=Path, default=OUTPUT_MANIFEST)
    args = parser.parse_args()
    if args.mode == "preflight":
        rows, overrides = authority_rows(REPO)
        payload = {
            "status": "PREFLIGHT_PASS",
            "row_count": len(rows),
            "override_status": overrides["status"],
            "a3_lock_exists": A3_LOCK.is_file(),
            "full_cem_started_rows": 0,
        }
    else:
        payload = build_payload(REPO)
        output = args.output if args.output.is_absolute() else REPO / args.output
        write_payload(payload, output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
