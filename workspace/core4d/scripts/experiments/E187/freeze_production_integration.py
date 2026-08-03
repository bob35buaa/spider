#!/usr/bin/env python3
"""Freeze the completed E187 A2 production-integration evidence."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[5]
ROOT = REPO / "workspace/core4d/results/E187/s3_prg_audit"
REWARD_GRID_LOCK = (
    REPO / "workspace/core4d/results/E187/s2_canonical_grid_sdf/reward_grid_lock.json"
)
OVERRIDES = ROOT / "production_overrides.json"
INTEGRATION = ROOT / "production_integration.json"
GATE_AGGREGATE = ROOT / "reference_final_gate/aggregate.json"
GATE_TAPE = ROOT / "reference_final_gate/tape_metrics.tsv"
OUTPUT = ROOT / "production_integration_lock.json"
SNAPSHOTS = tuple(
    ROOT / "process_snapshots" / f"local_{kind}_{when}.csv"
    for kind in ("gpu", "compute")
    for when in ("before", "after")
)


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    """Serialize one repository-relative path."""
    return path.absolute().relative_to(REPO.absolute()).as_posix()


def artifact(path: Path) -> dict[str, str]:
    """Return one frozen path/SHA record."""
    return {"path": relative(path), "sha256": sha256(path)}


def build_payload() -> dict[str, Any]:
    """Validate A2 gates and build the deterministic lock payload."""
    reward_grid = json.loads(REWARD_GRID_LOCK.read_text(encoding="utf-8"))
    overrides = json.loads(OVERRIDES.read_text(encoding="utf-8"))
    integration = json.loads(INTEGRATION.read_text(encoding="utf-8"))
    gate = json.loads(GATE_AGGREGATE.read_text(encoding="utf-8"))
    if reward_grid.get("status") != "FROZEN":
        raise RuntimeError("A1 reward/grid lock is not frozen")
    if overrides.get("status") != "PASS" or overrides.get("row_count") != 22:
        raise RuntimeError("A2 production overrides are not 22/22 PASS")
    if integration.get("status") != "PASS" or integration.get("row_count") != 22:
        raise RuntimeError("A2 production compile is not 22/22 PASS")
    recorder = integration.get("recorder_off", {})
    if recorder.get("status") != "PASS" or (ROOT / "raw_chunks").exists():
        raise RuntimeError("A2 recorder-off contract failed")
    if (
        gate.get("status") != "PASS"
        or gate.get("tape_count") != 44
        or gate.get("finite_tape_count") != 44
        or gate.get("false_safe_accept_total") != 0
        or gate["tape_metrics"]["sha256"] != sha256(GATE_TAPE)
    ):
        raise RuntimeError("A2 44-tape G contract failed")
    if integration["production_overrides"]["sha256"] != sha256(OVERRIDES):
        raise RuntimeError("A2 override SHA linkage failed")
    return {
        "schema": "e187_production_integration_lock_v1",
        "experiment_id": "E187",
        "stage": "A2_S3_PRODUCTION_INTEGRATION_LOCK",
        "status": "FROZEN",
        "gate0_technical_status": "FAIL",
        "progression_authority": "USER_WAIVED",
        "reward_grid_lock": artifact(REWARD_GRID_LOCK),
        "production_overrides": {
            **artifact(OVERRIDES),
            "row_count": overrides["row_count"],
            "allowed_effective_diff": overrides["allowed_effective_diff"],
        },
        "production_integration": {
            **artifact(INTEGRATION),
            "row_count": integration["row_count"],
            "recorder_off": recorder,
        },
        "reference_final_gate": {
            "aggregate": artifact(GATE_AGGREGATE),
            "tape_metrics": artifact(GATE_TAPE),
            "tape_count": gate["tape_count"],
            "finite_tape_count": gate["finite_tape_count"],
            "false_safe_accept_total": gate["false_safe_accept_total"],
        },
        "process_snapshots": [artifact(path) for path in SNAPSHOTS],
        "full_cem_started_rows": 0,
        "next_gate": "A3_S4_THREE_DEVICE_CANARY",
    }


def write_immutable(path: Path, payload: bytes) -> None:
    """Create the A2 lock or verify byte identity."""
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"immutable A2 lock mismatch: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def main() -> int:
    """Freeze the A2 integration lock."""
    payload = build_payload()
    write_immutable(
        OUTPUT, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
