#!/usr/bin/env python3
"""Freeze the E187 A3 canary/efficiency gate before any Full row starts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[5]
RESULTS = REPO / "workspace/core4d/results/E187"
A1_LOCK = RESULTS / "s2_canonical_grid_sdf/reward_grid_lock.json"
A2_LOCK = RESULTS / "s3_prg_audit/production_integration_lock.json"
C9_FAILURE = RESULTS / "s4_canary/efficiency_gate_local_failure.json"
C9_WAIVER = RESULTS / "s4_canary/c9_user_waiver/waiver_manifest.json"
OUTPUT = RESULTS / "s4_canary/canary_gate_lock.json"
DEPLOYMENT = RESULTS / "s4_canary/deployment/remote_deployment_manifest.json"
ASSIGNMENTS = {
    "bucket003_20231018_003_p1": "local-0",
    "bucket004_20231002_021_p1": "remote-0",
    "bucket007_20231020_055_p1": "remote-1",
}
EXPECTED_A1_SHA256 = "2cc949e19cb313e58883fd5d0446872220188ea9ecce06cb23d39e2a6b42194f"
EXPECTED_A2_SHA256 = "400c98b422ac4458eccffb589eecc1d776fe5cd3968eb855cd89a485bb7d5441"


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path, root: Path) -> str:
    """Return a repository-relative artifact path."""
    return path.absolute().relative_to(root.absolute()).as_posix()


def authority(root: Path = REPO) -> dict[str, Any]:
    """Validate frozen A1/A2/deployment authority before reading canaries."""
    a1_path = root / A1_LOCK.relative_to(REPO)
    a2_path = root / A2_LOCK.relative_to(REPO)
    deployment_path = root / DEPLOYMENT.relative_to(REPO)
    if sha256(a1_path) != EXPECTED_A1_SHA256:
        raise RuntimeError("A1 reward/grid lock SHA changed")
    if sha256(a2_path) != EXPECTED_A2_SHA256:
        raise RuntimeError("A2 production lock SHA changed")
    a1 = json.loads(a1_path.read_text(encoding="utf-8"))
    a2 = json.loads(a2_path.read_text(encoding="utf-8"))
    deployment = json.loads(deployment_path.read_text(encoding="utf-8"))
    if (
        a1.get("status") != "FROZEN"
        or a2.get("status") != "FROZEN"
        or a2.get("full_cem_started_rows") != 0
        or deployment.get("status") != "PASS"
    ):
        raise RuntimeError("A1/A2/deployment authority is not frozen before A3")
    if not all(a1["global_gates"].values()):
        raise RuntimeError("A1 global gate changed after freeze")
    return {"a1": a1, "a2": a2, "deployment": deployment}


def validate_canary(root: Path, case_id: str, worker: str) -> dict[str, Any]:
    """Validate one complete, recorder-off, Full-promotable canary row."""
    row_root = root / "workspace/core4d/results/E187/s4_canary/rows" / case_id
    manifest_path = row_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "status": "PASS",
        "case_id": case_id,
        "worker": worker,
        "gate0_technical_status": "FAIL",
        "progression_authority": "USER_WAIVED",
        "full_promotable": True,
        "a2_lock_sha256": EXPECTED_A2_SHA256,
        "budget": {"samples": 1024, "iterations": 32, "seed": 0},
    }
    mismatches = {
        key: [manifest.get(key), value]
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"{case_id}: canary manifest mismatch: {mismatches}")
    artifacts: dict[str, Any] = {}
    for key in ("result", "config", "video", "log"):
        artifact = root / manifest[key]["path"]
        if sha256(artifact) != manifest[key]["sha256"]:
            raise RuntimeError(f"{case_id}: {key} artifact SHA changed")
        artifacts[key] = manifest[key]
    config = yaml.safe_load((root / manifest["config"]["path"]).read_text())
    if (
        config.get("query_tape_enabled") is not False
        or config.get("query_tape_record_geometry_state") is not False
        or config.get("surface_band_score_mode") != "distance_continuation"
    ):
        raise RuntimeError(f"{case_id}: recorder/reward config mismatch")
    if any(path.name == "raw_chunks" for path in row_root.rglob("raw_chunks")):
        raise RuntimeError(f"{case_id}: query recorder output exists")
    if (
        not manifest["video_contract"]["readable"]
        or float(manifest["video_contract"]["duration_seconds"]) <= 0.0
        or int(manifest["numeric_array_count"]) < 1
        or int(manifest["plan_time_count"]) < 1
        or int(manifest["peak_total_gpu_memory_mib"]) > 6144
    ):
        raise RuntimeError(f"{case_id}: runtime/video/memory gate failed")
    return {
        "case_id": case_id,
        "worker": worker,
        "manifest": {
            "path": relative(manifest_path, root),
            "sha256": sha256(manifest_path),
        },
        "wall_seconds": manifest["wall_seconds"],
        "peak_total_gpu_memory_mib": manifest["peak_total_gpu_memory_mib"],
        "plan_time_median_seconds": manifest["plan_time_median_seconds"],
        "a2_lock_sha256": manifest["a2_lock_sha256"],
        "override_sha256": manifest["override_sha256"],
        "scene_sha256": manifest["scene_sha256"],
        "grid_manifest_sha256": manifest["grid_manifest_sha256"],
        "artifacts": artifacts,
        "full_promotable": True,
    }


def build_payload(root: Path = REPO) -> dict[str, Any]:
    """Build the A3 lock when C8 passes and the measured C9 failure is waived."""
    frozen = authority(root)
    failure_path = root / C9_FAILURE.relative_to(REPO)
    waiver_path = root / C9_WAIVER.relative_to(REPO)
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    waiver = json.loads(waiver_path.read_text(encoding="utf-8"))
    if (
        failure.get("status") != "FAIL"
        or failure.get("stop_rule") != "PLAN_TIME_RATIO_GT_2P0"
        or float(failure.get("plan_time_ratio", 0.0)) <= 2.0
    ):
        raise RuntimeError("C9 technical failure evidence changed")
    if (
        waiver.get("status") != "FROZEN"
        or waiver.get("scope") != "C9_PLAN_TIME_RATIO_STOP_GATE_ONLY"
        or waiver["c9_failure"]["sha256"] != sha256(failure_path)
        or waiver.get("full_cem_started_rows") != 0
    ):
        raise RuntimeError("C9 user waiver authority is invalid")
    canaries = [
        validate_canary(root, case_id, worker)
        for case_id, worker in ASSIGNMENTS.items()
    ]
    a1 = frozen["a1"]
    deployment = frozen["deployment"]
    return {
        "schema": "e187_a3_canary_gate_lock_v1",
        "experiment_id": "E187",
        "stage": "A3_S4_CANARY_GATE_LOCK",
        "status": "FROZEN",
        "gate0_technical_status": "FAIL",
        "progression_authority": "USER_WAIVED",
        "a1_lock": {"path": relative(A1_LOCK, REPO), "sha256": EXPECTED_A1_SHA256},
        "a2_lock": {"path": relative(A2_LOCK, REPO), "sha256": EXPECTED_A2_SHA256},
        "remote_deployment": {
            "path": relative(DEPLOYMENT, REPO),
            "sha256": sha256(root / DEPLOYMENT.relative_to(REPO)),
            "source_snapshot_sha256": deployment["source_snapshot_sha256"],
            "shared_checkout_mutated": deployment["shared_checkout_mutated"],
            "existing_processes_modified": deployment["existing_processes_modified"],
        },
        "canaries": canaries,
        "efficiency": {
            "technical_status": "FAIL",
            "failure_path": relative(C9_FAILURE, REPO),
            "failure_sha256": sha256(failure_path),
            "measured_plan_time_ratio": failure["plan_time_ratio"],
            "waiver_path": relative(C9_WAIVER, REPO),
            "waiver_sha256": sha256(waiver_path),
            "progression_authority": "USER_WAIVED",
            "max_peak_total_gpu_memory_mib": max(
                int(row["peak_total_gpu_memory_mib"]) for row in canaries
            ),
            "max_query_kernel_ratio": max(
                float(value["metrics"]["query_kernel_ratio"])
                for value in a1["objects"].values()
            ),
            "max_query_peak_allocated_mib": max(
                float(value["metrics"]["query_peak_allocated_mib"])
                for value in a1["objects"].values()
            ),
        },
        "c8_pass": True,
        "c9_technical_pass": False,
        "c9_progression_authority": "USER_WAIVED",
        "c9_progression_allowed": True,
        "full_cem_started_rows": 0,
        "next_gate": "A4_S5_KEEP22_FULL_CEM",
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write the lock atomically and refuse replacement."""
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.read_text(encoding="utf-8") != serialized:
            raise RuntimeError(f"refusing to replace A3 lock: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    """Preflight immutable inputs or freeze the completed A3 gate."""
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "freeze"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.mode == "preflight":
        frozen = authority(REPO)
        payload = {
            "status": "PREFLIGHT_PASS",
            "a1_status": frozen["a1"]["status"],
            "a2_status": frozen["a2"]["status"],
            "full_cem_started_rows": frozen["a2"]["full_cem_started_rows"],
            "expected_canaries": ASSIGNMENTS,
        }
    else:
        payload = build_payload(REPO)
        output = args.output if args.output.is_absolute() else REPO / args.output
        atomic_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
