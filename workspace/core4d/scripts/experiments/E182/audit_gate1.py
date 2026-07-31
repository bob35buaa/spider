#!/usr/bin/env python3
"""Aggregate the complete dev3 evidence required by E182 Gate1."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from build_prg_query_tape import verify_case_query_tape
from e182_common import atomic_json, relative_to_repo, sha256_file
from run_query_tape_replay import DEFAULT_RESULT_ROOT
from runtime_inputs import verify_runtime_inputs

REPO_ROOT = Path(__file__).resolve().parents[5]
EXPECTED_CASE_IDS = (
    "bucket003_20231018_001_p1",
    "bucket004_20231002_021_p1",
    "bucket007_20231020_055_p1",
)
DIRECT_TEST_FILES = (
    "test_query_tape.py",
    "test_query_tape_pipeline.py",
    "test_runtime_inputs.py",
    "test_prg_query_tape.py",
    "test_preflight.py",
)
IMPLEMENTATION_FILES = (
    "audit_gate1.py",
    "build_prg_query_tape.py",
    "audit_query_tape_replay.py",
    "run_query_tape_replay.py",
)


def _error(errors: list[dict[str, Any]], field: str, **values: Any) -> None:
    """Append one structured Gate1 error."""
    errors.append({"field": field, **values})


def audit_prg_case(
    tape_root: Path,
    same_run_row: dict[str, Any],
    *,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Tie one factored P/R/G tape exactly to its frozen on_a replay."""
    case_id = same_run_row["case_id"]
    manifest_path = tape_root / case_id / "manifest.json"
    errors: list[dict[str, Any]] = []
    if payload is None:
        if not manifest_path.is_file():
            return {
                "case_id": case_id,
                "status": "FAIL",
                "selection_mode": None,
                "errors": [{"field": "manifest", "error": "missing"}],
            }
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    if same_run_row.get("status") != "PASS":
        _error(errors, "same_run_status", actual=same_run_row.get("status"))
    if payload.get("case_id") != case_id:
        _error(errors, "case_id", actual=payload.get("case_id"), expected=case_id)
    if payload.get("selection_cem_mode") != "on_a":
        _error(
            errors,
            "selection_cem_mode",
            actual=payload.get("selection_cem_mode"),
            expected="on_a",
        )

    same_run_on_a = (
        same_run_row.get("hard_gate", {}).get("same_run_integrity", {}).get("on_a", {})
    )
    expected_content_sha = same_run_on_a.get("content_sha256")
    if payload.get("raw_cem_content_sha256") != expected_content_sha:
        _error(
            errors,
            "selection_content_sha256",
            actual=payload.get("raw_cem_content_sha256"),
            expected=expected_content_sha,
        )
    source_counts = payload.get("source_chunk_counts", {})
    expected_cem_chunks = same_run_on_a.get("chunk_count")
    expected_source_counts = {
        "reference": 1,
        "e178_final": 1,
        "cem_on_a": expected_cem_chunks,
    }
    if source_counts != expected_source_counts:
        _error(
            errors,
            "source_chunk_counts",
            actual=source_counts,
            expected=expected_source_counts,
        )
    if payload.get("chunk_count") != (expected_cem_chunks or 0) + 2:
        _error(
            errors,
            "chunk_count",
            actual=payload.get("chunk_count"),
            expected=(expected_cem_chunks or 0) + 2,
        )

    consumers = payload.get("consumer_names", [])
    consumer_ids = payload.get("consumer_geom_ids", {})
    if not consumer_ids.get("P_collision"):
        _error(errors, "P_collision", error="empty")
    if not any(name.startswith("R_") for name in consumers):
        _error(errors, "R_consumers", error="missing")
    if not any(name.startswith("G_") for name in consumers):
        _error(errors, "G_consumers", error="missing")
    if payload.get("points_per_pose", 0) <= 0:
        _error(errors, "points_per_pose", actual=payload.get("points_per_pose"))

    verifier = verify_case_query_tape(tape_root, payload)
    if verifier["status"] != "PASS":
        _error(
            errors,
            "prg_tape_verifier",
            mismatch_count=verifier["mismatch_count"],
            mismatches=verifier["mismatches"],
        )
    return {
        "case_id": case_id,
        "status": "PASS" if not errors else "FAIL",
        "selection_mode": payload.get("selection_cem_mode"),
        "selection_content_sha256": payload.get("raw_cem_content_sha256"),
        "manifest": (
            {
                "path": relative_to_repo(manifest_path),
                "sha256": sha256_file(manifest_path),
            }
            if manifest_path.is_file()
            else None
        ),
        "chunk_count": payload.get("chunk_count"),
        "source_chunk_counts": source_counts,
        "stored_size_bytes": payload.get("stored_size_bytes"),
        "estimated_expanded_point_bytes": payload.get("estimated_expanded_point_bytes"),
        "consumer_names": consumers,
        "physics_geom_count": len(consumer_ids.get("P_collision", [])),
        "points_per_pose": payload.get("points_per_pose"),
        "verifier": verifier,
        "errors": errors,
    }


def _run_direct_tests() -> dict[str, Any]:
    """Run the frozen default-off/mock/integrity tests and record their evidence."""
    script_root = Path(__file__).resolve().parent
    rows = []
    for filename in DIRECT_TEST_FILES:
        path = script_root / filename
        command = ["uv", "run", "python", relative_to_repo(path)]
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        output = completed.stdout + completed.stderr
        rows.append(
            {
                "test": filename,
                "source": {
                    "path": relative_to_repo(path),
                    "sha256": sha256_file(path),
                },
                "command": command,
                "exit_code": completed.returncode,
                "status": "PASS" if completed.returncode == 0 else "FAIL",
                "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
                "output_tail": output.splitlines()[-8:],
            }
        )
    return {
        "status": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL",
        "rows": rows,
    }


def audit_gate1_artifacts(
    result_root: Path = DEFAULT_RESULT_ROOT,
    *,
    run_direct_tests: bool = True,
) -> dict[str, Any]:
    """Audit runtime, same-run, factored query tapes, selection identity, and tests."""
    same_run_path = result_root / "same_run_integrity_audit.json"
    runtime_manifest = result_root / "runtime_inputs_manifest.json"
    errors: list[dict[str, Any]] = []
    if not same_run_path.is_file():
        raise RuntimeError(f"same-run aggregate is missing: {same_run_path}")
    same_run = json.loads(same_run_path.read_text(encoding="utf-8"))
    rows_by_case = {row["case_id"]: row for row in same_run.get("rows", [])}
    actual_case_ids = tuple(row["case_id"] for row in same_run.get("rows", []))
    if actual_case_ids != EXPECTED_CASE_IDS:
        _error(
            errors,
            "same_run_case_order",
            actual=list(actual_case_ids),
            expected=list(EXPECTED_CASE_IDS),
        )
    if same_run.get("status") != "PASS":
        _error(errors, "same_run_status", actual=same_run.get("status"))

    runtime = verify_runtime_inputs(REPO_ROOT, runtime_manifest)
    if runtime["status"] != "PASS":
        _error(errors, "runtime_inputs", actual=runtime)

    tape_root = result_root / "prg_query_tape"
    cases = []
    for case_id in EXPECTED_CASE_IDS:
        row = rows_by_case.get(case_id)
        if row is None:
            cases.append(
                {
                    "case_id": case_id,
                    "status": "FAIL",
                    "selection_mode": None,
                    "errors": [{"field": "same_run_row", "error": "missing"}],
                }
            )
            continue
        cases.append(audit_prg_case(tape_root, row))
    if not all(case["status"] == "PASS" for case in cases):
        _error(
            errors,
            "case_artifacts",
            failed=[case["case_id"] for case in cases if case["status"] != "PASS"],
        )

    direct_tests = (
        _run_direct_tests() if run_direct_tests else {"status": "SKIPPED", "rows": []}
    )
    if run_direct_tests and direct_tests["status"] != "PASS":
        _error(errors, "direct_tests", actual=direct_tests["status"])
    selection_identity = [
        {
            "case_id": case["case_id"],
            "manifest_sha256": (case.get("manifest") or {}).get("sha256"),
            "content_sha256": case.get("selection_content_sha256"),
        }
        for case in cases
    ]
    selection_set_sha256 = hashlib.sha256(
        json.dumps(
            selection_identity,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return {
        "experiment_id": "E182",
        "stage": "S1_gate1_aggregate",
        "gate": "Gate1_real_query_tape",
        "status": "PASS" if not errors else "FAIL",
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "cross_run_policy": "REPORT_ONLY_CUDA_NON_BITWISE",
        "selection_policy": "ON_A_ONLY_ON_B_DIAGNOSTIC",
        "case_ids": list(EXPECTED_CASE_IDS),
        "same_run_status": same_run.get("status"),
        "same_run_audit": {
            "path": relative_to_repo(same_run_path),
            "sha256": sha256_file(same_run_path),
        },
        "runtime_inputs": runtime,
        "cases": cases,
        "selection_identity": selection_identity,
        "selection_set_sha256": selection_set_sha256,
        "implementation_sources": [
            {
                "path": relative_to_repo(Path(__file__).resolve().parent / filename),
                "sha256": sha256_file(Path(__file__).resolve().parent / filename),
            }
            for filename in IMPLEMENTATION_FILES
        ],
        "direct_tests": direct_tests,
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    """Parse Gate1 audit arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-direct-tests", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Write the aggregate Gate1 artifact and return a gate-compatible exit code."""
    args = parse_args()
    payload = audit_gate1_artifacts(
        args.result_root,
        run_direct_tests=not args.skip_direct_tests,
    )
    output = args.output or args.result_root / "gate1_audit.json"
    atomic_json(output, payload)
    print(
        f"E182_GATE1={payload['status']} cases={len(payload['cases'])} "
        f"selection_sha={payload['selection_set_sha256']}"
    )
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
