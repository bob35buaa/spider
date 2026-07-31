#!/usr/bin/env python3
"""Direct-main tests for the E182 S0 authority and environment contracts."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

from audit_preflight import evaluate_gate
from build_authority import (
    DEFAULT_SOURCE,
    DEV_CASE_IDS,
    EXPECTED_SOURCE_SHA256,
    build_authority,
)
from deploy_remote_snapshot import build_source_manifest, derive_remote_root
from probe_environment import (
    compile_probe,
    dependency_versions,
    parse_gpu_csv,
    validate_gpu_contract,
)


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a TSV authority projection."""
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def test_authority_projection() -> None:
    """Authority must preserve E178 order, budgets, and dev isolation."""
    with tempfile.TemporaryDirectory(prefix="e182_authority_test_") as directory:
        root = Path(directory)
        manifest = build_authority(
            source_path=DEFAULT_SOURCE,
            output_root=root / "authority",
            snapshot_root=root / "scene_snapshot",
            expected_sha256=EXPECTED_SOURCE_SHA256,
            copy_snapshots=False,
        )
        full_rows = read_rows(root / "authority/full27/manifest.tsv")
        dev_rows = read_rows(root / "authority/dev3/manifest.tsv")
        heldout_rows = read_rows(root / "authority/heldout24/manifest.tsv")

        assert len(full_rows) == 27
        assert len(dev_rows) == 3
        assert len(heldout_rows) == 24
        assert [row["case_id"] for row in dev_rows] == list(DEV_CASE_IDS)
        assert not (
            {row["case_id"] for row in dev_rows}
            & {row["case_id"] for row in heldout_rows}
        )
        assert [int(row["authority_row_index"]) for row in full_rows] == list(
            range(1, 28)
        )
        assert {row["cem_samples"] for row in full_rows} == {"1024"}
        assert {row["cem_opt_steps"] for row in full_rows} == {"32"}
        assert {row["cem_seed"] for row in full_rows} == {"0"}
        assert {row["selection_role"] for row in dev_rows} == {"selection_eligible"}
        assert {row["selection_role"] for row in heldout_rows} == {
            "evaluation_only_after_freeze"
        }
        assert manifest["source_manifest_sha256"] == EXPECTED_SOURCE_SHA256
        assert manifest["object_counts"] == {
            "bucket003": 9,
            "bucket004": 4,
            "bucket007": 14,
        }
        assert manifest["heldout_policy"]["pre_freeze"] == "selection_forbidden"
        assert manifest["heldout_policy"]["post_freeze"] == "evaluation_only"


def test_gpu_parser_and_contract() -> None:
    """GPU validation must allow overlap but enforce the selected devices."""
    rows = parse_gpu_csv(
        "0, NVIDIA RTX 6000 Ada Generation, GPU-a, 49140, 9000, 40140, 50\n"
        "1, NVIDIA RTX 6000 Ada Generation, GPU-b, 49140, 8000, 41140, 40\n"
    )
    result = validate_gpu_contract(
        rows,
        expected_ids=(0, 1),
        required_name="RTX 6000 Ada",
        allow_existing_compute_overlap=True,
        compute_processes=("GPU-a, 123, python, 7000",),
    )
    assert result["status"] == "PASS"
    assert result["existing_compute_overlap"] is True
    assert result["kill_existing_processes"] is False


def test_dependency_and_compile_contract() -> None:
    """Pinned geometry packages and MuJoCo/MJWarp probes must work."""
    versions = dependency_versions()
    assert versions["coacd"] == "1.0.11"
    assert versions["trimesh"] == "4.11.5"
    for geom_type in ("mesh", "sdf"):
        result = compile_probe(geom_type)
        assert result["status"] == "PASS"
        assert result["cpu_ngeom"] == 1
        assert result["warp_ngeom"] == 1


def test_source_snapshot_contract() -> None:
    """Deployment source identity must include code/lock and use an isolated root."""
    with tempfile.TemporaryDirectory(prefix="e182_source_test_") as directory:
        root = Path(directory)
        manifest = build_source_manifest(
            output_path=root / "source_snapshot_manifest.json",
            file_list_path=root / "source_snapshot_files.txt",
        )
        paths = {entry["path"] for entry in manifest["files"]}
        assert "uv.lock" in paths
        assert "workspace/core4d/scripts/experiments/E182/test_preflight.py" in paths
        assert manifest["source_file_count"] == len(paths)
        assert len(manifest["source_snapshot_sha256"]) == 64
        remote_root = derive_remote_root(manifest["source_snapshot_sha256"])
        assert remote_root.startswith("/home/xiayb/pHRI_workspace/e182_runs/")
        assert remote_root.endswith("/spider")


def test_gate_audit_contract() -> None:
    """The S0 audit must join authority, deployment, and environment evidence."""
    authority = {
        "status": "PASS",
        "case_count": 27,
        "dev_heldout_overlap": [],
        "cem_contract": {"samples": 1024, "opt_steps": 32, "seed": 0},
        "snapshot": {"candidate_manifests": 54},
    }
    deployment = {
        "status": "PASS",
        "remote_root": "/home/xiayb/pHRI_workspace/e182_runs/e182_0123456789abcdef/spider",
        "source_manifest_sha256": "a" * 64,
        "source_verification": {"status": "PASS"},
        "dependencies": {"status": "PASS"},
        "existing_processes_modified": False,
    }
    environment = {
        "status": "PASS",
        "remote_deployment": {
            "status": "PASS",
            "remote_root": deployment["remote_root"],
            "expected": {"source_manifest_sha": "a" * 64},
        },
        "execution_contract": {
            "allow_existing_compute_overlap": True,
            "kill_existing_processes": False,
            "pause_existing_processes": False,
            "preempt_existing_processes": False,
        },
    }
    assert evaluate_gate(authority, deployment, environment)["status"] == "PASS"
    environment["remote_deployment"]["remote_root"] = "/wrong/root"
    assert evaluate_gate(authority, deployment, environment)["status"] == "FAIL"


def main() -> int:
    """Run the S0 tests without relying on pytest discovery."""
    tests = (
        test_authority_projection,
        test_gpu_parser_and_contract,
        test_dependency_and_compile_contract,
        test_source_snapshot_contract,
        test_gate_audit_contract,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"E182_PREFLIGHT_TESTS=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
