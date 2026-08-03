#!/usr/bin/env python3
"""Direct-main contracts for E187 immutable Full Ada deployment."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).with_name("deploy_remote_full.py")
SPEC = importlib.util.spec_from_file_location("e187_remote_full_deploy", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_runtime_closure_contains_keep22_and_canary_evidence() -> None:
    """The new snapshot closes all rows, grids, locks, queues, and canaries."""
    queue = json.loads(MODULE.QUEUE_MANIFEST.read_text(encoding="utf-8"))
    records = [
        row for worker in queue["worker_order"] for row in queue["queues"][worker]
    ]
    assert len(records) == len({row["case_id"] for row in records}) == 22
    assert {row["worker"] for row in records} == {"local-0", "remote-0", "remote-1"}
    assert sum(row["promoted_canary"] for row in records) == 3
    for required, expected in (
        (MODULE.A1_LOCK, MODULE.A1_LOCK_SHA256),
        (MODULE.A2_LOCK, MODULE.A2_LOCK_SHA256),
        (MODULE.A3_LOCK, MODULE.A3_LOCK_SHA256),
        (MODULE.QUEUE_MANIFEST, MODULE.QUEUE_MANIFEST_SHA256),
    ):
        assert MODULE.sha256(required) == expected
    a3 = json.loads(MODULE.A3_LOCK.read_text(encoding="utf-8"))
    for canary in a3["canaries"]:
        manifest = MODULE.REPO / canary["manifest"]["path"]
        assert MODULE.sha256(manifest) == canary["manifest"]["sha256"]
    for grid in (
        "workspace/core4d/results/E186/s1_canonical_grid_sdf_v4/bucket003/grid.npy",
        "workspace/core4d/results/E186/s1_canonical_grid_sdf_v4/bucket004/grid.npy",
        "workspace/core4d/results/E187/s2_canonical_grid_sdf/candidates/bucket007_2p5mm/grid.npy",
    ):
        assert (MODULE.REPO / grid).is_file()
    if not MODULE.PROMOTION_MANIFEST.is_file():
        return
    files, frozen_records = MODULE.runtime_files()
    assert [row["case_id"] for row in frozen_records] == [
        row["case_id"] for row in records
    ]
    for required in (
        MODULE.A1_LOCK,
        MODULE.A2_LOCK,
        MODULE.A3_LOCK,
        MODULE.QUEUE_MANIFEST,
        MODULE.PROMOTION_MANIFEST,
    ):
        assert MODULE.relative_file(required) in files
    assert any(value.endswith("bucket003/grid.npy") for value in files)
    assert any(value.endswith("bucket004/grid.npy") for value in files)
    assert any(value.endswith("bucket007_2p5mm/grid.npy") for value in files)
    assert (
        len(
            [
                value
                for value in files
                if "s4_canary/rows" in value and value.endswith("manifest.json")
            ]
        )
        == 3
    )


def test_full_remote_root_and_verifier_fail_closed() -> None:
    """Only a SHA-derived Full root is accepted and tampering is detected."""
    identity = "b" * 64
    root = MODULE.derive_remote_root(identity)
    MODULE.validate_remote_root(root)
    for unsafe in (
        "/home/xiayb/pHRI_workspace/spider",
        "/home/xiayb/pHRI_workspace/e187_runs/e187_3c3ce06a2a81c5a4/spider",
    ):
        try:
            MODULE.validate_remote_root(unsafe)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"unsafe root accepted: {unsafe}")
    with tempfile.TemporaryDirectory() as temporary:
        sandbox = Path(temporary)
        artifact = sandbox / "payload.bin"
        artifact.write_bytes(b"frozen")
        manifest = sandbox / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "source_snapshot_sha256": identity,
                    "files": [
                        {"path": "payload.bin", "sha256": MODULE.sha256(artifact)}
                    ],
                }
            ),
            encoding="utf-8",
        )
        assert MODULE.verify_snapshot(sandbox, manifest)["status"] == "PASS"
        artifact.write_bytes(b"changed")
        assert MODULE.verify_snapshot(sandbox, manifest)["status"] == "FAIL"


def main() -> int:
    """Run all direct-main contracts."""
    tests = (
        test_runtime_closure_contains_keep22_and_canary_evidence,
        test_full_remote_root_and_verifier_fail_closed,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"E187_REMOTE_FULL_DEPLOYMENT_TESTS=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
