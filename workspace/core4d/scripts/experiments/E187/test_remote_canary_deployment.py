#!/usr/bin/env python3
"""Direct-main contracts for E187 immutable Ada canary deployment."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).with_name("deploy_remote_canary.py")
SPEC = importlib.util.spec_from_file_location("e187_remote_deploy", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_runtime_closure_is_exact_and_frozen() -> None:
    """The deployment contains only the two fixed Ada representatives."""
    files, records = MODULE.runtime_files()
    assert tuple(record["case_id"] for record in records) == MODULE.REMOTE_CASES
    assert {record["worker"] for record in records} == {"remote-0", "remote-1"}
    assert any(value.endswith("bucket004/grid.npy") for value in files)
    assert any(value.endswith("bucket007_2p5mm/grid.npy") for value in files)
    assert not any("s4_canary/rows" in value for value in files)
    source = MODULE.source_files()
    for selector in (
        "examples/config/override/"
        "core4d_dcv3_omnirt_v1_ref_fk_bucket004_20231002_021_p1.yaml",
        "examples/config/override/"
        "core4d_dcv3_omnirt_v2_ref_fk_bucket007_20231020_055_p1.yaml",
    ):
        assert selector in source
    assert MODULE.sha256(MODULE.A2_LOCK) == (
        "400c98b422ac4458eccffb589eecc1d776fe5cd3968eb855cd89a485bb7d5441"
    )


def test_remote_root_and_verifier_fail_closed() -> None:
    """Unsafe roots and tampered snapshot files are rejected."""
    sha = "a" * 64
    root = MODULE.derive_remote_root(sha)
    MODULE.validate_remote_root(root)
    try:
        MODULE.validate_remote_root("/home/xiayb/pHRI_workspace/spider")
    except RuntimeError:
        pass
    else:
        raise AssertionError("shared checkout was accepted as a run root")
    with tempfile.TemporaryDirectory() as temporary:
        sandbox = Path(temporary)
        artifact = sandbox / "payload.bin"
        artifact.write_bytes(b"frozen")
        manifest = sandbox / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "source_snapshot_sha256": sha,
                    "files": [
                        {
                            "path": "payload.bin",
                            "sha256": MODULE.sha256(artifact),
                        }
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
        test_runtime_closure_is_exact_and_frozen,
        test_remote_root_and_verifier_fail_closed,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"E187_REMOTE_CANARY_DEPLOYMENT_TESTS=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
