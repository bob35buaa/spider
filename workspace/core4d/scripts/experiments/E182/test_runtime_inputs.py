#!/usr/bin/env python3
"""Direct-main tests for E182 immutable runtime inputs and replay isolation."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from run_query_tape_replay import case_manifest_path
from runtime_inputs import (
    assert_compatible_frozen_manifest,
    build_runtime_input_manifest,
    verify_runtime_inputs,
)


def test_real_dev3_runtime_closure() -> None:
    """The frozen closure must cover all ignored inputs needed by dev3 replay."""
    with tempfile.TemporaryDirectory(prefix="e182_runtime_manifest_") as directory:
        root = Path(directory)
        manifest = build_runtime_input_manifest(
            output_path=root / "runtime_inputs_manifest.json",
            file_list_path=root / "runtime_input_files.txt",
        )
        paths = {entry["path"] for entry in manifest["files"]}
        assert manifest["status"] == "FROZEN"
        assert manifest["case_count"] == 3
        assert manifest["case_ids"] == [
            "bucket003_20231018_001_p1",
            "bucket004_20231002_021_p1",
            "bucket007_20231020_055_p1",
        ]
        assert manifest["runtime_file_count"] == len(paths)
        assert len(manifest["runtime_snapshot_sha256"]) == 64
        assert "workspace/core4d/results/E182/authority/dev3/manifest.tsv" in paths
        assert (
            "workspace/core4d/results/E178/s6_downstream/manifests/"
            "semantic_bucket_full_manifest.tsv" in paths
        )
        for object_key in ("bucket003", "bucket004", "bucket007"):
            assert any(
                path.startswith(
                    f"example_datasets/processed/core4d/assets/objects/{object_key}/"
                )
                for path in paths
            )
            assert any(
                object_key in path and path.endswith("config_act.yaml")
                for path in paths
            )
            assert any(
                object_key in path and path.endswith("_full.npz") for path in paths
            )
            assert any(
                object_key in path and path.endswith("raw_contact_mask_3cm.npz")
                for path in paths
            )


def test_verifier_and_frozen_manifest_rejection() -> None:
    """Verification catches mutation and frozen identity rejects replacement."""
    with tempfile.TemporaryDirectory(prefix="e182_runtime_verify_") as directory:
        root = Path(directory)
        source = root / "source"
        target = root / "target"
        (source / "inputs").mkdir(parents=True)
        (source / "inputs/a.bin").write_bytes(b"alpha")
        (source / "inputs/b.bin").write_bytes(b"beta")
        manifest_path = root / "manifest.json"
        file_list = root / "files.txt"
        manifest = build_runtime_input_manifest(
            output_path=manifest_path,
            file_list_path=file_list,
            repo_root=source,
            explicit_paths=("inputs/a.bin", "inputs/b.bin"),
            case_ids=("unit_case",),
        )
        shutil.copytree(source, target)
        assert verify_runtime_inputs(target, manifest_path)["status"] == "PASS"
        (target / "inputs/a.bin").write_bytes(b"changed")
        failed = verify_runtime_inputs(target, manifest_path)
        assert failed["status"] == "FAIL"
        assert failed["mismatches"][0]["error"] == "sha256_mismatch"

        assert_compatible_frozen_manifest(manifest, dict(manifest))
        changed = json.loads(json.dumps(manifest))
        changed["runtime_snapshot_sha256"] = "0" * 64
        try:
            assert_compatible_frozen_manifest(manifest, changed)
        except RuntimeError as error:
            assert "runtime snapshot mismatch" in str(error)
        else:
            raise AssertionError("incompatible frozen runtime manifest was accepted")


def test_case_scoped_replay_manifests() -> None:
    """Concurrent single-case workers must never share a run-manifest path."""
    root = Path("/tmp/e182_case_manifest_contract")
    left = case_manifest_path(root, "off", "bucket003_case")
    right = case_manifest_path(root, "off", "bucket004_case")
    assert left != right
    assert left.as_posix().endswith("off/manifests/bucket003_case.json")
    assert right.as_posix().endswith("off/manifests/bucket004_case.json")


def test_query_launcher_contracts() -> None:
    """S1 launch/pull scripts must use local0+Ada0/1 without process interference."""
    repo_root = Path(__file__).resolve().parents[5]
    active = repo_root / "workspace/core4d/scripts/launch/active"
    local = active / "run_E182_local.sh"
    remote = active / "run_E182_remote_a6000.sh"
    worker = active / "run_E182_query_tape_worker.sh"
    pull = active / "pull_E182_remote_a6000_results.sh"
    for script in (local, remote, worker, pull):
        assert script.is_file(), script
        subprocess.run(("bash", "-n", str(script)), check=True)
        for line in script.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            assert not stripped.startswith(("kill ", "pkill ", "killall "))
            assert "tmux kill-session" not in stripped
    local_text = local.read_text(encoding="utf-8")
    remote_text = remote.read_text(encoding="utf-8")
    worker_text = worker.read_text(encoding="utf-8")
    pull_text = pull.read_text(encoding="utf-8")
    assert "bucket007_20231020_055_p1" in local_text
    assert "query-tape" in local_text
    assert "bucket003_20231018_001_p1" in remote_text
    assert "bucket004_20231002_021_p1" in remote_text
    assert 'EXPECTED_GPUS="${ADA_EXPECTED_GPUS:-0 1}"' in remote_text
    assert "runtime_inputs.py" in remote_text
    assert "tmux new-session" in remote_text
    assert 'CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU_ID"' in worker_text
    assert '"--gpu-id" "0"' in worker_text
    assert "--ignore-existing" in pull_text
    assert "--delete" not in pull_text
    assert "audit_query_tape_replay.py" in pull_text


def main() -> int:
    """Run tests without pytest discovery."""
    tests = (
        test_real_dev3_runtime_closure,
        test_verifier_and_frozen_manifest_rejection,
        test_case_scoped_replay_manifests,
        test_query_launcher_contracts,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"E182_RUNTIME_INPUT_TESTS=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
