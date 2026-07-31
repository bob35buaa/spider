#!/usr/bin/env python3
"""Direct-main tests for E182 query-tape replay and exact comparison."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from audit_query_tape_replay import audit_same_run_integrity, compare_npz_arrays
from run_query_tape_replay import build_command

from spider.query_tape import finalize_cem_query_tape, record_cem_query_chunk


def test_replay_command_contract() -> None:
    """Replay command must preserve E178 config while freezing the canary budget."""
    row = {
        "case_id": "bucket003_case",
        "override_id": "core4d_E178_bucket003_case_contactAlignedTop",
        "target_task": "dcv3_bucket003_case",
    }
    command = build_command(
        row=row,
        mode="on_a",
        python_bin="python",
        output_dir=Path("/tmp/e182/out"),
        tape_root=Path("/tmp/e182/tape"),
        gpu_id=0,
    )
    joined = " ".join(command)
    assert "+override=core4d_E178_bucket003_case_contactAlignedTop" in joined
    assert "task=dcv3_bucket003_case" in joined
    assert "num_samples=64" in joined
    assert "max_num_iterations=4" in joined
    assert "seed=0" in joined
    assert "+query_tape_enabled=true" in joined
    assert "+query_tape_run_id=on_a/bucket003_case" in joined


def test_exact_npz_comparison() -> None:
    """Comparison must prove array exactness and identify changed keys."""
    with tempfile.TemporaryDirectory(prefix="e182_compare_test_") as directory:
        root = Path(directory)
        left = root / "left.npz"
        right = root / "right.npz"
        np.savez(left, qpos=np.arange(6), rew=np.array([1.0], dtype=np.float32))
        np.savez(right, qpos=np.arange(6), rew=np.array([1.0], dtype=np.float32))
        equal = compare_npz_arrays(left, right)
        assert equal["status"] == "PASS"
        assert equal["mismatched_keys"] == []
        np.savez(right, qpos=np.arange(6) + 1, rew=np.array([1.0], dtype=np.float32))
        changed = compare_npz_arrays(left, right)
        assert changed["status"] == "FAIL"
        assert changed["mismatched_keys"] == ["qpos"]


def test_same_run_integrity() -> None:
    """A frozen chunk must exactly match its own optimizer summaries."""
    with tempfile.TemporaryDirectory(prefix="e182_integrity_test_") as directory:
        root = Path(directory)
        config = SimpleNamespace(
            query_tape_enabled=True,
            query_tape_output_dir=str(root / "raw"),
            query_tape_run_id="on_a/unit_case",
        )
        rewards = torch.tensor([1.0, 3.0], dtype=torch.float32)
        valid = torch.tensor([False, True])
        record_cem_query_chunk(
            config,
            {
                "qpos": torch.zeros((2, 3, 4), dtype=torch.float32),
                "rewards": rewards,
                "selected_indices": torch.tensor([1], dtype=torch.int64),
                "sample_gate_valid_mask": valid,
            },
        )
        finalize_cem_query_tape(config, provenance={"case_id": "unit_case"})
        result = root / "trajectory.npz"
        np.savez(
            result,
            opt_steps=np.array([[0], [1]], dtype=np.int64),
            rew_max=np.array([[0.0], [3.0]], dtype=np.float32),
            rew_min=np.array([[0.0], [1.0]], dtype=np.float32),
            rew_median=np.array([[0.0], [2.0]], dtype=np.float32),
            rew_mean=np.array([[0.0], [2.0]], dtype=np.float32),
            cem_selected_index0=np.array([[0], [1]], dtype=np.int64),
            sample_gate_valid_mask_max=np.array([[False], [True]]),
            sample_gate_valid_mask_min=np.array([[False], [False]]),
            sample_gate_valid_mask_median=np.array([[0.0], [0.5]]),
            sample_gate_valid_mask_mean=np.array([[0.0], [0.5]]),
        )
        manifest = root / "raw/on_a/unit_case/chunk_manifest.json"
        assert audit_same_run_integrity(result, manifest)["status"] == "PASS"
        with np.load(result) as values:
            changed = {key: values[key] for key in values.files}
        changed["cem_selected_index0"] = np.array([[0], [0]], dtype=np.int64)
        np.savez(result, **changed)
        failed = audit_same_run_integrity(result, manifest)
        assert failed["status"] == "FAIL"
        assert failed["mismatch_count"] == 1


def test_relocated_same_run_integrity() -> None:
    """Pulled remote chunks must audit after their absolute source path disappears."""
    with tempfile.TemporaryDirectory(prefix="e182_relocated_test_") as directory:
        root = Path(directory)
        source = root / "remote_like"
        config = SimpleNamespace(
            query_tape_enabled=True,
            query_tape_output_dir=str(source / "raw"),
            query_tape_run_id="on_a/unit_case",
        )
        record_cem_query_chunk(
            config,
            {
                "qpos": torch.zeros((2, 3, 4), dtype=torch.float32),
                "rewards": torch.tensor([1.0, 3.0], dtype=torch.float32),
                "selected_indices": torch.tensor([1], dtype=torch.int64),
            },
        )
        finalize_cem_query_tape(config, provenance={"case_id": "unit_case"})
        pulled = root / "pulled"
        shutil.copytree(source, pulled)
        shutil.rmtree(source)
        result = root / "trajectory.npz"
        np.savez(
            result,
            opt_steps=np.array([[0], [1]], dtype=np.int64),
            rew_max=np.array([[0.0], [3.0]], dtype=np.float32),
            rew_min=np.array([[0.0], [1.0]], dtype=np.float32),
            rew_median=np.array([[0.0], [2.0]], dtype=np.float32),
            rew_mean=np.array([[0.0], [2.0]], dtype=np.float32),
            cem_selected_index0=np.array([[0], [1]], dtype=np.int64),
        )
        manifest = pulled / "raw/on_a/unit_case/chunk_manifest.json"
        assert audit_same_run_integrity(result, manifest)["status"] == "PASS"


def main() -> int:
    """Run pipeline tests without pytest discovery."""
    tests = (
        test_replay_command_contract,
        test_exact_npz_comparison,
        test_same_run_integrity,
        test_relocated_same_run_integrity,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"E182_QUERY_PIPELINE_TESTS=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
