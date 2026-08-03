#!/usr/bin/env python3
"""Static contracts for the E187 E178-compatible replay runner."""

from __future__ import annotations

import tempfile
from pathlib import Path

from build_e178_compat_replay import REPRESENTATIVE_CASES, build
from run_e178_compat_replay import _read, build_command, validate_inputs


def test_manifest_authority_and_isolation() -> None:
    """All three rows keep scientific inputs and isolate every output path."""
    with tempfile.TemporaryDirectory(prefix="e187_compat_manifest_") as directory:
        root = Path(directory)
        result = build(root)
        assert result["rows"] == 3
        fields, rows = _read(root / "execution_manifest.tsv")
        assert fields
        assert tuple(row["case_id"] for row in rows) == REPRESENTATIVE_CASES
        for row in rows:
            validate_inputs(row)
            assert str(root.resolve()) in row["result_npz"]
            assert "results/E178" in row["source_e178_result_npz"]
            assert row["result_npz"] != row["source_e178_result_npz"]
            assert row["status"] == "not_run"


def test_explicit_frozen_command() -> None:
    """Every replay command freezes budget, seed, legacy override, and recorder-off."""
    with tempfile.TemporaryDirectory(prefix="e187_compat_command_") as directory:
        root = Path(directory)
        build(root)
        _, rows = _read(root / "execution_manifest.tsv")
        for row in rows:
            command = build_command(row, python_bin="python")
            required = {
                f"+override={row['override_id']}",
                f"task={row['target_task']}",
                "save_video=false",
                "num_samples=1024",
                "max_num_iterations=32",
                "seed=0",
            }
            assert required <= set(command)
            assert not any("distance_continuation" in token for token in command)
            assert "query_tape_enabled" not in " ".join(command)


def main() -> int:
    """Run direct-main tests."""
    tests = (test_manifest_authority_and_isolation, test_explicit_frozen_command)
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"E187_COMPAT_REPLAY_TESTS=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
