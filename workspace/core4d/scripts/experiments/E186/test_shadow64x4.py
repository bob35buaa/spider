#!/usr/bin/env python3
"""Static contracts for the bounded E186 shadow runner."""

from __future__ import annotations

import tempfile
from pathlib import Path

from run_shadow64x4 import (
    RECORD_START_SIM_STEP,
    REPRESENTATIVE_CASES,
    build_command,
    rows_by_case,
)


def test_representative_authority() -> None:
    """The frozen representatives cover the three object-specific colliders."""
    rows = rows_by_case()
    assert tuple(rows) == REPRESENTATIVE_CASES
    assert [rows[case]["object_key"] for case in REPRESENTATIVE_CASES] == [
        "bucket003",
        "bucket004",
        "bucket007",
    ]
    assert all(
        rows[case]["physics_status"] == "COMPOUND_CPU_MJWARP_PASS" for case in rows
    )


def test_command_is_one_bounded_query() -> None:
    """The worker command freezes budget, seed, tape cap, and recorder mode."""
    row = rows_by_case()[REPRESENTATIVE_CASES[0]]
    with tempfile.TemporaryDirectory(prefix="e186_shadow_command_") as directory:
        root = Path(directory)
        command = build_command(
            row,
            python_bin="python",
            gpu_id=0,
            output_dir=root / "out",
            tape_root=root / "tape",
        )
    required = {
        "num_samples=64",
        "max_num_iterations=4",
        "seed=0",
        "+query_tape_enabled=true",
        "+query_tape_max_chunks=1",
        "+query_tape_stop_after_chunks=1",
        "+query_tape_record_geometry_state=true",
        "+use_torch_compile=false",
        "save_video=false",
        "device=cuda:0",
    }
    assert required <= set(command)
    assert (
        f"+query_tape_record_start_sim_step={RECORD_START_SIM_STEP[row['case_id']]}"
        in command
    )
    assert not any(value.startswith("max_sim_steps=") for value in command)


def main() -> int:
    """Run direct-main contracts."""
    tests = (test_representative_authority, test_command_is_one_bounded_query)
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"E186_SHADOW64X4_TESTS=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
