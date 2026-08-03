#!/usr/bin/env python3
"""Static contracts for the bounded E186 formal-budget probe."""

from __future__ import annotations

import tempfile
from pathlib import Path

from run_fullbudget_feasibility_probe import (
    PROBE_CASES,
    RECORD_START_SIM_STEP,
    build_command,
)
from run_shadow64x4 import rows_by_case


def test_probe_command() -> None:
    """The probe uses Full budget but remains one bounded recorded query."""
    rows = rows_by_case()
    assert PROBE_CASES == (
        "bucket003_20231018_003_p1",
        "bucket007_20231020_055_p1",
    )
    with tempfile.TemporaryDirectory(prefix="e186_fullbudget_probe_") as directory:
        root = Path(directory)
        for case_id in PROBE_CASES:
            command = build_command(
                rows[case_id],
                python_bin="python",
                gpu_id=0,
                output_dir=root / case_id / "out",
                tape_root=root / case_id / "tape",
            )
            required = {
                "num_samples=1024",
                "max_num_iterations=32",
                "seed=0",
                "+query_tape_max_chunks=1",
                "+query_tape_stop_after_chunks=1",
                "+query_tape_record_geometry_state=false",
                f"+query_tape_record_start_sim_step={RECORD_START_SIM_STEP[case_id]}",
            }
            assert required <= set(command)
            assert not any(value.startswith("max_sim_steps=") for value in command)


def main() -> int:
    """Run direct-main contracts."""
    test_probe_command()
    print("PASS test_probe_command")
    print("E186_FULLBUDGET_PROBE_TESTS=PASS count=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
