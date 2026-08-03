#!/usr/bin/env python3
"""Static contracts for E187's missing bucket004 formal tape."""

from __future__ import annotations

import sys

from run_bucket004_formal_tape import (
    CASE_ID,
    DEFAULT_OUTPUT_ROOT,
    RECORD_START_SIM_STEP,
    build_command,
    load_row,
)


def test_bucket004_authority() -> None:
    """The representative row and production-fixed contact step are frozen."""
    row = load_row()
    assert CASE_ID == "bucket004_20231002_021_p1"
    assert row["object_key"] == "bucket004"
    assert RECORD_START_SIM_STEP == 12


def test_formal_command_contract() -> None:
    """Capture uses full budget, seed0, one transform chunk, and the old reward."""
    row = load_row()
    command = build_command(
        row,
        python_bin=sys.executable,
        gpu_id=0,
        output_dir=DEFAULT_OUTPUT_ROOT / "runs" / f"{CASE_ID}_outdir",
        tape_root=DEFAULT_OUTPUT_ROOT / "raw_chunks",
    )
    joined = "\n".join(command)
    for value in (
        "num_samples=1024",
        "max_num_iterations=32",
        "seed=0",
        "device=cuda:0",
        "+query_tape_max_chunks=1",
        "+query_tape_stop_after_chunks=1",
        "+query_tape_record_start_sim_step=12",
        "+query_tape_record_geometry_state=true",
    ):
        assert value in joined
    assert "surface_band_score_mode" not in joined


def main() -> int:
    """Run direct-main contracts."""
    tests = (test_bucket004_authority, test_formal_command_contract)
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"E187_BUCKET004_FORMAL_TAPE_TESTS=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
