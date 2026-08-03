#!/usr/bin/env python3
"""Static contracts for E187 production canaries."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from run_production_canary import CANARY_ASSIGNMENT, build_command


def test_canary_commands_are_full_and_recorder_off() -> None:
    """Every representative uses Full budget and can be promoted unchanged."""
    assert CANARY_ASSIGNMENT == {
        "bucket003_20231018_003_p1": "local-0",
        "bucket004_20231002_021_p1": "remote-0",
        "bucket007_20231020_055_p1": "remote-1",
    }
    with tempfile.TemporaryDirectory(prefix="e187_canary_") as directory:
        root = Path(directory)
        for case_id in CANARY_ASSIGNMENT:
            command = build_command(
                case_id,
                python_bin=sys.executable,
                gpu_id=0,
                output_dir=root / case_id / "outdir",
                video_path=root / case_id / "video.mp4",
            )
            required = {
                "num_samples=1024",
                "max_num_iterations=32",
                "seed=0",
                "save_video=true",
                "save_info=true",
                "+query_tape_enabled=false",
                "+query_tape_record_geometry_state=false",
            }
            assert required <= set(command)
            assert not any(value.startswith("max_sim_steps=") for value in command)


def main() -> int:
    """Run direct-main tests."""
    test_canary_commands_are_full_and_recorder_off()
    print("PASS test_canary_commands_are_full_and_recorder_off")
    print("E187_PRODUCTION_CANARY_TESTS=PASS count=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
