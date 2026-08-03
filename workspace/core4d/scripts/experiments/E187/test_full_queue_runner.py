#!/usr/bin/env python3
"""Direct-main contracts for the standalone E187 Full queue runner."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import run_full_queue as runner


def test_frozen_queues_build_only_full_recorder_off_commands() -> None:
    """All 19 new rows retain the frozen Full scientific command."""
    cases: set[str] = set()
    total = 0
    with tempfile.TemporaryDirectory(prefix="e187_full_commands_") as temporary:
        root = Path(temporary)
        for worker in runner.WORKERS:
            rows = runner.load_worker_queue(worker)
            assert rows[0]["initial_status"].startswith("PROMOTED_CANARY")
            for row in rows[1:]:
                command = runner.build_command(
                    row,
                    python_bin=sys.executable,
                    gpu_id=0,
                    output_dir=root / row["case_id"] / "outdir",
                    video_path=root / row["case_id"] / "video.mp4",
                )
                assert {
                    "num_samples=1024",
                    "max_num_iterations=32",
                    "seed=0",
                    "save_video=true",
                    "save_info=true",
                    "+query_tape_enabled=false",
                    "+query_tape_record_geometry_state=false",
                } <= set(command)
                assert not any(value.startswith("max_sim_steps=") for value in command)
                cases.add(row["case_id"])
                total += 1
    assert total == len(cases) == 19


def test_promotions_are_exact_and_source_artifacts_are_frozen() -> None:
    """Exactly three queue heads map to their immutable A3 canaries."""
    a3, _ = runner.frozen_authority()
    canaries = {row["case_id"]: row for row in a3["canaries"]}
    promoted = []
    for worker in runner.WORKERS:
        row = runner.load_worker_queue(worker)[0]
        payload = runner.promotion_payload(row, canaries[row["case_id"]])
        assert payload["execution_kind"] == "PROMOTED_CANARY"
        assert payload["worker"] == worker
        assert payload["queue_position"] == 1
        promoted.append(payload["case_id"])
    assert set(promoted) == set(canaries)


def main() -> int:
    """Run all direct-main contracts."""
    tests = (
        test_frozen_queues_build_only_full_recorder_off_commands,
        test_promotions_are_exact_and_source_artifacts_are_frozen,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"E187_FULL_QUEUE_RUNNER_TESTS=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
