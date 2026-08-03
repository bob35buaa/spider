#!/usr/bin/env python3
"""Direct-main contracts for E187 A3 same-device efficiency evidence."""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).with_name("audit_canary_efficiency.py")
SPEC = importlib.util.spec_from_file_location("e187_efficiency", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_commands_keep_full_density_and_isolate_outputs() -> None:
    """Every baseline keeps 1024x32 seed0 and uses a separate short-probe root."""
    for case_id, assignment in MODULE.ASSIGNMENTS.items():
        command = MODULE.build_command(
            case_id,
            python_bin="python",
            gpu_id=int(assignment["gpu_id"]),
            repo_root=Path("/immutable/spider"),
        )
        assert "num_samples=1024" in command
        assert "max_num_iterations=32" in command
        assert "seed=0" in command
        assert "max_sim_steps=30" in command
        assert "+query_tape_enabled=false" in command
        assert any("efficiency_baseline/rows" in value for value in command)
        assert not any("s4_canary/rows/" in value for value in command)


def test_parser_uses_only_optimized_records() -> None:
    """Warm-up records cannot dilute the same-device timing ratio."""
    with tempfile.TemporaryDirectory() as temporary:
        log = Path(temporary) / "run.log"
        log.write_text(
            "plan time: 0.01s, sim_steps: 2/30, opt_steps: 0\n"
            "plan time: 12.5s, sim_steps: 14/30, opt_steps: 32\n"
            "plan time: 12.7s, sim_steps: 16/30, opt_steps: 32\n",
            encoding="utf-8",
        )
        assert MODULE.optimized_times(log) == [12.5, 12.7]


def main() -> int:
    """Run direct-main tests."""
    tests = (
        test_commands_keep_full_density_and_isolate_outputs,
        test_parser_uses_only_optimized_records,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"E187_CANARY_EFFICIENCY_TESTS=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
