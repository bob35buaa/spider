#!/usr/bin/env python3
"""Direct-main tests for the E187 A2 production-integration lock."""

from __future__ import annotations

from freeze_production_integration import build_payload


def test_all_a2_gates_are_frozen() -> None:
    """A2 lock must close overrides, compile, recorder, and 44 G tapes."""
    payload = build_payload()
    assert payload["status"] == "FROZEN"
    assert payload["production_overrides"]["row_count"] == 22
    assert payload["production_integration"]["row_count"] == 22
    assert payload["production_integration"]["recorder_off"]["status"] == "PASS"
    assert payload["reference_final_gate"]["tape_count"] == 44
    assert payload["reference_final_gate"]["finite_tape_count"] == 44
    assert payload["reference_final_gate"]["false_safe_accept_total"] == 0
    assert payload["full_cem_started_rows"] == 0
    assert payload["next_gate"] == "A3_S4_THREE_DEVICE_CANARY"


def main() -> int:
    """Run direct-main tests."""
    test_all_a2_gates_are_frozen()
    print("PASS test_all_a2_gates_are_frozen")
    print("E187_PRODUCTION_INTEGRATION_LOCK_TESTS=PASS count=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
