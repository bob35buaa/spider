#!/usr/bin/env python3
"""Direct-main contracts for the E187 A3 canary gate lock."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).with_name("freeze_canary_gate.py")
SPEC = importlib.util.spec_from_file_location("e187_canary_lock", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_authority_stays_waived_and_full_zero() -> None:
    """A3 cannot rewrite technical Gate0 or claim an early Full row."""
    frozen = MODULE.authority()
    assert frozen["a1"]["gate0_technical_status"] == "FAIL"
    assert frozen["a1"]["progression_authority"] == "USER_WAIVED"
    assert frozen["a2"]["full_cem_started_rows"] == 0
    assert frozen["a2"]["next_gate"] == "A3_S4_THREE_DEVICE_CANARY"


def test_atomic_lock_refuses_replacement() -> None:
    """A different completed A3 payload cannot overwrite a frozen lock."""
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "lock.json"
        MODULE.atomic_json(path, {"status": "FROZEN", "value": 1})
        MODULE.atomic_json(path, {"status": "FROZEN", "value": 1})
        assert json.loads(path.read_text()) == {"status": "FROZEN", "value": 1}
        try:
            MODULE.atomic_json(path, {"status": "FROZEN", "value": 2})
        except RuntimeError:
            pass
        else:
            raise AssertionError("a different A3 lock replaced frozen evidence")


def main() -> int:
    """Run direct-main tests."""
    tests = (
        test_authority_stays_waived_and_full_zero,
        test_atomic_lock_refuses_replacement,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"E187_CANARY_GATE_LOCK_TESTS=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
