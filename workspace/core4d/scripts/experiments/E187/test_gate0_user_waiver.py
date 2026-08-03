#!/usr/bin/env python3
"""Contracts for E187's user-authorized Gate S0 waiver."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import freeze_gate0_user_waiver as waiver


def test_live_authority_and_honest_status() -> None:
    """The waiver must preserve technical failure and every frozen source SHA."""
    payload = waiver.waiver_payload()
    assert payload["technical_gate_status"] == "FAIL"
    assert payload["progression_authority"] == "USER_WAIVED"
    assert payload["full_contract"]["use_a100"] is False
    assert len(payload["sources"]) == len(waiver.EXPECTED_SHA256)
    assert "continuation_reward_and_grid_fidelity" in payload["not_waived"]
    assert "three_device_production_canary" in payload["not_waived"]


def test_idempotent_and_tamper_rejection() -> None:
    """A frozen waiver can resume byte-exactly and rejects mutation."""
    with tempfile.TemporaryDirectory(prefix="e187_waiver_") as directory:
        root = Path(directory)
        first = waiver.freeze(root)
        second = waiver.freeze(root)
        assert first == second
        path = root / "waiver_manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["technical_gate_status"] = "PASS"
        path.write_text(json.dumps(payload), encoding="utf-8")
        try:
            waiver.freeze(root)
        except RuntimeError as error:
            assert "immutable E187 artifact mismatch" in str(error)
        else:
            raise AssertionError("tampered E187 waiver was accepted")


def main() -> int:
    """Run contracts directly."""
    tests = (
        test_live_authority_and_honest_status,
        test_idempotent_and_tamper_rejection,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"E187_GATE0_USER_WAIVER_TESTS=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
