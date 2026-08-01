#!/usr/bin/env python3
"""Boundary, source, and regression contracts for E185 gate060."""

from __future__ import annotations

import tempfile
from pathlib import Path

import reaggregate_gate060 as gate


def test_gate060_boundaries() -> None:
    """The 0.60 gate admits the exact 2TP/3 integer phantom boundary."""
    assert gate.phantom_limit(18, 0.60) == 12
    assert gate.phantom_limit(19, 0.60) == 12
    assert gate.phantom_limit(22, 0.60) == 14
    assert gate.gate_from_confusion(18, 12, 12, 0.60, zero_aware=True)
    assert not gate.gate_from_confusion(18, 13, 12, 0.60, zero_aware=True)
    assert not gate.gate_from_confusion(18, 12, 13, 0.60, zero_aware=True)


def test_zero_oracle_contract() -> None:
    """No-contact oracle passes only for no candidate contact."""
    assert gate.gate_from_confusion(0, 0, 0, 0.60, zero_aware=True)
    assert not gate.gate_from_confusion(0, 1, 0, 0.60, zero_aware=True)
    assert not gate.gate_from_confusion(0, 0, 0, 0.60, zero_aware=False)


def test_sources_and_tamper_rejection() -> None:
    """The exact E183 tables load and a byte change is rejected."""
    cases, summaries = gate.load_sources()
    assert len(cases) == 540
    assert len(summaries) == 60
    with tempfile.TemporaryDirectory(prefix="e185_tamper_") as directory:
        copied = Path(directory) / "case.tsv"
        copied.write_bytes(gate.SOURCE_CASE_PATH.read_bytes() + b"\n")
        try:
            gate.load_sources(copied, gate.SOURCE_SUMMARY_PATH)
        except RuntimeError as error:
            assert "case table SHA changed" in str(error)
        else:
            raise AssertionError("tampered E183 case table was accepted")


def test_end_to_end_and_e184_regression() -> None:
    """A temporary run reproduces E184 thresholds and validates all outputs."""
    with tempfile.TemporaryDirectory(prefix="e185_test_") as directory:
        root = Path(directory)
        gate.freeze_protocol(root)
        aggregate = gate.aggregate(root)
        gate.render_comparison(root)
        validation = gate.validate(root)
        assert validation["status"] == "PASS"
        assert aggregate["case_candidate_row_count"] == 540
        for key, expected in gate.EXPECTED_E184_OVERALL.items():
            assert aggregate["overall"][key] == expected


def main() -> int:
    """Run the standalone E185 contracts without pytest collection."""
    tests = (
        test_gate060_boundaries,
        test_zero_oracle_contract,
        test_sources_and_tamper_rejection,
        test_end_to_end_and_e184_regression,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"E185_TESTS=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
