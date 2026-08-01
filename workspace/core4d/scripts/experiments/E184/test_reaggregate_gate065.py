#!/usr/bin/env python3
"""Boundary and source-integrity contracts for E184 gate re-aggregation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import reaggregate_gate065 as gate


def test_positive_boundaries() -> None:
    """TP=18 admits 9 phantom, but not 10; recall has the analogous boundary."""
    assert gate.phantom_limit(18, 0.65) == 9
    assert gate.phantom_limit(19, 0.65) == 10
    assert gate.phantom_limit(22, 0.65) == 11
    assert gate.gate_from_confusion(18, 9, 9, 0.65, zero_aware=True)
    assert not gate.gate_from_confusion(18, 10, 9, 0.65, zero_aware=True)
    assert not gate.gate_from_confusion(18, 9, 10, 0.65, zero_aware=True)


def test_zero_oracle_contract() -> None:
    """A no-contact oracle passes only when the candidate also predicts no contact."""
    assert gate.gate_from_confusion(0, 0, 0, 0.65, zero_aware=True)
    assert not gate.gate_from_confusion(0, 1, 0, 0.65, zero_aware=True)
    assert not gate.gate_from_confusion(0, 0, 0, 0.65, zero_aware=False)


def test_frozen_sources_and_tamper_rejection() -> None:
    """The exact 540-row E183 input loads, while a byte change is rejected."""
    cases, summaries = gate.load_sources()
    assert len(cases) == 540
    assert len(summaries) == 60
    with tempfile.TemporaryDirectory(prefix="e184_tamper_") as directory:
        copied = Path(directory) / "case.tsv"
        copied.write_bytes(gate.SOURCE_CASE_PATH.read_bytes() + b"\n")
        try:
            gate.load_sources(
                copied,
                gate.SOURCE_SUMMARY_PATH,
                expected_case_sha=gate.EXPECTED_CASE_SHA256,
            )
        except RuntimeError as error:
            assert "case table SHA changed" in str(error)
        else:
            raise AssertionError("tampered E183 case table was accepted")


def test_full_reaggregation_in_temporary_root() -> None:
    """The offline pipeline preserves E183 legacy counts and validates end to end."""
    with tempfile.TemporaryDirectory(prefix="e184_test_") as directory:
        root = Path(directory)
        gate.freeze_protocol(root)
        aggregate = gate.aggregate(root)
        gate.render_comparison(root)
        validation = gate.validate(root)
        assert validation["status"] == "PASS"
        assert aggregate["case_candidate_row_count"] == 540
        assert (
            aggregate["overall"]["pooled_pass_070"],
            aggregate["overall"]["macro_legacy_pass_070"],
            aggregate["overall"]["legacy_all_case_pass_070"],
        ) == (23, 19, 3)


def main() -> int:
    """Run the standalone E184 contracts without pytest collection."""
    tests = (
        test_positive_boundaries,
        test_zero_oracle_contract,
        test_frozen_sources_and_tamper_rejection,
        test_full_reaggregation_in_temporary_root,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"E184_TESTS=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
