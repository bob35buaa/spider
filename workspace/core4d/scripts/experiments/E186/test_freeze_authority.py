#!/usr/bin/env python3
"""Standalone contracts for the E186 authority freeze."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import freeze_authority as authority


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def test_p_gate_boundaries() -> None:
    """Check positive and zero-oracle gate boundaries."""
    assert authority.p_gate(7, 3, 3)
    assert not authority.p_gate(7, 4, 3)
    assert not authority.p_gate(7, 3, 4)
    assert authority.p_gate(0, 0, 0)
    assert not authority.p_gate(0, 1, 0)


def test_formal_sources_and_colliders() -> None:
    """Check exact upstream sources and all three collider locks."""
    _, source, metrics = authority._validate_source()
    colliders = authority._load_colliders()
    evidence, keeps, drops = authority._selection_rows(source, metrics, colliders)
    assert len(evidence) == 27
    assert len(keeps) == 22
    assert len(drops) == 5
    assert {row["case_id"] for row in drops} == authority.EXPECTED_DROPPED
    assert {key: value["actual_hulls"] for key, value in colliders.items()} == {
        "bucket003": 16,
        "bucket004": 8,
        "bucket007": 8,
    }


def test_tamper_rejection() -> None:
    """Reject a changed immutable keep22 artifact."""
    with tempfile.TemporaryDirectory(prefix="e186_tamper_") as directory:
        root = Path(directory)
        authority.freeze(root)
        keep = root / "s5_handoff/keep22_authority_manifest.tsv"
        keep.write_bytes(keep.read_bytes() + b"\n")
        try:
            authority.validate(root)
        except RuntimeError as error:
            assert "immutable E186 artifact mismatch" in str(error)
        else:
            raise AssertionError("tampered keep22 authority was accepted")


def test_end_to_end_idempotent() -> None:
    """Freeze twice and validate the same temporary authority."""
    with tempfile.TemporaryDirectory(prefix="e186_authority_") as directory:
        root = Path(directory)
        first = authority.freeze(root)
        second = authority.freeze(root)
        validation = authority.validate(root)
        assert first == second
        assert validation["status"] == "PASS"
        keeps = _read(root / "s5_handoff/keep22_authority_manifest.tsv")
        drops = _read(root / "s5_handoff/p_rejected5_manifest.tsv")
        assert len(keeps) == 22 and len(drops) == 5
        assert [row["authority_row_index"] for row in keeps] == sorted(
            (row["authority_row_index"] for row in keeps), key=int
        )


def main() -> int:
    """Run contracts directly without pytest collection."""
    tests = (
        test_p_gate_boundaries,
        test_formal_sources_and_colliders,
        test_tamper_rejection,
        test_end_to_end_idempotent,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"E186_AUTHORITY_TESTS=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
