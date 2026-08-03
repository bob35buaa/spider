#!/usr/bin/env python3
"""Standalone contracts for the E187 authority freeze."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import freeze_authority as authority


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def test_formal_sources_and_projection() -> None:
    """Check all upstream SHAs and the ordered keep22 projection."""
    e178, keeps = authority._validate_authority()
    assert len(e178) == 27 and len(keeps) == 22
    drop_ids = {row["case_id"] for row in authority._read_tsv(authority.E186_DROP)[1]}
    assert [row["case_id"] for row in keeps] == [
        row["case_id"] for row in e178 if row["case_id"] not in drop_ids
    ]
    assert [row["authority_row_index"] for row in keeps] == sorted(
        (row["authority_row_index"] for row in keeps), key=int
    )


def test_inventory_coverage() -> None:
    """Inventory every row artifact and all shared E178 authority files."""
    e178, _ = authority._validate_authority()
    inventory = authority._inventory(e178)
    expected = len(e178) * len(authority.E178_ROW_ARTIFACT_FIELDS) + len(
        authority.E178_SHARED_ARTIFACTS
    )
    assert len(inventory) == expected
    assert (
        len({(row["case_id"], row["artifact_kind"], row["path"]) for row in inventory})
        == expected
    )


def test_end_to_end_idempotent_and_tamper_rejection() -> None:
    """Freeze twice, then reject a changed immutable inventory."""
    with tempfile.TemporaryDirectory(prefix="e187_authority_") as directory:
        root = Path(directory)
        first = authority.freeze(root)
        second = authority.freeze(root)
        assert first == second
        keeps = _read(root / "s0_environment/keep22_protocol_manifest.tsv")
        assert len(keeps) == 22
        assert {row["reward_score_mode"] for row in keeps} == {"distance_continuation"}
        inventory = root / "s0_environment/e178_authority_sha_inventory.tsv"
        inventory.write_bytes(inventory.read_bytes() + b"\n")
        try:
            authority.validate(root)
        except RuntimeError as error:
            assert "immutable E187 artifact mismatch" in str(error)
        else:
            raise AssertionError("tampered E187 inventory was accepted")


def main() -> int:
    """Run contracts directly without pytest collection."""
    tests = (
        test_formal_sources_and_projection,
        test_inventory_coverage,
        test_end_to_end_idempotent_and_tamper_rejection,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"E187_AUTHORITY_TESTS=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
