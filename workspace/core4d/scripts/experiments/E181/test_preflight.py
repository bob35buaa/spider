#!/usr/bin/env python3
"""Direct-main tests for E181 S0 authority and environment probes."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

from build_authority import (
    DEFAULT_SOURCE,
    DEV_CASE_IDS,
    EXPECTED_SOURCE_SHA256,
    build_authority,
)
from probe_environment import compile_probe, dependency_versions


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a TSV file."""
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def test_authority_projection() -> None:
    """The projection must remain 27/3/24 and preserve ordered authority."""
    with tempfile.TemporaryDirectory(prefix="e181_authority_test_") as directory:
        output_root = Path(directory)
        manifest = build_authority(
            source_path=DEFAULT_SOURCE,
            output_root=output_root,
            expected_sha256=EXPECTED_SOURCE_SHA256,
        )
        full_rows = read_rows(output_root / "full27.tsv")
        dev_rows = read_rows(output_root / "dev3.tsv")
        heldout_rows = read_rows(output_root / "heldout24.tsv")
        assert len(full_rows) == 27
        assert len(dev_rows) == 3
        assert len(heldout_rows) == 24
        assert [row["case_id"] for row in dev_rows] == list(DEV_CASE_IDS)
        assert not (
            {row["case_id"] for row in dev_rows}
            & {row["case_id"] for row in heldout_rows}
        )
        assert [int(row["authority_row_index"]) for row in full_rows] == list(
            range(1, 28)
        )
        assert manifest["source_manifest_sha256"] == EXPECTED_SOURCE_SHA256
        assert manifest["input_check_count"] == 162


def test_dependency_contract() -> None:
    """Pinned packages and CPU/MJWarp compile probes must be usable."""
    versions = dependency_versions()
    assert versions["coacd"] == "1.0.11"
    assert versions["trimesh"] == "4.11.5"
    for geom_type in ("mesh", "sdf"):
        result = compile_probe(geom_type)
        assert result["status"] == "PASS"
        assert result["cpu_ngeom"] == 1
        assert result["warp_ngeom"] == 1


def main() -> int:
    """Run tests without requiring pytest."""
    tests = (
        test_authority_projection,
        test_dependency_contract,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"E181_PREFLIGHT_TESTS=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
