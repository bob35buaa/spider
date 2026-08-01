#!/usr/bin/env python3
"""Contracts and dev-case regression for the E183 Full27 static-P audit."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import audit_full27_static_p as audit
import numpy as np


def test_authority_and_candidate_inventory() -> None:
    """The frozen scope must remain 27 cases and 60 object-specific candidates."""
    rows = audit.load_full27_rows()
    candidates = audit.candidate_inventory()
    assert len(rows) == 27
    assert len(candidates) == 60
    assert {
        key: sum(row["object_key"] == key for row in rows) for key in audit.OBJECT_KEYS
    } == audit.EXPECTED_OBJECT_COUNTS
    assert {
        key: sum(row["object_key"] == key for row in candidates)
        for key in audit.OBJECT_KEYS
    } == {"bucket003": 24, "bucket004": 18, "bucket007": 18}
    assert sum(row["source"] == "E181" for row in candidates) == 54
    assert sum(row["source"] == "E182_V9" for row in candidates) == 6


def test_contact_metric_gate_and_pooling() -> None:
    """The exact 0.70 precision/recall floor and count pooling must be stable."""
    oracle = np.asarray([1, 1, 1, 0, 0, 0, 0], dtype=bool)
    passing = audit.contact_metrics(
        np.asarray([1, 1, 1, 1, 0, 0, 0], dtype=bool), oracle
    )
    failing = audit.contact_metrics(
        np.asarray([1, 1, 0, 1, 0, 0, 0], dtype=bool), oracle
    )
    assert passing["precision"] == 0.75
    assert passing["recall"] == 1.0
    assert passing["gate_status"] == "PASS"
    assert failing["gate_status"] == "FAIL"
    pooled = audit._merge_confusions([passing, failing])
    assert pooled["pose_count"] == 14
    assert pooled["true_positive_count"] == 5
    assert pooled["phantom_contact_count"] == 2
    assert pooled["missed_contact_count"] == 1
    assert pooled["true_negative_count"] == 6


def test_protocol_and_v9_dev_regression() -> None:
    """A temporary dev query must reproduce all six immutable v9 confusion rows."""
    rows = audit.load_full27_rows()
    dev = next(row for row in rows if row["case_id"] == "bucket003_20231018_001_p1")
    v9_candidates = [
        row for row in audit.candidate_inventory() if row["source"] == "E182_V9"
    ]
    with tempfile.TemporaryDirectory(prefix="e183_test_") as directory:
        root = Path(directory)
        protocol = audit.freeze_protocol(root)
        assert protocol["candidate_count"] == 60
        query = audit.build_query_case(dev, str(root))
        assert query["pose_count"] == 882
        assert query["oracle_contact_count"] == 27
        for entry in v9_candidates:
            result = audit.score_candidate(entry, [dev], str(root))
            metrics = result["case_metrics"][0]
            actual = (
                metrics["true_positive_count"],
                metrics["phantom_contact_count"],
                metrics["missed_contact_count"],
            )
            assert actual == audit.V9_REGRESSION[entry["candidate_id"]]
        first = v9_candidates[0]
        result_path = audit._result_path(root, first)
        tampered = json.loads(result_path.read_text(encoding="utf-8"))
        tampered["case_metrics"][0]["phantom_contact_count"] += 1
        result_path.write_text(json.dumps(tampered), encoding="utf-8")
        try:
            audit.score_candidate(first, [dev], str(root))
        except RuntimeError as error:
            assert "pooled confusion mismatch" in str(error)
        else:
            raise AssertionError("tampered E183 result resumed")


def main() -> int:
    """Run the standalone tests without requiring pytest collection."""
    tests = (
        test_authority_and_candidate_inventory,
        test_contact_metric_gate_and_pooling,
        test_protocol_and_v9_dev_regression,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"E183_TESTS=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
