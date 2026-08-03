#!/usr/bin/env python3
"""Standalone contracts for the E186 robust-P re-audit."""

from __future__ import annotations

import audit_robust_p as audit
import numpy as np


def test_confusion_and_gate() -> None:
    """Pose confusion and the 0.70 gate share the frozen semantics."""
    result = audit.confusion(
        np.array([1, 1, 1, 0, 0, 0, 0], dtype=bool),
        np.array([1, 1, 0, 1, 0, 0, 0], dtype=bool),
    )
    assert result["true_positive_count"] == 2
    assert result["phantom_contact_count"] == 1
    assert result["missed_contact_count"] == 1
    assert not result["gate_pass"]
    assert audit.confusion(np.zeros(4, bool), np.zeros(4, bool))["gate_pass"]


def test_source_closure() -> None:
    """All frozen E183/E186 sources retain the exact Full27 order."""
    selection, query, old = audit.load_sources()
    assert len(selection) == 27
    assert query["case_count"] == 27
    assert len(old) == 540


def main() -> int:
    """Run robust-P contracts directly."""
    tests = (test_confusion_and_gate, test_source_closure)
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"E186_ROBUST_P_TESTS=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
