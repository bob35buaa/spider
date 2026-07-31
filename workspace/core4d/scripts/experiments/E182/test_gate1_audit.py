#!/usr/bin/env python3
"""Direct-main tests for the aggregate E182 Gate1 artifact audit."""

from __future__ import annotations

import copy
import json

from audit_gate1 import (
    DEFAULT_RESULT_ROOT,
    EXPECTED_CASE_IDS,
    audit_gate1_artifacts,
    audit_prg_case,
)


def test_real_dev3_gate1_artifacts() -> None:
    """All three current dev tapes must close the artifact-only portion of Gate1."""
    result = audit_gate1_artifacts(DEFAULT_RESULT_ROOT, run_direct_tests=False)
    assert result["status"] == "PASS"
    assert result["case_ids"] == list(EXPECTED_CASE_IDS)
    assert result["same_run_status"] == "PASS"
    assert result["runtime_inputs"]["status"] == "PASS"
    assert result["heldout_access"] == "NOT_ACCESSED_DEV3_ONLY"
    assert all(case["status"] == "PASS" for case in result["cases"])
    assert all(case["selection_mode"] == "on_a" for case in result["cases"])


def test_selection_sha_tamper_fails_case_audit() -> None:
    """A PRG manifest cannot point at any tape other than frozen on_a."""
    root = DEFAULT_RESULT_ROOT
    same_run = json.loads((root / "same_run_integrity_audit.json").read_text())
    row = same_run["rows"][0]
    case_id = row["case_id"]
    manifest_path = root / "prg_query_tape" / case_id / "manifest.json"
    payload = json.loads(manifest_path.read_text())
    changed = copy.deepcopy(payload)
    changed["raw_cem_content_sha256"] = "0" * 64
    failed = audit_prg_case(
        root / "prg_query_tape",
        row,
        payload=changed,
    )
    assert failed["status"] == "FAIL"
    assert any(item["field"] == "selection_content_sha256" for item in failed["errors"])


def main() -> int:
    """Run tests without pytest discovery."""
    tests = (
        test_real_dev3_gate1_artifacts,
        test_selection_sha_tamper_fails_case_audit,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"E182_GATE1_AUDIT_TESTS=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
