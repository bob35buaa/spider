from __future__ import annotations

from build_evaluation_manifest import build_rows


def test_evaluation_manifest_is_exact_keep22() -> None:
    rows = build_rows()
    assert len(rows) == 22
    assert len({row["case_id"] for row in rows}) == 22
    assert sum(row["execution_kind"] == "PROMOTED_CANARY" for row in rows) == 3
    assert sum(row["execution_kind"] == "FULL_CEM" for row in rows) == 19


def test_evaluation_manifest_preserves_c9_waiver() -> None:
    rows = build_rows()
    assert {row["c9_technical_status"] for row in rows} == {"FAIL"}
    assert {row["c9_progression_authority"] for row in rows} == {"USER_WAIVED"}
    assert {
        (row["cem_samples"], row["cem_opt_steps"], row["cem_seed"]) for row in rows
    } == {(1024, 32, 0)}


def main() -> int:
    tests = (
        test_evaluation_manifest_is_exact_keep22,
        test_evaluation_manifest_preserves_c9_waiver,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"E187_EVALUATION_MANIFEST_TESTS=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
