#!/usr/bin/env python3
"""Direct-main contracts for E187 versus E178 paired review videos."""

from __future__ import annotations

from render_paired_evaluation_videos import authority_rows, filter_graph


def test_authority_is_exact_keep22() -> None:
    """Every E187 evaluation row has exactly one E178 video peer."""
    rows = authority_rows()
    assert len(rows) == 22
    assert len({row["case_id"] for row in rows}) == 22
    assert all(row["e178_video"].is_file() for row in rows)
    assert all(row["e187_video"].is_file() for row in rows)


def test_layout_freezes_e178_left_e187_right() -> None:
    """The filter graph visibly freezes experiment order and labels."""
    graph = filter_graph("case_id", "E178_PASS_TO_E187_FAIL")
    assert "[0:v]" in graph and "LEFT - E178 BASELINE" in graph
    assert "[1:v]" in graph and "RIGHT - E187 CONTINUATION" in graph
    assert "[left][right]hstack=inputs=2" in graph
    assert "case_id | E178_PASS_TO_E187_FAIL" in graph


def main() -> int:
    """Run both direct-main video contracts."""
    tests = (
        test_authority_is_exact_keep22,
        test_layout_freezes_e178_left_e187_right,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"E187_PAIRED_VIDEO_TESTS=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
