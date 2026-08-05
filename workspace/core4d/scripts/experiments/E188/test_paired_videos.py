#!/usr/bin/env python3
"""Direct contracts for E188 final and paired video authority."""

from __future__ import annotations

from render_paired_videos import authority, filter_graph


def main() -> int:
    rows = authority()
    assert len(rows) == 15 and len({row["case_id"] for row in rows}) == 15
    graph = filter_graph(rows[0])
    assert "LEFT - E187 2KG" in graph and "RIGHT - E188 5KG" in graph
    assert "[left][right]hstack=inputs=2" in graph
    print("E188_PAIRED_VIDEO_TESTS=PASS rows=15 layout=E187_left_E188_right")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
