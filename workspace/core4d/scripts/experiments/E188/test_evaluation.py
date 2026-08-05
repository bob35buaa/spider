#!/usr/bin/env python3
"""Direct contracts for E188 evaluation authority and shared metrics."""

from __future__ import annotations

from build_evaluation_manifest import build_rows


def main() -> int:
    rows = build_rows()
    assert len(rows) == 15
    assert len({row["case_id"] for row in rows}) == 15
    assert sum(row["device_scope"] == "same_device_local4" for row in rows) == 4
    assert sum(row["render_mode"] == "INLINE_CEM" for row in rows) == 7
    assert sum(row["render_mode"] == "DEFERRED_LOCAL_RENDER" for row in rows) == 8
    print("E188_EVALUATION_TESTS=PASS rows=15 inline=7 deferred=8 same_device=4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
