#!/usr/bin/env python3
"""Select bounded E027 full-rollout candidates from offline evidence."""

from __future__ import annotations

import argparse
from typing import Any

import e027_common as C


def select_candidates(limit: int) -> list[dict[str, Any]]:
    timing = C.read_rows(C.TIMING / "timing_summary.csv")
    quality = C.index_by_case(C.load_quality_rows())
    candidates: list[dict[str, Any]] = []
    for row in timing:
        if row.get("recommended_next_action") != "E027_full_variant":
            continue
        case = row["case"]
        q = quality.get(case, {})
        shift = int(float(row.get("recommended_shift_frames") or 0))
        shift_tag = f"m{abs(shift)}" if shift < 0 else f"p{shift}"
        variant = f"E027_{case}_phase_{shift_tag}_hold2_surfacegate"
        candidates.append(
            {
                "variant": variant,
                "case": case,
                "base_variant": row.get("selected_variant", ""),
                "quality_label": q.get("quality_label", ""),
                "proposed_change": "contact_phase_shift+hold_window+nearfield_surface_gate",
                "recommended_shift_frames": shift,
                "expected_contact_gain_pp": row.get("expected_contact_gain_pp", ""),
                "dominant_miss_class": row.get("dominant_miss_class", ""),
                "priority": "P0" if case in C.P0_CASES else "P1",
                "requires_core_change": True,
                "evidence_rationale": (
                    f"timing panel suggests shift {shift} frames; "
                    f"expected gain {row.get('expected_contact_gain_pp')}pp; "
                    f"label={q.get('quality_label', '')}"
                ),
            }
        )
    return candidates[:limit]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=6)
    args = ap.parse_args()
    candidates = select_candidates(args.limit)
    C.write_rows(C.RESULTS / "full_variant_candidates.tsv", candidates, delimiter="\t")
    C.write_json(
        C.RESULTS / "full_variant_candidates.json",
        {"num_candidates": len(candidates), "candidates": candidates},
    )
    print(f"[E027] selected {len(candidates)} full candidates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
