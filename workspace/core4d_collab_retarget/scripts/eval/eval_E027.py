#!/usr/bin/env python3
"""Summarize E027 offline audit outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

THIS = Path(__file__).resolve()
REPO = THIS.parents[4]
E027_DIR = REPO / "workspace/core4d_collab_retarget/scripts/E027"
sys.path.insert(0, str(E027_DIR))

import e027_common as C  # noqa: E402


def summarize_offline() -> dict[str, Any]:
    quality = C.load_quality_rows()
    timing = C.read_rows(C.TIMING / "timing_summary.csv")
    candidates = C.read_rows(C.RESULTS / "full_variant_candidates.tsv", delimiter="\t")
    labels: dict[str, int] = {}
    for row in quality:
        label = row.get("quality_label", "")
        labels[label] = labels.get(label, 0) + 1
    summary = {
        "num_quality_rows": len(quality),
        "quality_coverage_ok": len(quality) == len(C.CASES_13),
        "label_counts": labels,
        "discard_from_p0": [r["case"] for r in quality if C.as_bool(r.get("discard_from_p0"))],
        "discard_from_success_denominator": [
            r["case"] for r in quality if C.as_bool(r.get("discard_from_success_denominator"))
        ],
        "num_timing_panels": len(timing),
        "timing_cases": [r["case"] for r in timing],
        "num_full_candidates": len(candidates),
        "full_candidates": [r.get("variant", "") for r in candidates],
    }
    C.write_json(C.RESULTS / "offline_summary.json", summary)
    lines = [
        "# E027 Offline Summary",
        "",
        f"- Quality rows: `{summary['num_quality_rows']}/13`",
        f"- Timing panels: `{summary['num_timing_panels']}`",
        f"- Full candidates: `{summary['num_full_candidates']}`",
        "",
        "## Label Counts",
        "",
    ]
    for label, count in sorted(labels.items()):
        lines.append(f"- `{label}`: {count}")
    lines.extend(
        [
            "",
            "## Discard Decisions",
            "",
            f"- drop from P0: {', '.join(summary['discard_from_p0']) or 'none'}",
            f"- drop from success denominator: {', '.join(summary['discard_from_success_denominator']) or 'none'}",
            "",
            "## Full Candidates",
            "",
        ]
    )
    if candidates:
        lines.append("| Variant | Case | Proposed Change | Rationale |")
        lines.append("|---|---|---|---|")
        for row in candidates:
            lines.append(
                f"| `{row.get('variant', '')}` | `{row.get('case', '')}` | "
                f"{row.get('proposed_change', '')} | {row.get('evidence_rationale', '')} |"
            )
    else:
        lines.append("No E027 full rollout candidates selected by offline evidence.")
    (C.RESULTS / "offline_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    if not args.offline:
        ap.error("Only --offline is implemented for E027 first phase")
    summary = summarize_offline()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
