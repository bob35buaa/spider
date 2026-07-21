#!/usr/bin/env python3
"""Build E166 remaining4 A/A_B2_postSmooth manifest."""

from __future__ import annotations

import json
from pathlib import Path

import build_foot_smooth_manifest as base


REPO = base.REPO
SCRIPT_ROOT = base.SCRIPT_ROOT
PREFLIGHT_ROOT = base.PREFLIGHT_ROOT

TARGET_CASES = [
    "box023_person2",
    "box021_029_p2",
    "box021_035_p1",
    "box004_083_p1",
]
CASE_SPLITS = {
    "box023_person2": "local-gpu0",
    "box021_035_p1": "local-gpu0",
    "box021_029_p2": "remote-gpu0",
    "box004_083_p1": "remote-gpu1",
}
ARMS = ["baseline", "A", "A_B2_postSmooth"]

VARIANTS_TSV = SCRIPT_ROOT / "remaining4_variants.tsv"
PREFLIGHT_TSV = PREFLIGHT_ROOT / "e166_remaining4_A_A_B2_preflight.tsv"
SUMMARY_JSON = PREFLIGHT_ROOT / "e166_remaining4_A_A_B2_preflight_summary.json"


def main() -> None:
    base.TARGET_CASES = TARGET_CASES
    base.CASE_SPLITS = CASE_SPLITS
    base.ARMS = ARMS
    rows, preflight, summary = base.build()
    for item in preflight:
        if item["arm"] == "A_B2_postSmooth" and not item["postprocess_source_exists"]:
            item["postprocess_source_pending"] = True
            required = [
                "override_exists",
                "base_override_exists",
                "task_dir_exists",
                "object_asset_exists",
                "mask_exists",
                "e163_result_npz_exists",
                "e163_outdir_exists",
                "e163_video_exists",
            ]
            item["ok"] = all(bool(item[key]) for key in required)
    summary.update(
        {
            "scope": "remaining4_A_A_B2",
            "excluded_cases": [
                "box021_035_p2",
                "box004_082_p1",
                "box004_083_p2",
                "box026_139_p1",
            ],
            "variants_tsv": base.rel(VARIANTS_TSV),
            "preflight_note": "A_B2_postSmooth postprocess source is expected after A CEM finishes.",
        }
    )
    summary["preflight_ok"] = all(bool(row["ok"]) for row in preflight)
    base.write_tsv(VARIANTS_TSV, rows, base.FIELDS)
    base.write_tsv(PREFLIGHT_TSV, preflight, base.PREFLIGHT_FIELDS)
    base.write_json(SUMMARY_JSON, summary)
    print(
        "E166 remaining4 manifest: "
        f"rows={len(rows)} cem_to_run={summary['cem_to_run_total']} "
        f"postprocess_to_run={summary['postprocess_to_run_total']} "
        f"preflight_ok={summary['preflight_ok']} split_counts={summary['split_counts']}"
    )
    if not summary["preflight_ok"]:
        for row in preflight:
            if not row["ok"]:
                print(json.dumps(row, ensure_ascii=False, sort_keys=True))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
