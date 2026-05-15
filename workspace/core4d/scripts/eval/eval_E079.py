#!/usr/bin/env python3
"""E079 evaluation wrapper for multi-case CORE4D generalization."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
EVAL_DIR = REPO / "workspace/core4d/scripts/eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import eval_E078 as e078  # noqa: E402


RESULTS = REPO / "workspace/core4d/results/E079"
VARIANTS_FILE = REPO / "workspace/core4d/scripts/E079/variants.tsv"


def read_variants() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    fieldnames = ["variant", "task", "mask_slug", "person_idx", "split", "role"]
    with VARIANTS_FILE.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(
            (line for line in f if line.strip() and not line.startswith("#")),
            delimiter="\t",
            fieldnames=fieldnames,
        )
        for row in reader:
            if row["role"] not in {"main", "guard"}:
                continue
            out[row["variant"]] = {
                "name": row["variant"],
                "case": row["task"],
                "override": f"core4d_{row['variant']}",
                "split": row["split"],
                "role": row["role"],
            }
    return out


def main() -> None:
    variants = read_variants()
    selected = sys.argv[1:] or list(variants.keys())

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "plots").mkdir(parents=True, exist_ok=True)
    e078.RESULTS = RESULTS
    e078.VARIANTS = variants

    summaries = []
    for variant in selected:
        if variant not in variants:
            print(f"[SKIP] unknown variant {variant}")
            continue
        try:
            summary = e078.evaluate_variant(variant)
        except FileNotFoundError as exc:
            print(f"[SKIP] {exc}")
            continue
        summary["split"] = variants[variant]["split"]
        summary["role"] = variants[variant]["role"]
        summary["E079_success_numeric"] = bool(
            summary["post2_pelvis_z_min_m"] >= 0.55
            and summary["post2_sim_contact_frames_pct"] >= 50.0
            and summary["post2_obj_err_mean_m"] <= 0.20
        )
        summaries.append(summary)

    if not summaries:
        raise SystemExit("No E079 variant results found.")

    keys = sorted({k for row in summaries for k in row.keys()})
    comparison = RESULTS / "comparison.csv"
    with comparison.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summaries)

    main_rows = [r for r in summaries if r["role"] == "main"]
    aggregate = {
        "num_results": len(summaries),
        "num_main_results": len(main_rows),
        "num_main_numeric_success": sum(bool(r["E079_success_numeric"]) for r in main_rows),
        "main_numeric_success_pct": (
            100.0 * sum(bool(r["E079_success_numeric"]) for r in main_rows) / len(main_rows)
            if main_rows
            else 0.0
        ),
        "guard_results": [r["variant"] for r in summaries if r["role"] == "guard"],
    }
    (RESULTS / "aggregate_summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(f"Wrote {comparison}")
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
