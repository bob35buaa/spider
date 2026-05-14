#!/usr/bin/env python3
"""E075 evaluation for limited hold/contact variants.

This reuses the E074 evaluator and redirects its paths to E075 so the metric
definitions stay byte-for-byte consistent across the comparison.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
EVAL_DIR = REPO / "workspace/core4d/scripts/eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import eval_E074 as e074  # noqa: E402

RESULTS = REPO / "workspace/core4d/results/E075"

e074.RESULTS = RESULTS
e074.SCENE_SNAPSHOT = RESULTS / "scene_snapshot/box023_person1/scene_act.xml"


def main() -> None:
    variants = sys.argv[1:] or ["E075B", "E075A"]
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "plots").mkdir(parents=True, exist_ok=True)
    model, scene_used = e074.load_scene_model()
    summaries = []
    for variant in variants:
        try:
            summaries.append(e074.evaluate_variant(variant, model, scene_used))
        except FileNotFoundError as exc:
            print(f"[SKIP] {exc}")

    if summaries:
        keys = sorted({k for row in summaries for k in row.keys()})
        comparison = RESULTS / "comparison.csv"
        with comparison.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(summaries)
        print(f"Wrote {comparison}")
        print(json.dumps(summaries, indent=2, sort_keys=True))
    else:
        raise SystemExit("No E075 variant results found.")


if __name__ == "__main__":
    main()
