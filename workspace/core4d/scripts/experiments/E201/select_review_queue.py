#!/usr/bin/env python3
"""E201 · select the L2/L3-review rollouts that need human (and VLM pre-screen).

Joins the funnel classifier output (layer / family_flag / failed gates) with the
source case_metrics (render paths + frame count) and emits a review-queue TSV:
one row per rollout whose layer is L2_review or L3_review.

Usage:
    .venv/bin/python workspace/core4d/scripts/experiments/E201/select_review_queue.py --exp E199
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/eval/reports"))
import gen_E199_fullscale_gate_xlsx as X  # noqa: E402  (CASE_METRICS + read_tsv)

FUNNEL_DIR = REPO / "workspace/core4d/results/E201/funnel"
VLM_DIR = REPO / "workspace/core4d/results/E201/vlm_review"

# experiment -> source case_metrics tsv (funnel_rollout path is derived by name)
CASE_METRICS = {"E199": X.CASE_METRICS}

REVIEW_LAYERS = ("L2_review", "L3_review")

OUT_COLS = [
    "exp", "object_key", "case_id", "aug_variant", "group", "layer", "family_flag",
    "wide_failed", "narrow_failed", "qpos_frames", "duration_s",
    "qpos_path", "scene_act", "scene_xml",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="E199", choices=sorted(CASE_METRICS))
    ap.add_argument("--rollout", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rollout_tsv = args.rollout or (FUNNEL_DIR / f"{args.exp}_funnel_rollout.tsv")
    layer_rows = X.read_tsv(rollout_tsv)
    src = {(m["case_id"], m.get("aug_variant", "")): m
           for m in X.read_tsv(CASE_METRICS[args.exp])}

    out_rows = []
    for lr in layer_rows:
        if lr["layer"] not in REVIEW_LAYERS:
            continue
        m = src.get((lr["case_id"], lr.get("aug_variant", "")), {})
        out_rows.append({
            "exp": args.exp,
            "object_key": lr.get("object_key", ""),
            "case_id": lr["case_id"],
            "aug_variant": lr.get("aug_variant", ""),
            "group": lr.get("group", ""),
            "layer": lr["layer"],
            "family_flag": lr.get("family_flag", ""),
            "wide_failed": lr.get("wide_failed", ""),
            "narrow_failed": lr.get("narrow_failed", ""),
            "qpos_frames": m.get("qpos_frames", ""),
            "duration_s": m.get("duration_s", ""),
            "qpos_path": m.get("qpos_path", ""),
            "scene_act": m.get("scene_act", ""),
            "scene_xml": m.get("scene_xml", ""),
        })

    out = args.out or (VLM_DIR / f"{args.exp}_review_queue.tsv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=OUT_COLS, delimiter="\t")
        w.writeheader()
        w.writerows(out_rows)

    n_l2 = sum(1 for r in out_rows if r["layer"] == "L2_review")
    n_l3 = sum(1 for r in out_rows if r["layer"] == "L3_review")
    miss = sum(1 for r in out_rows if not r["qpos_path"] or not r["scene_act"])
    print(f"[done] wrote {out} ({len(out_rows)} rows: L2={n_l2} L3_review={n_l3}; missing paths={miss})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
