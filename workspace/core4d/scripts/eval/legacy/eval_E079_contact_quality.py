#!/usr/bin/env python3
"""Summarize E079 3cm contact-mask quality before CEM runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[4]
RESULTS = REPO / "workspace/core4d/results/E079"
VARIANTS = REPO / "workspace/core4d/scripts/E079/variants.tsv"


def read_variants() -> list[dict[str, str]]:
    fieldnames = ["variant", "task", "mask_slug", "person_idx", "split", "role"]
    with VARIANTS.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(
            (line for line in f if line.strip() and not line.startswith("#")),
            delimiter="\t",
            fieldnames=fieldnames,
        )
        return [row for row in reader if row["role"] in {"main", "guard"}]


def pct(x: np.ndarray) -> float:
    return float(np.asarray(x, dtype=float).mean() * 100.0)


def main() -> None:
    rows = []
    for row in read_variants():
        slug = row["mask_slug"]
        person_idx = int(row["person_idx"])
        path = RESULTS / "contact_masks" / slug / "raw_contact_mask_3cm.npz"
        if not path.is_file():
            rows.append(
                {
                    "variant": row["variant"],
                    "task": row["task"],
                    "mask_path": str(path.relative_to(REPO)),
                    "status": "missing",
                }
            )
            continue
        data = np.load(path, allow_pickle=True)
        spider_mask = data["spider_contact_mask_3cm"][:, person_idx, :]
        spider_dist = data["spider_min_dist_m"][:, person_idx, :]
        active_any = spider_mask.any(axis=1)
        active_hand = spider_mask.astype(bool)
        if active_hand.any():
            active_dist = spider_dist[active_hand]
            active_dist_mean = float(active_dist.mean())
            active_dist_max = float(active_dist.max())
        else:
            active_dist_mean = float("nan")
            active_dist_max = float("nan")

        rows.append(
            {
                "variant": row["variant"],
                "task": row["task"],
                "split": row["split"],
                "role": row["role"],
                "mask_path": str(path.relative_to(REPO)),
                "status": "ok",
                "T_spider": int(spider_mask.shape[0]),
                "trim_start": int(data["trim_start"]),
                "left_active_pct": pct(spider_mask[:, 0]),
                "right_active_pct": pct(spider_mask[:, 1]),
                "any_active_pct": pct(active_any),
                "left_dist_mean_m": float(spider_dist[:, 0].mean()),
                "right_dist_mean_m": float(spider_dist[:, 1].mean()),
                "active_dist_mean_m": active_dist_mean,
                "active_dist_max_m": active_dist_max,
                "high_quality_proxy": bool(
                    pct(active_any) >= 40.0 and active_dist_mean <= 0.03
                ),
            }
        )

    out_csv = RESULTS / "contact_quality.csv"
    out_json = RESULTS / "contact_quality.json"
    RESULTS.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r.keys()})
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    out_json.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
