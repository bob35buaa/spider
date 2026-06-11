#!/usr/bin/env python3
"""Build E091 top-bank manifest from D005b and visual-QC outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_V2_ROOT = Path("/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def as_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


def score_row(row: dict[str, str]) -> tuple[float, str]:
    gate_pass = as_bool(row["gate_pass"])
    reasons = [r for r in row.get("reject_reasons", "").split(",") if r]
    inside_max = max(as_float(row["L_inside_frac"]), as_float(row["R_inside_frac"]))
    support = as_float(row["support_face_frac_either"])
    pelvis = as_float(row["pelvis_z_min"])
    signed_min = min(as_float(row["L_signed_dist_mean_m"]), as_float(row["R_signed_dist_mean_m"]))
    gap_max = max(as_float(row["L_wrist_below_pelvis_gap_m"]), as_float(row["R_wrist_below_pelvis_gap_m"]))

    score = 50.0
    if gate_pass:
        score += 25.0
    score -= 7.0 * len(reasons)
    score += min(15.0, support * 20.0)
    score += min(10.0, signed_min * 40.0)
    score -= inside_max * 60.0
    score -= max(0.0, gap_max - 0.20) * 20.0
    score += max(0.0, pelvis - 0.60) * 20.0

    hard_reject = False
    hard_reasons: list[str] = []
    if inside_max > 0.20:
        hard_reject = True
        hard_reasons.append("inside_max_>20%")
    if pelvis < 0.55:
        hard_reject = True
        hard_reasons.append("pelvis_z_min_<0.55")
    if not row.get("overlay_png") or not row.get("timeline_png") or not row.get("keyframes_png"):
        hard_reject = True
        hard_reasons.append("missing_visual_evidence")

    if hard_reject:
        return score, "hard_reject:" + ",".join(hard_reasons)
    if gate_pass:
        return score, "top_candidate"
    return score, "review_only"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-root", type=Path, default=DEFAULT_V2_ROOT)
    args = parser.parse_args()

    d005b_tsv = args.v2_root / "results/d005b_g1_feasibility/d005b_summary.tsv"
    rows = read_tsv(d005b_tsv)
    ranked: list[dict[str, Any]] = []
    for row in rows:
        score, bank_decision = score_row(row)
        out = {
            "rank": 0,
            "task": row["task"],
            "object_name": row["object_name"],
            "sequence": row["sequence"],
            "person": row["person"],
            "bank_decision": bank_decision,
            "top_bank_score": round(score, 3),
            "gate_pass": row["gate_pass"],
            "reject_reasons": row["reject_reasons"],
            "T": row["T"],
            "inside_max": round(max(as_float(row["L_inside_frac"]), as_float(row["R_inside_frac"])), 6),
            "support_face_frac_either": row["support_face_frac_either"],
            "pelvis_z_min": row["pelvis_z_min"],
            "overlay_png": row["overlay_png"],
            "timeline_png": row["timeline_png"],
            "keyframes_png": row["keyframes_png"],
        }
        ranked.append(out)

    decision_order = {"top_candidate": 0, "review_only": 1}
    ranked.sort(key=lambda r: (decision_order.get(str(r["bank_decision"]).split(":")[0], 2), -float(r["top_bank_score"])))
    for i, row in enumerate(ranked, start=1):
        row["rank"] = i

    fields = [
        "rank",
        "task",
        "object_name",
        "sequence",
        "person",
        "bank_decision",
        "top_bank_score",
        "gate_pass",
        "reject_reasons",
        "T",
        "inside_max",
        "support_face_frac_either",
        "pelvis_z_min",
        "overlay_png",
        "timeline_png",
        "keyframes_png",
    ]
    out_dir = args.v2_root / "results/top_bank"
    write_tsv(out_dir / "top_medium_box_bank_manifest.tsv", ranked, fields)
    write_json(out_dir / "top_medium_box_bank_manifest.json", ranked)

    top = [row for row in ranked if row["bank_decision"] == "top_candidate"]
    lines = [
        "# E091 Top Medium-Box Bank",
        "",
        f"- Input D005b TSV: `{d005b_tsv}`",
        f"- Top candidates: `{len(top)}`",
        f"- Review-only / rejects: `{len(ranked) - len(top)}`",
        "",
        "| rank | task | decision | score | D005b | reason |",
        "|---:|---|---|---:|---:|---|",
    ]
    for row in ranked:
        lines.append(
            f"| {row['rank']} | `{row['task']}` | {row['bank_decision']} | "
            f"{row['top_bank_score']} | {row['gate_pass']} | {row['reject_reasons']} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote top bank rows: {len(ranked)}")
    print(f"Top candidates: {len(top)}")


if __name__ == "__main__":
    main()
