#!/usr/bin/env python3
"""Evaluate E090 retargeted SPIDER trajectory geometry with the G1 gate."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "workspace/exp_diagnostic/scripts"))

from g1_feasibility_gate import evaluate, GATE  # noqa: E402

TASKS = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
DEFAULT_VARIANTS = REPO / "workspace/core4d/results/E090/variants.tsv"
DEFAULT_OUT = REPO / "workspace/core4d/results/E090/geometry"


def read_variants(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open() as f:
        for raw in f:
            if not raw.strip() or raw.startswith("#"):
                continue
            parts = raw.rstrip("\n").split("\t")
            if len(parts) != 8:
                raise ValueError(f"Bad variants row ({len(parts)} cols): {raw}")
            rows.append(
                dict(
                    variant=parts[0],
                    base_task=parts[1],
                    target_task=parts[2],
                    case_root=parts[3],
                    converted_dir=parts[4],
                    retargeted_dir=parts[5],
                    trimmed_dir=parts[6],
                    source_scene_task=parts[7],
                )
            )
    return rows


def flatten(row: dict[str, str], result: dict) -> dict[str, object]:
    out: dict[str, object] = {
        "variant": row["variant"],
        "base_task": row["base_task"],
        "target_task": row["target_task"],
        "gate_pass": result.get("gate_pass", False),
        "gate_reject_reasons": ",".join(result.get("gate_reject_reasons", [])),
        "T": result.get("T"),
        "pelvis_z_min": result.get("pelvis_z_min"),
    }
    for hand in ["L", "R"]:
        stats = result.get(hand, {})
        for key in [
            "inside_box_frac",
            "signed_dist_mean_m",
            "signed_dist_min_m",
            "top_face_frac",
            "legacy_local_z_face_frac",
            "support_face_frac",
            "wrist_below_pelvis_gap_m",
            "wrist_world_z_mean_m",
            "wrist_world_z_min_m",
        ]:
            out[f"{hand}_{key}"] = stats.get(key)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", type=Path, default=DEFAULT_VARIANTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_variants(args.variants)
    results = []
    flat_rows = []
    for row in rows:
        result = evaluate(TASKS / row["target_task"])
        result["variant"] = row
        results.append(result)
        flat_rows.append(flatten(row, result))

    json_path = args.out_dir / "geometry_summary.json"
    json_path.write_text(json.dumps({"gate": GATE, "results": results}, indent=2))

    csv_path = args.out_dir / "geometry_summary.csv"
    fieldnames = list(flat_rows[0].keys()) if flat_rows else []
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)

    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    for row in flat_rows:
        print(
            f"{row['target_task']}: pass={row['gate_pass']} "
            f"L_in={float(row['L_inside_box_frac'] or 0)*100:.1f}% "
            f"R_in={float(row['R_inside_box_frac'] or 0)*100:.1f}% "
            f"L_sup={float(row['L_support_face_frac'] or 0)*100:.1f}% "
            f"R_sup={float(row['R_support_face_frac'] or 0)*100:.1f}% "
            f"reasons={row['gate_reject_reasons']}"
        )


if __name__ == "__main__":
    main()
