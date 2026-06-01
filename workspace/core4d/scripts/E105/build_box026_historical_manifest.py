#!/usr/bin/env python3
"""Freeze historical Box026 full-CEM metrics used by E105 comparisons."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))

from e105_common import OLD_SUMMARIES, REPO, RESULTS_ROOT, rel, rows  # noqa: E402


OUT_TSV = RESULTS_ROOT / "box026_historical_full_cem_manifest.tsv"
OUT_JSON = RESULTS_ROOT / "box026_historical_full_cem_manifest.json"
OUT_MD = RESULTS_ROOT / "box026_historical_full_cem_manifest.md"

FIELDS = [
    "new_variant",
    "old_variant",
    "old_experiment",
    "source_task",
    "target_route",
    "old_summary_path",
    "old_status",
    "old_T",
    "old_contact_frac_either",
    "old_obj_err_mean_m",
    "old_obj_err_max_m",
    "old_pelvis_min_m",
    "old_head_pen_frac",
    "old_upper_pen_frac",
    "old_handL_floor_lt_5cm_frac",
    "old_handR_floor_lt_5cm_frac",
    "validity",
]


def read_csv(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {row["variant"]: row for row in csv.DictReader(f)}


def pick(row: dict[str, str], *names: str) -> str:
    for name in names:
        if name in row and row[name] != "":
            return row[name]
    return ""


def main() -> None:
    old_tables = {
        "E092": read_csv(REPO / OLD_SUMMARIES["E092"]),
        "E094": read_csv(REPO / OLD_SUMMARIES["E094"]),
    }
    out_rows: list[dict[str, str]] = []
    for variant in rows():
        old = variant["old_variant"]
        if not old:
            continue
        exp = "E092" if old.startswith("E092") else "E094"
        old_row = old_tables.get(exp, {}).get(old, {})
        out_rows.append(
            {
                "new_variant": variant["variant"],
                "old_variant": old,
                "old_experiment": exp,
                "source_task": variant["source_task"],
                "target_route": variant["target_route"],
                "old_summary_path": variant["old_summary_path"],
                "old_status": pick(old_row, "work_status", "status"),
                "old_T": pick(old_row, "T"),
                "old_contact_frac_either": pick(old_row, "contact_frac_either"),
                "old_obj_err_mean_m": pick(old_row, "obj_err_mean_m"),
                "old_obj_err_max_m": pick(old_row, "obj_err_max_m"),
                "old_pelvis_min_m": pick(old_row, "pelvis_min_m"),
                "old_head_pen_frac": pick(old_row, "head_pen_frac"),
                "old_upper_pen_frac": pick(old_row, "upper_pen_frac"),
                "old_handL_floor_lt_5cm_frac": pick(old_row, "handL_floor_lt_5cm_frac"),
                "old_handR_floor_lt_5cm_frac": pick(old_row, "handR_floor_lt_5cm_frac"),
                "validity": "invalidated_by_e103_scene_inertial_bug",
            }
        )

    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(out_rows)
    OUT_JSON.write_text(json.dumps(out_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# E105 Historical Box026 Full-CEM Manifest",
        "",
        "| new variant | old variant | old status | contact | obj mean | obj max | pelvis | validity |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in out_rows:
        lines.append(
            f"| `{row['new_variant']}` | `{row['old_variant']}` | {row['old_status']} | "
            f"{row['old_contact_frac_either']} | {row['old_obj_err_mean_m']} | "
            f"{row['old_obj_err_max_m']} | {row['old_pelvis_min_m']} | {row['validity']} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {rel(OUT_TSV)} rows={len(out_rows)}")
    print(f"wrote {rel(OUT_MD)}")


if __name__ == "__main__":
    main()
