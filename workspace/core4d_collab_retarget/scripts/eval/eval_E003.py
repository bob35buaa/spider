#!/usr/bin/env python3
"""E003 evaluation wrapper for true-freejoint physics-sweep variants."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
SCRIPT_EVAL = REPO / "workspace/core4d_collab_retarget/scripts/eval"
if str(SCRIPT_EVAL) not in sys.path:
    sys.path.insert(0, str(SCRIPT_EVAL))

import eval_E002 as e002  # noqa: E402


RESULTS = REPO / "workspace/core4d_collab_retarget/results/E003"
VARIANTS_FILE = REPO / "workspace/core4d_collab_retarget/scripts/E003/variants.tsv"

FIELDNAMES = [
    "variant",
    "source_task",
    "derived_task",
    "mask_source_exp",
    "mask_slug",
    "person_idx",
    "remote_group",
    "role",
    "mass_kg",
    "inertia_scale",
    "hand_object_friction",
    "object_floor_friction",
]


def read_variants() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    with VARIANTS_FILE.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(
            (line for line in f if line.strip() and not line.startswith("#")),
            delimiter="\t",
            fieldnames=FIELDNAMES,
        )
        for row in reader:
            if row["role"] not in {"main", "guard"}:
                continue
            out[row["variant"]] = {
                "name": row["variant"],
                "case": row["derived_task"],
                "source_task": row["source_task"],
                "override": f"core4d_collab_{row['variant']}",
                "split": row["remote_group"],
                "role": row["role"],
                "mass_kg": row["mass_kg"],
                "inertia_scale": row["inertia_scale"],
                "hand_object_friction": row["hand_object_friction"],
                "object_floor_friction": row["object_floor_friction"],
            }
    return out


def _write_summary(summary: dict[str, object]) -> None:
    variant = str(summary["variant"])
    summary_json = RESULTS / f"eval_summary_{variant}.json"
    summary_csv = RESULTS / f"eval_summary_{variant}.csv"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def evaluate_variant(variant: str, variants: dict[str, dict[str, str]]) -> dict[str, object]:
    if variant not in variants:
        raise ValueError(f"Unknown E003 variant: {variant}")

    e002.RESULTS = RESULTS
    e002.VARIANTS_FILE = VARIANTS_FILE
    summary = e002.evaluate_variant(variant, variants)
    meta = variants[variant]

    summary["E003_success_numeric"] = bool(summary["E002_success_numeric"])
    summary["E003_success_case_window"] = bool(summary["E002_success_case_window"])
    summary["E003_success_legobj_strict_proxy"] = bool(summary["E002_success_legobj_strict_proxy"])
    summary["E003_physics_mass_kg"] = float(meta["mass_kg"])
    summary["E003_physics_inertia_scale"] = float(meta["inertia_scale"])
    summary["E003_physics_hand_object_friction"] = float(meta["hand_object_friction"])
    summary["E003_physics_object_floor_friction"] = float(meta["object_floor_friction"])
    summary["E003_guard_physics_feasible_proxy"] = bool(
        summary["role"] == "guard"
        and summary["case_window_obj_err_mean_m"] <= 0.25
        and summary["case_window_sim_object_floor_contact_frames_pct"] <= 50.0
        and summary["case_window_sim_leg_box_interference_frames_pct"] <= 5.0
    )
    _write_summary(summary)
    return summary


def main() -> None:
    variants = read_variants()
    selected = sys.argv[1:] or list(variants.keys())

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "plots").mkdir(parents=True, exist_ok=True)

    summaries = []
    for variant in selected:
        if variant not in variants:
            print(f"[SKIP] unknown variant {variant}")
            continue
        try:
            summaries.append(evaluate_variant(variant, variants))
        except FileNotFoundError as exc:
            print(f"[SKIP] {exc}")
            continue

    if not summaries:
        raise SystemExit("No E003 variant results found.")

    keys = sorted({k for row in summaries for k in row.keys()})
    comparison = RESULTS / "comparison.csv"
    with comparison.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summaries)

    main_rows = [r for r in summaries if r["role"] == "main"]
    guard_rows = [r for r in summaries if r["role"] == "guard"]
    aggregate = {
        "num_results": len(summaries),
        "num_main_results": len(main_rows),
        "num_guard_results": len(guard_rows),
        "num_main_numeric_success": sum(bool(r["E003_success_numeric"]) for r in main_rows),
        "num_main_case_window_success": sum(bool(r["E003_success_case_window"]) for r in main_rows),
        "num_main_legobj_strict_proxy_success": sum(
            bool(r["E003_success_legobj_strict_proxy"]) for r in main_rows
        ),
        "num_guard_physics_feasible_proxy": sum(
            bool(r["E003_guard_physics_feasible_proxy"]) for r in guard_rows
        ),
        "guard_results": [r["variant"] for r in guard_rows],
    }
    (RESULTS / "aggregate_summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(f"Wrote {comparison}")
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
