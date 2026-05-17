#!/usr/bin/env python3
"""E004 evaluation wrapper for true-freejoint virtual-partner support."""

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


RESULTS = REPO / "workspace/core4d_collab_retarget/results/E004"
VARIANTS_FILE = REPO / "workspace/core4d_collab_retarget/scripts/E004/variants.tsv"

FIELDNAMES = [
    "variant",
    "source_task",
    "mask_slug",
    "person_idx",
    "queue",
    "role",
    "wave",
    "partner_force_scale",
    "partner_force_spring_kp",
    "partner_force_spring_kd",
    "partner_force_spring_kp_rot",
    "partner_force_spring_kd_rot",
    "partner_force_rot_clamp",
    "hold_contact_rew_scale",
    "hold_contact_sigma",
    "hold_contact_start_eval_time",
    "hold_contact_end_eval_time",
    "hold_contact_require_ref_contact",
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
            if row["queue"] == "probe":
                continue
            out[row["variant"]] = {
                "name": row["variant"],
                "case": row["source_task"],
                "source_task": row["source_task"],
                "override": f"core4d_collab_{row['variant']}",
                "split": row["queue"],
                "role": row["role"],
                **row,
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
        raise ValueError(f"Unknown E004 variant: {variant}")

    e002.RESULTS = RESULTS
    e002.VARIANTS_FILE = VARIANTS_FILE
    summary = e002.evaluate_variant(variant, variants)
    meta = variants[variant]

    summary["E004_wave"] = meta["wave"]
    summary["E004_queue"] = meta["queue"]
    summary["E004_partner_force_scale"] = float(meta["partner_force_scale"])
    summary["E004_partner_force_spring_kp"] = float(meta["partner_force_spring_kp"])
    summary["E004_partner_force_spring_kd"] = float(meta["partner_force_spring_kd"])
    summary["E004_partner_force_spring_kp_rot"] = float(meta["partner_force_spring_kp_rot"])
    summary["E004_partner_force_spring_kd_rot"] = float(meta["partner_force_spring_kd_rot"])
    summary["E004_partner_force_rot_clamp"] = float(meta["partner_force_rot_clamp"])
    summary["E004_hold_contact_rew_scale"] = float(meta["hold_contact_rew_scale"])
    summary["E004_hold_contact_sigma"] = float(meta["hold_contact_sigma"])
    summary["E004_hold_contact_start_eval_time"] = float(meta["hold_contact_start_eval_time"])
    summary["E004_hold_contact_end_eval_time"] = float(meta["hold_contact_end_eval_time"])
    summary["E004_freejoint_parity_ok"] = bool(
        not summary["config_contact_guidance"]
        and summary["config_scene_name"] == ""
        and summary["config_nu"] == 29
        and summary["config_nq_obj"] == 7
        and summary["config_object_action_dims"] == 0
        and len(summary["config_object_actuator_ids"]) == 0
    )
    summary["E004_useful_main_proxy"] = bool(
        summary["role"] == "main"
        and summary["case_window_obj_err_mean_m"] < 0.30
        and summary["case_window_obj_err_max_m"] < 0.70
        and summary["case_window_sim_object_floor_contact_frames_pct"] < 76.9
        and summary["E004_freejoint_parity_ok"]
    )
    summary["E004_strong_main_proxy"] = bool(
        summary["role"] == "main"
        and summary["case_window_obj_err_mean_m"] < 0.20
        and summary["case_window_obj_err_max_m"] < 0.35
        and summary["case_window_sim_object_floor_contact_frames_pct"] <= 59.5
        and summary["E004_freejoint_parity_ok"]
    )
    summary["E004_guard_stable_proxy"] = bool(
        summary["role"] == "guard"
        and summary["post2_pelvis_z_min_m"] >= 0.55
        and summary["case_window_sim_leg_box_interference_frames_pct"] <= 5.0
        and summary["E004_freejoint_parity_ok"]
    )
    _write_summary(summary)
    return summary


def normalize_args(args: list[str], variants: dict[str, dict[str, str]]) -> list[str]:
    if not args or args == ["--all"]:
        return list(variants.keys())
    selected: list[str] = []
    for arg in args:
        if arg == "--all":
            selected.extend(variants.keys())
        else:
            selected.append(arg)
    return selected


def main() -> None:
    variants = read_variants()
    selected = normalize_args(sys.argv[1:], variants)

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
        raise SystemExit("No E004 variant results found.")

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
        "num_freejoint_parity_ok": sum(bool(r["E004_freejoint_parity_ok"]) for r in summaries),
        "num_main_useful_proxy": sum(bool(r["E004_useful_main_proxy"]) for r in main_rows),
        "num_main_strong_proxy": sum(bool(r["E004_strong_main_proxy"]) for r in main_rows),
        "num_guard_stable_proxy": sum(bool(r["E004_guard_stable_proxy"]) for r in guard_rows),
        "main_results": [r["variant"] for r in main_rows],
        "guard_results": [r["variant"] for r in guard_rows],
    }
    (RESULTS / "aggregate_summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(f"Wrote {comparison}")
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
