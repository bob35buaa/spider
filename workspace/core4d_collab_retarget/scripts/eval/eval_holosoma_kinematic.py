#!/usr/bin/env python3
"""Run paper-aligned eval on holosoma v2 kinematic retarget outputs (E019 P1).

For each case present in ``adapters.kinematic_to_common.CASE_MAP``:
  1. Load the spider scene model corresponding to the matching E018b variant.
  2. Load holosoma's kinematic NPZ (qpos + human_joints + fps + cost).
  3. Compute physics-only paper metrics via
     ``paper_metrics.add_paper_metrics_physics``.
  4. Write per-case JSON / CSV under ``results/holosoma_v2_kinematic/`` and a
     combined ``comparison.csv`` consumable by ``unified_eval.py``.

Mapping (short case -> spider E018b variant -> holosoma NPZ file) lives in
``adapters/kinematic_to_common.py``. Only ``box025_p1`` / ``box025_p2`` are
covered until holosoma re-runs retarget on the remaining 11 spider cases.

Usage:
    .venv/bin/python workspace/core4d_collab_retarget/scripts/eval/\
        eval_holosoma_kinematic.py [--case box025_p2] [--all]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

THIS = Path(__file__).resolve()
EVAL_DIR = THIS.parent
sys.path.insert(0, str(EVAL_DIR))

import paper_metrics  # noqa: E402
from adapters import HOLOSOMA_V2_CASE_MAP, list_holosoma_v2_cases, load_kinematic_inputs  # noqa: E402

import eval_E002 as e002  # noqa: E402
import eval_E018b as e018b  # noqa: E402


REPO = THIS.parents[4]  # spider/
RESULTS = REPO / "workspace/core4d_collab_retarget/results/holosoma_v2_kinematic"

# Holosoma case (short) -> spider E018b variant (for scene model + meta)
SHORT_TO_E018B = {short: f"E018b_{short}_canonical_t02" for short in HOLOSOMA_V2_CASE_MAP}

E018B_MANIFEST = (
    REPO / "workspace/core4d_collab_retarget/results/E018b/manifest.tsv"
)


def _load_e018b_meta(spider_variant: str) -> dict[str, str]:
    e002.RESULTS = REPO / "workspace/core4d_collab_retarget/results/E018b"
    variants = e018b.read_manifest()
    if spider_variant not in variants:
        raise KeyError(f"spider variant {spider_variant} not in E018b manifest")
    return variants[spider_variant]


def _flatten_dict(d: dict[str, Any], parent: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{parent}.{k}" if parent else str(k)
        if isinstance(v, dict):
            out.update(_flatten_dict(v, key))
        elif isinstance(v, (list, tuple)):
            out[key] = json.dumps(v, ensure_ascii=False, default=str)
        else:
            out[key] = v
    return out


def evaluate_case(case: str) -> dict[str, Any]:
    if case not in SHORT_TO_E018B:
        raise KeyError(f"No spider E018b mapping for kinematic case '{case}'")
    spider_variant = SHORT_TO_E018B[case]
    meta = _load_e018b_meta(spider_variant)
    model, _scene = e002.load_scene_model(str(meta["case"]))

    eval_in = load_kinematic_inputs(case, model)
    metrics = paper_metrics.add_paper_metrics_physics(
        model=eval_in.model,
        qpos=eval_in.qpos_sim,
        fps=eval_in.fps,
        human_joints=eval_in.human_joints,
        case=case,
    )
    metrics["method"] = "holosoma_v2_kinematic"
    metrics["case"] = case  # short canonical
    metrics["variant"] = f"holosoma_v2_kinematic_{case}"
    metrics["source_npz"] = eval_in.extras.get("source_npz", "")
    metrics["companion_npz"] = eval_in.extras.get("companion_npz", "")
    metrics["fps"] = eval_in.fps
    metrics["cost"] = eval_in.extras.get("cost", float("nan"))
    # Mirror E018b columns expected by unified_eval (fall back values).
    metrics["E018b_diagnostic_class"] = "kinematic_baseline"
    metrics["E018b_robot_fall_detected"] = False
    metrics["case_window_pelvis_z_min_m"] = float(eval_in.qpos_sim[:, 2].min())
    metrics["paper_carry_progress_ratio_case"] = 1.0  # kin = ref by construction
    metrics["paper_transport_success"] = True
    metrics["paper_dynaretarget_object_success"] = True
    metrics["paper_spider_object_success"] = True
    return metrics


def write_outputs(results: list[dict[str, Any]]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    # Per-case JSON + CSV
    for row in results:
        case = row["case"]
        with (RESULTS / f"eval_summary_holosoma_v2_kinematic_{case}.json").open("w") as f:
            json.dump(row, f, indent=2, default=str)
        with (RESULTS / f"eval_summary_holosoma_v2_kinematic_{case}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
    # Combined comparison.csv
    if not results:
        return
    # Union of all keys
    keys = sorted({k for r in results for k in r.keys()}, key=str)
    with (RESULTS / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in keys})
    # Aggregate summary
    aggregate = {
        "num_results": len(results),
        "num_cases_available": len(list_holosoma_v2_cases()),
        "method": "holosoma_v2_kinematic",
        "mean_paper_omniretarget_mj_penetration_duration_pct": float(
            np.nanmean(
                [
                    r.get("paper_omniretarget_mj_penetration_duration_pct", float("nan"))
                    for r in results
                ]
            )
        ),
        "mean_paper_omniretarget_mj_penetration_max_depth_cm": float(
            np.nanmean(
                [
                    r.get("paper_omniretarget_mj_penetration_max_depth_cm", float("nan"))
                    for r in results
                ]
            )
        ),
        "mean_paper_omniretarget_contact_preservation_local_case_pct": float(
            np.nanmean(
                [
                    r.get(
                        "paper_omniretarget_contact_preservation_local_case_pct",
                        float("nan"),
                    )
                    for r in results
                ]
            )
        ),
        "variants": [r["case"] for r in results],
    }
    with (RESULTS / "aggregate_summary.json").open("w") as f:
        json.dump(aggregate, f, indent=2)
    print(json.dumps(aggregate, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--case", help="single short case name (e.g. box025_p2)")
    grp.add_argument("--all", action="store_true", help="evaluate every available case")
    args = ap.parse_args()

    if args.case:
        targets = [args.case]
    else:
        targets = list_holosoma_v2_cases()
        if not targets:
            print("[eval_holosoma_kinematic] no available cases in CASE_MAP", file=sys.stderr)
            return 1

    results: list[dict[str, Any]] = []
    for case in targets:
        print(f"[eval_holosoma_kinematic] case={case}")
        row = evaluate_case(case)
        results.append(row)
    write_outputs(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
