#!/usr/bin/env python3
"""Evaluate E155 release-smoothing variants with the E154+ metric standard.

E155 runs four variants (ramp5/ramp10/decay/neutral) on the three E153
selected cases. The incremental reference is E153 gateA_b1 at
min_sdf=-0.010, max_viol=0.10.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval.core.core_metrics import (  # noqa: E402
    EVAL_METRIC_STANDARD_ID,
    EvalConfig,
    STANDARD_DELTA_METRICS,
    STANDARD_MASK_DELTA_METRICS,
    STANDARD_TRACK_DIAG,
    contact_mask_for_case,
    evaluate_sequence,
    kin_ref_for_scene,
    person_idx_from_case,
)

REPO = Path(__file__).resolve().parents[5]
RESULT_ROOT = REPO / "workspace/core4d/results/E155"
E153_ROOT = REPO / "workspace/core4d/results/E153/gate_threshold_sweep"
HARD_FLOOR_M = -0.020
MIN_SDF_M = -0.010
MAX_VIOL = 0.10

METHODS = ["ramp5", "ramp10", "decay", "neutral"]

CASES: dict[str, dict[str, str]] = {
    "box021_029_p2": {
        "object_key": "box021",
        "scene": "example_datasets/processed/core4d/unitree_g1/humanoid_object/d003_box021_20231018_029_p2_e107_clean/scene_act_E148_rubber_hull.xml",
    },
    "box004_083_p2": {
        "object_key": "box004",
        "scene": "example_datasets/processed/core4d/unitree_g1/humanoid_object/e091_box004_20231003_2_083_p2_e092_dyn/scene_act_E148_rubber_hull.xml",
    },
    "box023_person2": {
        "object_key": "box023",
        "scene": "example_datasets/processed/core4d/unitree_g1/humanoid_object/box023_person2_legobj/scene_act_E147_rubber_hull.xml",
    },
}

DELTA_METRICS = list(STANDARD_DELTA_METRICS)
TRACK_DIAG = list(STANDARD_TRACK_DIAG)
MASK_DELTA = list(STANDARD_MASK_DELTA_METRICS)

GATE_HEALTH_KEYS = {
    "cem_hand_gate_valid_frac": ("mean", "hand_gate_valid_frac"),
    "cem_gate_fallback_used": ("mean", "gate_fallback_used"),
    "cem_hand_gate_min_sdf_min": ("min", "hand_gate_min_sdf_min_m"),
    "sample_hand_gate_violation_pct_mean": ("mean", "hand_gate_violation_pct"),
}


def repo_path(text: str | Path) -> Path:
    p = Path(text)
    return p if p.is_absolute() else REPO / p


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}" if math.isfinite(value) else ""
    return str(value)


def finite(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({k: fmt(row.get(k, "")) for k in fields})


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def gate_health(outdir_npz: Path) -> dict[str, Any]:
    out: dict[str, Any] = {dst: "" for _, (_, dst) in GATE_HEALTH_KEYS.items()}
    if not outdir_npz.is_file():
        return out
    data = np.load(outdir_npz, allow_pickle=True)
    for key, (agg, dst) in GATE_HEALTH_KEYS.items():
        if key not in data.files:
            continue
        arr = np.asarray(data[key], dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if not arr.size:
            continue
        out[dst] = float(arr.mean()) if agg == "mean" else float(arr.min())
    return out


def evaluate(case: str, variant: str, method: str, qpos_path: Path, scene_rel: str, cfg: EvalConfig) -> dict[str, Any] | None:
    scene = repo_path(scene_rel)
    if not qpos_path.is_file() or not scene.is_file():
        return None
    row = {
        "case_id": case,
        "variant": variant,
        "object_key": CASES[case]["object_key"],
        "object_category": "box",
        "expected_quality": "review",
    }
    return evaluate_sequence(
        row=row,
        method=method,
        hand_collision_variant_id="rubber_hull",
        qpos_path=qpos_path,
        scene_xml=scene,
        config=cfg,
        kin_ref_path=kin_ref_for_scene(scene),
        contact_mask_path=contact_mask_for_case(case),
        person_idx=person_idx_from_case(case),
    )


def e153_ref_qpos(case: str, stage: str) -> Path:
    variant = f"E153_{case}_gateA_b1_sdf010_v10"
    return E153_ROOT / "cem" / stage / f"{variant}_outdir_{stage}" / "trajectory_mjwp_act.npz"


def e155_qpos(case: str, method: str, stage: str) -> Path:
    variant = f"E155_{case}_{method}"
    return RESULT_ROOT / "cem" / stage / f"{variant}_outdir_{stage}" / "trajectory_mjwp_act.npz"


def add_success_flags(row: dict[str, Any], cfg: EvalConfig) -> None:
    pz_term = finite(row.get("track_pelvis_z_err_terminal_m"), math.inf)
    row["success_tracked"] = bool(
        not bool(row.get("fall_flag"))
        and math.isfinite(pz_term)
        and pz_term <= cfg.track_pelvis_terminal_th_m
    )


def build_delta_row(case: str, method: str, run: dict[str, Any], ref: dict[str, Any], cfg: EvalConfig) -> dict[str, Any]:
    variant = f"E155_{case}_{method}"
    row: dict[str, Any] = {
        "case_id": case,
        "variant": variant,
        "method": method,
        "reference_variant": f"E153_{case}_gateA_b1_sdf010_v10",
        "min_sdf_m": MIN_SDF_M,
        "max_viol": MAX_VIOL,
        "hard_floor_m": HARD_FLOOR_M,
        "fall_flag": run.get("fall_flag", ""),
        "success_tracked": run.get("success_tracked", False),
    }
    for key in ("hand_gate_valid_frac", "gate_fallback_used", "hand_gate_min_sdf_min_m", "hand_gate_violation_pct"):
        row[key] = run.get(key, "")
    for key in TRACK_DIAG:
        row[key] = run.get(key, math.nan)
    for key in MASK_DELTA:
        row[f"{key}_ref"] = ref.get(key, math.nan)
        row[f"{key}_run"] = run.get(key, math.nan)
        row[f"{key}_delta"] = finite(run.get(key)) - finite(ref.get(key))
    for key in DELTA_METRICS:
        row[f"{key}_ref"] = ref.get(key, math.nan)
        row[f"{key}_run"] = run.get(key, math.nan)
        row[f"{key}_delta"] = finite(run.get(key)) - finite(ref.get(key))

    row["success_guarded_vs_ref"] = bool(
        row["success_tracked"]
        and row["hand_geom_penetration_2mm_frac_delta"] <= 0.0
        and row["hand_geom_near_5cm_frac_delta"] >= -0.02
        and row["obj_err_mean_m_delta"] <= 0.02
    )
    return row


def mean(values: list[Any]) -> float:
    vals = [finite(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return statistics.fmean(vals) if vals else math.nan


def worst(values: list[Any], high_is_bad: bool = True) -> float:
    vals = [finite(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return math.nan
    return max(vals) if high_is_bad else min(vals)


def summarize_method(method: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "method": method,
        "n_cases": len(rows),
        "success_tracked_cases": sum(1 for r in rows if r.get("success_tracked")),
        "success_guarded_vs_ref_cases": sum(1 for r in rows if r.get("success_guarded_vs_ref")),
        "fall_cases": sum(1 for r in rows if r.get("fall_flag")),
    }
    summary_keys = [
        "track_pelvis_z_err_terminal_m",
        "hand_object_physics_contact_in_mask_frac",
        "hand_object_physics_contact_3mm_in_mask_frac",
        "hand_object_physics_contact_5mm_in_mask_frac",
        "hand_object_release_false_contact_frac",
        "hand_object_release_false_contact_3mm_frac",
        "hand_object_release_false_contact_5mm_frac",
        "hand_object_false_contact_3mm_frac",
        "hand_object_false_contact_5mm_frac",
        "hand_geom_penetration_2mm_frac",
        "hand_geom_penetration_5mm_frac",
        "hand_object_physics_penetration_3mm_frame_frac",
        "hand_object_physics_penetration_5mm_frame_frac",
        "leg_penetration_frac",
        "obj_err_mean_m",
        "hand_gate_valid_frac",
        "gate_fallback_used",
    ]
    for key in summary_keys:
        source = f"{key}_run" if any(f"{key}_run" in r for r in rows) else key
        out[f"{key}_mean"] = mean([r.get(source) for r in rows])
        out[f"{key}_worst"] = worst(
            [r.get(source) for r in rows],
            high_is_bad=key not in {
                "hand_object_physics_contact_in_mask_frac",
                "hand_object_physics_contact_3mm_in_mask_frac",
                "hand_object_physics_contact_5mm_in_mask_frac",
                "hand_gate_valid_frac",
            },
        )
    delta_keys = [
        "hand_object_physics_contact_in_mask_frac",
        "hand_object_physics_contact_3mm_in_mask_frac",
        "hand_object_physics_contact_5mm_in_mask_frac",
        "hand_object_release_false_contact_frac",
        "hand_object_release_false_contact_3mm_frac",
        "hand_object_release_false_contact_5mm_frac",
        "hand_geom_penetration_2mm_frac",
        "hand_object_physics_penetration_3mm_frame_frac",
        "obj_err_mean_m",
    ]
    for key in delta_keys:
        dkey = f"{key}_delta"
        out[f"{dkey}_mean"] = mean([r.get(dkey) for r in rows])
        out[f"{dkey}_worst"] = worst(
            [r.get(dkey) for r in rows],
            high_is_bad=key not in {
                "hand_object_physics_contact_in_mask_frac",
                "hand_object_physics_contact_3mm_in_mask_frac",
                "hand_object_physics_contact_5mm_in_mask_frac",
            },
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", default="full", choices=["smoke", "full"])
    ap.add_argument("--allow-missing", action="store_true")
    args = ap.parse_args()

    cfg = EvalConfig()
    stage = args.stage
    eval_dir = RESULT_ROOT / "eval" / stage

    metric_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    ref_by_case: dict[str, dict[str, Any]] = {}
    missing: list[str] = []

    for case, meta in CASES.items():
        ref_variant = f"E153_{case}_gateA_b1_sdf010_v10"
        ref = evaluate(case, ref_variant, "e153_gateA_b1_sdf010_v10", e153_ref_qpos(case, stage), meta["scene"], cfg)
        if ref is None:
            missing.append(ref_variant)
            continue
        add_success_flags(ref, cfg)
        ref_by_case[case] = ref
        metric_rows.append({**ref, "method_group": "reference", "min_sdf_m": MIN_SDF_M, "max_viol": MAX_VIOL, "hard_floor_m": HARD_FLOOR_M})

    for case, method in itertools.product(CASES, METHODS):
        variant = f"E155_{case}_{method}"
        qpos = e155_qpos(case, method, stage)
        run = evaluate(case, variant, method, qpos, CASES[case]["scene"], cfg)
        if run is None:
            missing.append(variant)
            continue
        add_success_flags(run, cfg)
        run = {
            **run,
            "method_group": "e155",
            "min_sdf_m": MIN_SDF_M,
            "max_viol": MAX_VIOL,
            "hard_floor_m": HARD_FLOOR_M,
            **gate_health(qpos),
        }
        metric_rows.append(run)
        if case in ref_by_case:
            delta_rows.append(build_delta_row(case, method, run, ref_by_case[case], cfg))

    method_rows = [summarize_method(method, [r for r in delta_rows if r["method"] == method]) for method in METHODS]
    ref_summary = summarize_method("e153_gateA_b1_sdf010_v10", [
        {**r, **{f"{k}_run": r.get(k) for k in set(STANDARD_DELTA_METRICS + STANDARD_TRACK_DIAG + STANDARD_MASK_DELTA_METRICS)}}
        for r in metric_rows
        if r.get("method_group") == "reference"
    ])

    metric_fields = (
        ["case_id", "variant", "method", "method_group", "min_sdf_m", "max_viol", "hard_floor_m"]
        + [
            "qpos_frames",
            "pelvis_min_m",
            "fall_flag",
            "success_tracked",
            "hand_geom_near_5cm_frac",
            "hand_geom_near_10cm_frac",
            "hand_geom_penetration_frac",
            "hand_geom_penetration_2mm_frac",
            "hand_geom_penetration_5mm_frac",
            "hand_object_physics_contact_frac",
            "hand_object_physics_contact_3mm_frac",
            "hand_object_physics_contact_5mm_frac",
            "hand_object_physics_penetration_3mm_frame_frac",
            "hand_object_physics_penetration_5mm_frame_frac",
            "hand_object_con_dist_min_m",
            "leg_penetration_frac",
            "body_penetration_frac",
            "obj_err_mean_m",
        ]
        + TRACK_DIAG
        + [dst for _, (_, dst) in GATE_HEALTH_KEYS.items()]
    )
    delta_fields = (
        [
            "case_id",
            "variant",
            "method",
            "reference_variant",
            "min_sdf_m",
            "max_viol",
            "hard_floor_m",
            "fall_flag",
            "success_tracked",
            "success_guarded_vs_ref",
            "hand_gate_valid_frac",
            "gate_fallback_used",
            "hand_gate_min_sdf_min_m",
            "hand_gate_violation_pct",
        ]
        + TRACK_DIAG
        + [f"{k}_{s}" for k in MASK_DELTA for s in ("ref", "run", "delta")]
        + [f"{k}_{s}" for k in DELTA_METRICS for s in ("ref", "run", "delta")]
    )
    summary_fields = list(method_rows[0].keys()) if method_rows else ["method", "n_cases"]

    write_tsv(eval_dir / "e155_method_metrics.tsv", metric_rows, metric_fields)
    write_tsv(eval_dir / "e155_delta_vs_e153_sdf010_v10.tsv", delta_rows, delta_fields)
    write_tsv(eval_dir / "e155_method_summary.tsv", [ref_summary] + method_rows, summary_fields)
    write_json(
        eval_dir / "e155_eval_summary.json",
        {
            "metric_standard_id": EVAL_METRIC_STANDARD_ID,
            "stage": stage,
            "reference": "E153_gateA_b1_sdf010_v10",
            "metric_rows": len(metric_rows),
            "delta_rows": len(delta_rows),
            "method_rows": len(method_rows),
            "missing": missing,
            "allow_missing": bool(args.allow_missing),
        },
    )
    print(f"E155 eval: metric_rows={len(metric_rows)} delta_rows={len(delta_rows)} methods={len(method_rows)} missing={len(missing)}")
    if missing:
        print("MISSING:", missing)
        if not args.allow_missing:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
