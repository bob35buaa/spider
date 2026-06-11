#!/usr/bin/env python3
"""Evaluate E153 CEM hand-gate threshold sweep (gateA_b1, 3 cases x 3 min_sdf x 2 max_viol).

Per SKILL.md §13: uses lib.core_metrics directly (no importlib of other evaluators).
Compares each grid point vs the reward-only b1 reference (E151, gate off), and reports
the depth-aware success criterion + gate health. Baseline (E147/E148 rubber, gate off)
is also evaluated for context.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.core_metrics import (  # noqa: E402
    EvalConfig,
    EVAL_METRIC_STANDARD_ID,
    STANDARD_DELTA_METRICS,
    STANDARD_MASK_DELTA_METRICS,
    STANDARD_TRACK_DIAG,
    contact_mask_for_case,
    evaluate_sequence,
    kin_ref_for_scene,
    person_idx_from_case,
)

REPO = Path(__file__).resolve().parents[5]
RESULT_ROOT = REPO / "workspace/core4d/results/E153/gate_threshold_sweep"
HARD_FLOOR_M = -0.020

MIN_SDF_LIST = [-0.005, -0.010, -0.015]
MAX_VIOL_LIST = [0.05, 0.10]

# Per-case reuse references (gate-off, unaffected by Stage0). Paths relative to REPO.
CASES: dict[str, dict[str, str]] = {
    "box021_029_p2": {
        "object_key": "box021",
        "scene": "example_datasets/processed/core4d/unitree_g1/humanoid_object/d003_box021_20231018_029_p2_e107_clean/scene_act_E148_rubber_hull.xml",
        "baseline": "workspace/core4d/results/E148/e143_24case_rubber_hand_collision/cem/full/E148_box021_029_p2_rubber_hull_outdir_full/trajectory_mjwp_act.npz",
        "b1": "workspace/core4d/results/E151/route_b_hand_surface_contact/cem/full/E151_box021_029_p2_b1_mesh_outdir_full/trajectory_mjwp_act.npz",
    },
    "box004_083_p2": {
        "object_key": "box004",
        "scene": "example_datasets/processed/core4d/unitree_g1/humanoid_object/e091_box004_20231003_2_083_p2_e092_dyn/scene_act_E148_rubber_hull.xml",
        "baseline": "workspace/core4d/results/E148/e143_24case_rubber_hand_collision/cem/full/E148_box004_083_p2_rubber_hull_outdir_full/trajectory_mjwp_act.npz",
        "b1": "workspace/core4d/results/E151/route_b_hand_surface_contact/cem/full/E151_box004_083_p2_b1_mesh_outdir_full/trajectory_mjwp_act.npz",
    },
    "box023_person2": {
        "object_key": "box023",
        "scene": "example_datasets/processed/core4d/unitree_g1/humanoid_object/box023_person2_legobj/scene_act_E147_rubber_hull.xml",
        "baseline": "workspace/core4d/results/E147/rubber_hand_collision/cem/full/E147_box023_person2_rubber_hull_outdir_full/trajectory_mjwp_act.npz",
        "b1": "workspace/core4d/results/E151/route_b_hand_surface_contact/cem/full/E151_box023_person2_b1_mesh_outdir_full/trajectory_mjwp_act.npz",
    },
}

# Metrics carried into per-row TSV + deltas.
DELTA_METRICS = list(STANDARD_DELTA_METRICS)
GATE_HEALTH_KEYS = {
    "cem_hand_gate_valid_frac": ("mean", "hand_gate_valid_frac"),
    "cem_gate_fallback_used": ("mean", "gate_fallback_used"),
    "cem_hand_gate_min_sdf_min": ("min", "hand_gate_min_sdf_min_m"),
    "sample_hand_gate_violation_pct_mean": ("mean", "hand_gate_violation_pct"),
}

# E154 absolute diagnostics carried per grid row (tracking vs fixed kin truth +
# real-3cm masked contact). Tracking gates success; false-contact is diagnostic.
TRACK_DIAG = list(STANDARD_TRACK_DIAG)
# masked-contact metrics that also get a delta-vs-b1 (the legit replacements for
# the full-sequence physC / pen deltas)
MASK_DELTA = list(STANDARD_MASK_DELTA_METRICS)


def sdf_tag(x: float) -> str:
    return f"sdf{int(round(-x * 1000)):03d}"


def viol_tag(x: float) -> str:
    return f"v{int(round(x * 100)):02d}"


def repo_path(text: str) -> Path:
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


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r.get(k, "")) for k in fields})


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


def evaluate(case: str, variant: str, method: str, npz_rel: str, scene_rel: str, cfg: EvalConfig) -> dict[str, Any] | None:
    npz = repo_path(npz_rel)
    scene = repo_path(scene_rel)
    if not npz.is_file() or not scene.is_file():
        return None
    row = {
        "case_id": case,
        "variant": variant,
        "object_key": CASES[case]["object_key"],
        "object_category": "box",
        "expected_quality": "review",
    }
    m = evaluate_sequence(row=row, method=method, hand_collision_variant_id="rubber_hull",
                          qpos_path=npz, scene_xml=scene, config=cfg,
                          kin_ref_path=kin_ref_for_scene(scene),
                          contact_mask_path=contact_mask_for_case(case),
                          person_idx=person_idx_from_case(case))
    return m


def agg(vals: list[float]) -> tuple[float, float, float]:
    v = [x for x in vals if x is not None and math.isfinite(x)]
    if not v:
        return math.nan, math.nan, math.nan
    mean = statistics.fmean(v)
    std = statistics.pstdev(v) if len(v) > 1 else 0.0
    return mean, std, max(v)  # worst defined per-metric by caller orientation


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", default="full", choices=["smoke", "full"])
    args = ap.parse_args()
    stage = args.stage
    cem_dir = RESULT_ROOT / "cem" / stage
    eval_dir = RESULT_ROOT / "eval" / stage
    cfg = EvalConfig()

    metric_rows: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for case, ref in CASES.items():
        # references
        ref_metrics: dict[str, dict[str, Any]] = {}
        for mlabel, key in (("baseline", "baseline"), ("b1", "b1")):
            mm = evaluate(case, f"E153_{case}_{mlabel}", mlabel, ref[key], ref["scene"], cfg)
            if mm is None:
                missing.append(f"{case}:{mlabel}")
                continue
            ref_metrics[mlabel] = mm
            metric_rows.append({**mm, "min_sdf_m": "", "max_viol": "", "hard_floor_m": ""})
        b1 = ref_metrics.get("b1")

        # grid
        for min_sdf, max_viol in itertools.product(MIN_SDF_LIST, MAX_VIOL_LIST):
            variant = f"E153_{case}_gateA_b1_{sdf_tag(min_sdf)}_{viol_tag(max_viol)}"
            npz_rel = f"{cem_dir.relative_to(REPO)}/{variant}_outdir_{stage}/trajectory_mjwp_act.npz"
            mm = evaluate(case, variant, "gateA_b1", npz_rel, ref["scene"], cfg)
            if mm is None:
                missing.append(variant)
                continue
            health = gate_health(repo_path(npz_rel))
            mm = {**mm, "min_sdf_m": min_sdf, "max_viol": max_viol, "hard_floor_m": HARD_FLOOR_M, **health}
            metric_rows.append(mm)

            if b1 is None:
                continue
            d = {
                "case_id": case, "variant": variant, "min_sdf_m": min_sdf, "max_viol": max_viol,
                "hard_floor_m": HARD_FLOOR_M, "fall_flag": mm["fall_flag"],
                **{k: health.get(k, "") for k in ("hand_gate_valid_frac", "gate_fallback_used", "hand_gate_min_sdf_min_m", "hand_gate_violation_pct")},
            }
            for k in DELTA_METRICS:
                d[f"{k}_b1"] = b1.get(k, math.nan)
                d[f"{k}_run"] = mm.get(k, math.nan)
                d[f"{k}_delta"] = float(mm.get(k, math.nan)) - float(b1.get(k, math.nan))
            # depth-aware success vs b1: real(>2mm) penetration not up, contact kept, tracking ok, no fall
            d["success_pen2mm"] = bool(
                d["hand_geom_penetration_2mm_frac_delta"] <= 0.0
                and d["hand_geom_near_5cm_frac_delta"] >= -0.02
                and d["obj_err_mean_m_delta"] <= 0.02
                and not bool(mm["fall_flag"])
            )
            # legacy 0mm success (for comparison)
            d["success_pen0mm"] = bool(
                d["hand_geom_penetration_frac_delta"] <= 0.0
                and d["hand_geom_near_5cm_frac_delta"] >= -0.02
                and d["obj_err_mean_m_delta"] <= 0.02
                and not bool(mm["fall_flag"])
            )
            # E154 absolute diagnostics + masked-contact deltas vs b1
            for k in TRACK_DIAG:
                d[k] = mm.get(k, math.nan)
            for k in MASK_DELTA:
                d[f"{k}_b1"] = b1.get(k, math.nan)
                d[f"{k}_delta"] = float(mm.get(k, math.nan)) - float(b1.get(k, math.nan))
            # E154 tracking-gated success (user-confirmed): pen2mm verdict AND the
            # robot completes the standup (terminal pelvis-z tracking vs fixed truth).
            pz_term = mm.get("track_pelvis_z_err_terminal_m", math.inf)
            d["success_tracked"] = bool(
                d["success_pen2mm"]
                and math.isfinite(pz_term)
                and pz_term <= cfg.track_pelvis_terminal_th_m
            )
            grid_rows.append(d)

    # ---- per-combo 3-case aggregate ----
    combo_rows: list[dict[str, Any]] = []
    for min_sdf, max_viol in itertools.product(MIN_SDF_LIST, MAX_VIOL_LIST):
        rs = [r for r in grid_rows if r["min_sdf_m"] == min_sdf and r["max_viol"] == max_viol]
        if not rs:
            continue
        item: dict[str, Any] = {"min_sdf_m": min_sdf, "max_viol": max_viol, "n_cases": len(rs),
                                "success_tracked_cases": sum(1 for r in rs if r.get("success_tracked")),
                                "success_pen2mm_cases": sum(1 for r in rs if r["success_pen2mm"]),
                                "success_pen0mm_cases": sum(1 for r in rs if r["success_pen0mm"]),
                                "fall_cases": sum(1 for r in rs if r["fall_flag"])}
        for k in DELTA_METRICS:
            mean, std, _ = agg([r[f"{k}_delta"] for r in rs])
            # worst = least-favorable: for near5cm higher better -> worst=min; for the rest (penetration/err) lower better -> worst=max
            vals = [r[f"{k}_delta"] for r in rs if math.isfinite(r[f"{k}_delta"])]
            worst = (min(vals) if k == "hand_geom_near_5cm_frac" else max(vals)) if vals else math.nan
            item[f"{k}_mean"] = mean
            item[f"{k}_std"] = std
            item[f"{k}_worst"] = worst
        # E154 diagnostics: tracking (worst=max err) + release_false + in-mask contact
        for k in ("track_pelvis_z_err_terminal_m",
                  "hand_object_release_false_contact_3mm_frac", "hand_object_release_false_contact_5mm_frac",
                  "hand_object_physics_contact_3mm_in_mask_frac", "hand_object_physics_contact_5mm_in_mask_frac",
                  "hand_object_false_contact_3mm_frac", "hand_object_false_contact_5mm_frac",
                  "hand_object_clean_release_false_contact_frac",
                  "hand_object_clean_physics_contact_in_mask_frac", "hand_object_clean_false_contact_frac",
                  "hand_object_release_false_contact_frac", "hand_object_physics_contact_in_mask_frac",
                  "hand_object_false_contact_frac"):
            vals = [float(r[k]) for r in rs if r.get(k) not in ("", None) and math.isfinite(float(r[k]))]
            item[f"{k}_mean"] = statistics.fmean(vals) if vals else math.nan
            item[f"{k}_worst"] = max(vals) if vals else math.nan
        for k in ("hand_gate_valid_frac", "gate_fallback_used"):
            vals = [float(r[k]) for r in rs if r.get(k) not in ("", None) and math.isfinite(float(r[k]))]
            item[f"{k}_mean"] = statistics.fmean(vals) if vals else math.nan
        item["hand_gate_min_sdf_min_m_worst"] = min(
            [float(r["hand_gate_min_sdf_min_m"]) for r in rs if r.get("hand_gate_min_sdf_min_m") not in ("", None)],
            default=math.nan)
        combo_rows.append(item)

    # ---- write outputs ----
    eval_dir.mkdir(parents=True, exist_ok=True)
    metric_fields = (["case_id", "variant", "method", "min_sdf_m", "max_viol", "hard_floor_m"]
                     + [m for m in (
                         "qpos_frames", "pelvis_min_m", "fall_flag",
                         "hand_geom_near_5cm_frac", "hand_geom_near_10cm_frac",
                         "hand_geom_penetration_frac", "hand_geom_penetration_2mm_frac",
                         "hand_geom_penetration_5mm_frac", "hand_geom_deep_penetration_2cm_frac",
                         "hand_object_physics_contact_frac", "hand_object_clean_physics_contact_frac",
                         "hand_object_physics_contact_3mm_frac", "hand_object_physics_contact_5mm_frac",
                         "hand_object_physics_penetration_3mm_frame_frac",
                         "hand_object_physics_penetration_5mm_frame_frac",
                         "hand_object_con_dist_mean_m", "hand_object_con_dist_min_m",
                         "hand_object_con_dist_frac_lt_neg2mm", "hand_object_con_dist_frac_lt_neg3mm",
                         "hand_object_con_dist_frac_lt_neg5mm",
                         "leg_penetration_frac", "body_penetration_frac", "obj_err_mean_m")]
                     + TRACK_DIAG
                     + [dst for _, (_, dst) in GATE_HEALTH_KEYS.items()])
    write_tsv(eval_dir / "e153_method_metrics.tsv", metric_rows, metric_fields)

    grid_fields = (["case_id", "variant", "min_sdf_m", "max_viol", "hard_floor_m", "fall_flag",
                    "success_tracked", "success_pen2mm", "success_pen0mm",
                    "hand_gate_valid_frac", "gate_fallback_used", "hand_gate_min_sdf_min_m", "hand_gate_violation_pct"]
                   + TRACK_DIAG
                   + [f"{k}_{s}" for k in MASK_DELTA for s in ("b1", "delta")]
                   + [f"{k}_{s}" for k in DELTA_METRICS for s in ("b1", "run", "delta")])
    write_tsv(eval_dir / "e153_grid_delta_vs_b1.tsv", grid_rows, grid_fields)

    combo_fields = (["min_sdf_m", "max_viol", "n_cases", "success_tracked_cases",
                     "success_pen2mm_cases", "success_pen0mm_cases", "fall_cases",
                     "track_pelvis_z_err_terminal_m_mean", "track_pelvis_z_err_terminal_m_worst",
                     "hand_object_release_false_contact_3mm_frac_mean", "hand_object_release_false_contact_3mm_frac_worst",
                     "hand_object_release_false_contact_5mm_frac_mean", "hand_object_release_false_contact_5mm_frac_worst",
                     "hand_object_false_contact_3mm_frac_mean", "hand_object_false_contact_3mm_frac_worst",
                     "hand_object_false_contact_5mm_frac_mean", "hand_object_false_contact_5mm_frac_worst",
                     "hand_object_physics_contact_3mm_in_mask_frac_mean", "hand_object_physics_contact_3mm_in_mask_frac_worst",
                     "hand_object_physics_contact_5mm_in_mask_frac_mean", "hand_object_physics_contact_5mm_in_mask_frac_worst",
                     "hand_object_clean_release_false_contact_frac_mean", "hand_object_clean_release_false_contact_frac_worst",
                     "hand_object_clean_false_contact_frac_mean", "hand_object_clean_false_contact_frac_worst",
                     "hand_object_clean_physics_contact_in_mask_frac_mean", "hand_object_clean_physics_contact_in_mask_frac_worst",
                     "hand_object_release_false_contact_frac_mean", "hand_object_release_false_contact_frac_worst",
                     "hand_object_false_contact_frac_mean", "hand_object_false_contact_frac_worst",
                     "hand_object_physics_contact_in_mask_frac_mean", "hand_object_physics_contact_in_mask_frac_worst",
                     "hand_gate_valid_frac_mean", "gate_fallback_used_mean", "hand_gate_min_sdf_min_m_worst"]
                    + [f"{k}_{s}" for k in DELTA_METRICS for s in ("mean", "std", "worst")])
    write_tsv(eval_dir / "e153_combo_summary.tsv", combo_rows, combo_fields)

    write_json(eval_dir / "e153_eval_summary.json", {
        "metric_standard_id": EVAL_METRIC_STANDARD_ID,
        "stage": stage, "metric_rows": len(metric_rows), "grid_rows": len(grid_rows),
        "combos": len(combo_rows), "missing": missing,
    })
    print(f"E153 eval: metric_rows={len(metric_rows)} grid_rows={len(grid_rows)} combos={len(combo_rows)} missing={len(missing)}")
    if missing:
        print("MISSING:", missing)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
