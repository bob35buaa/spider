#!/usr/bin/env python3
"""E214c eval: per-part-penetration metric set for the part-wise threshold ablation.

Three series, ALL evaluated with the SAME current evaluator (rule 13: shared
eval.core.core_metrics + motion_health + fixed_reference_z_metrics):
  full             each case's baseline rollout npz (recomputed fresh here)
  A5_uniform_hand  body/leg gates <- hand thresholds
  A6_uniform_leg   body/hand gates <- leg thresholds

The "full" column is recomputed (not reused from the paper cache) because the
part-wise claims (C1/C2) need per-part penetration (leg/body/upper_body), which the
paper cache never stored -- and recomputing full with the identical evaluator makes
full|A5|A6 internally consistent (same code version, same metric definitions).

Outputs (results/E214c/eval/):
  e214c_metrics.jsonl   one record per (case, series)
  e214c_metrics.tsv     flat table

Usage:
    .venv/bin/python workspace/core4d/scripts/eval/runners/eval_E214c_ablation.py
    ... --series full,A5_uniform_hand   # subset
    ... --fresh                         # ignore cache
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "workspace/core4d/scripts"))
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/experiments/E214c"))

import e214c_common as C  # noqa: E402
from eval.core.core_metrics import (  # noqa: E402
    EvalConfig,
    evaluate_sequence,
    person_idx_from_case,
)
from eval.core.motion_health import run_health  # noqa: E402
from eval.runners.eval_E194_G1_expansion import fixed_reference_z_metrics  # noqa: E402

HAND_COLLISION_VARIANT = "rubber_hull"

# Metric set = E214's standard 15-ish (for 6-arm comparability) + PER-PART penetration
# (the part-wise story: leg/body for A5, hand for A6).
METRICS: list[dict[str, Any]] = [
    # --- contact (hand-object) ---
    {"key": "raw_contact",    "field": "hand_object_physics_contact_in_mask_frac",      "src": "result"},
    {"key": "contact_3mm",    "field": "hand_object_physics_contact_3mm_in_mask_frac",  "src": "result"},
    {"key": "contact_5mm",    "field": "hand_object_physics_contact_5mm_in_mask_frac",  "src": "result"},
    {"key": "contact_10mm",   "field": "hand_object_physics_contact_10mm_in_mask_frac", "src": "result"},
    # --- hand-object penetration ---
    {"key": "pen_3mm",        "field": "hand_object_physics_penetration_3mm_frame_frac","src": "result"},
    {"key": "pen_5mm",        "field": "hand_object_physics_penetration_5mm_frame_frac","src": "result"},
    {"key": "pen_10mm",       "field": "hand_object_physics_penetration_10mm_frame_frac","src": "result"},
    {"key": "pen_max_mm",     "field": "hand_object_physics_penetration_max_mm",        "src": "result"},
    {"key": "geom_2mm",       "field": "hand_geom_penetration_2mm_frac",                "src": "result"},
    # --- PER-PART penetration (NEW for E214c) ---
    {"key": "hand_geom_pen_2mm", "field": "hand_geom_penetration_2mm_frac",             "src": "result"},
    {"key": "leg_pen",        "field": "leg_penetration_frac",                          "src": "result"},
    {"key": "body_pen",       "field": "body_penetration_frac",                         "src": "result"},
    {"key": "upper_body_pen", "field": "upper_body_penetration_frac",                   "src": "result"},
    {"key": "leg_contact",    "field": "leg_object_physics_contact_frac",               "src": "result"},
    # --- tracking ---
    {"key": "track_root_pos", "field": "track_root_pos_err_cm_mean",                    "src": "result"},
    {"key": "track_root_ori", "field": "track_root_ori_err_deg_mean",                   "src": "result"},
    {"key": "track_eef_pos",  "field": "track_eef_pos_err_cm_mean",                     "src": "result"},
    {"key": "track_eef_ori",  "field": "track_eef_ori_err_deg_mean",                    "src": "result"},
    {"key": "track_obj_pos",  "field": "track_obj_pos_err_cm_mean",                     "src": "result"},
    {"key": "track_obj_ori",  "field": "track_obj_ori_err_deg_mean",                    "src": "result"},
    {"key": "fall_flag",      "field": "fall_flag",                                     "src": "result"},
    {"key": "body_z",         "field": "body_z_err_p95_m",                              "src": "zmetric"},
    {"key": "ankle_jerk",     "field": "ankle_jerk_p95",                                "src": "health"},
    {"key": "obj_speed",      "field": "obj_speed_max",                                 "src": "health"},
    {"key": "foot_slip",      "field": "foot_slip_max_m",                               "src": "health"},
    {"key": "foot_skate_mean","field": "foot_skate_speed_mean_m_s",                     "src": "health"},
    {"key": "foot_skate_max", "field": "foot_skate_speed_max_m_s",                      "src": "health"},
]
KEYS = [m["key"] for m in METRICS]
CACHE = C.EVAL_DIR / "e214c_metrics.jsonl"
OUT_TSV = C.EVAL_DIR / "e214c_metrics.tsv"


def _num(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return math.nan
    return f if math.isfinite(f) else math.nan


def eval_one(case_id: str, qpos: Path, scene: Path, kin_ref: Path | None,
             mask: Path | None, method: str) -> dict[str, float]:
    row = {"case_id": case_id, "variant": f"e214c_{case_id}",
           "object_key": C.object_key_of(case_id), "object_category": "",
           "expected_quality": "e214c"}
    result = evaluate_sequence(
        row=row, method=method, hand_collision_variant_id=HAND_COLLISION_VARIANT,
        qpos_path=qpos, scene_xml=scene, kin_ref_path=kin_ref,
        contact_mask_path=mask, person_idx=person_idx_from_case(case_id),
    )
    health = run_health(qpos, scene, EvalConfig())
    z = fixed_reference_z_metrics(qpos, scene, kin_ref) if kin_ref is not None else {}
    out: dict[str, float] = {}
    for m in METRICS:
        if m["src"] == "result":
            out[m["key"]] = _num(result.get(m["field"]))
        elif m["src"] == "health":
            out[m["key"]] = _num(health.get(m["field"]))
        elif m["src"] == "zmetric":
            out[m["key"]] = _num(z.get(m["field"]))
    return out


def load_cache() -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    if CACHE.is_file():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[(r["case_id"], r["series"])] = r
    return out


def append_cache(rec: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def jobs_from_manifest() -> list[dict[str, Any]]:
    """One eval job per cem_ok ablation row (A5/A6) + one per case's baseline (full)."""
    rows, _ = C.read_with_fields(C.MANIFEST)
    jobs: list[dict[str, Any]] = []
    # full: baseline rollout npz, recomputed fresh with the same evaluator
    for case in C.load_cases():
        bp = C.baseline_paths(case)
        ri = bp.get("run_inputs", {})
        if bp.get("cem_npz") and ri.get("model_path") and ri.get("data_path"):
            jobs.append({
                "case_id": case, "series": "full", "method": "SPIDER-CEM",
                "qpos": bp["cem_npz"], "scene": ri["model_path"], "kin_ref": ri["data_path"],
                "mask": ri.get("contact_hdmi_mask_path"),
            })
    # A5 / A6: cem_ok ablation rows
    for r in rows:
        if r.get("status") != "cem_ok":
            continue
        jobs.append({
            "case_id": r["case_id"], "series": r["ablation"], "method": "SPIDER-CEM",
            "qpos": C.repo_path(r["outdir_npz"]),
            "scene": C.repo_path(r["run_model_path"]),
            "kin_ref": C.repo_path(r["run_data_path"]),
            "mask": C.repo_path(r["run_contact_mask_path"]) if r.get("run_contact_mask_path") else None,
        })
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    C.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    if args.fresh and CACHE.is_file():
        CACHE.unlink()
    series_filter = {s for s in args.series.split(",") if s}
    cache = load_cache()
    jobs = jobs_from_manifest()
    if series_filter:
        jobs = [j for j in jobs if j["series"] in series_filter]
    records: list[dict] = []
    for i, j in enumerate(jobs, 1):
        key = (j["case_id"], j["series"])
        if key in cache and not args.fresh:
            records.append(cache[key])
            continue
        try:
            metrics = eval_one(j["case_id"], j["qpos"], j["scene"], j["kin_ref"], j["mask"], j["method"])
            rec = {"case_id": j["case_id"], "series": j["series"],
                   "object_key": C.object_key_of(j["case_id"]), "status": "ok", "metrics": metrics}
        except Exception as exc:  # noqa: BLE001
            rec = {"case_id": j["case_id"], "series": j["series"],
                   "object_key": C.object_key_of(j["case_id"]), "status": f"ERROR:{exc}", "metrics": {}}
        append_cache(rec)
        records.append(rec)
        print(f"[{i:03d}/{len(jobs)}] {j['series']:18s} {j['case_id']:30s} {rec['status']}", flush=True)

    header = ["case_id", "series", "object_key", "status"] + KEYS
    lines = ["\t".join(header)]
    for r in records:
        vals = [r["case_id"], r["series"], r["object_key"], r["status"]]
        vals += [("" if (k not in r["metrics"] or not math.isfinite(r["metrics"][k]))
                  else f"{r['metrics'][k]:.6g}") for k in KEYS]
        lines.append("\t".join(vals))
    OUT_TSV.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ok = sum(1 for r in records if r["status"] == "ok")
    print(f"\nwrote {C.rel(OUT_TSV)} ({ok}/{len(records)} ok)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
