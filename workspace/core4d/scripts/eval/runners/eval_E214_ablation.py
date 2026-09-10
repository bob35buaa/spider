#!/usr/bin/env python3
"""E214 eval: compute the paper's 15-metric set for every ablation rollout, plus
the box023 full-stack baseline (E173) rollouts (not in the paper cache).

Uses the shared eval module verbatim (rule 13): eval.core.core_metrics
(evaluate_sequence + EvalConfig + person_idx_from_case), eval.core.motion_health
(run_health), eval.runners.eval_E194_G1_expansion (fixed_reference_z_metrics) --
the exact definitions gen_paper_results.py uses, so ablation and paper numbers
are comparable.

Outputs (results/E214/eval/):
  e214_metrics.jsonl   one record per (case, series) with all 15 metrics
  e214_metrics.tsv     flat table

series:
  full__box023   the 7 box023 E173 baselines (recomputed here)
  A1_contactHDMI_only / A2_surfaceBand_only / A3_softPenalty_only / A4_hardGate_only

Usage:
    .venv/bin/python workspace/core4d/scripts/eval/runners/eval_E214_ablation.py
    ... --series A1_contactHDMI_only,...   # subset
    ... --fresh                            # ignore cache
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
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/experiments/E214"))

import e214_common as C  # noqa: E402
from eval.core.core_metrics import (  # noqa: E402
    EvalConfig,
    evaluate_sequence,
    person_idx_from_case,
)
from eval.core.motion_health import run_health  # noqa: E402
from eval.runners.eval_E194_G1_expansion import fixed_reference_z_metrics  # noqa: E402

HAND_COLLISION_VARIANT = "rubber_hull"

# 15-metric spec (mirrors report/0908/code/gen_paper_results.py METRICS).
METRICS: list[dict[str, Any]] = [
    {"key": "raw_contact",    "field": "hand_object_physics_contact_in_mask_frac",      "src": "result"},
    {"key": "contact_3mm",    "field": "hand_object_physics_contact_3mm_in_mask_frac",  "src": "result"},
    {"key": "contact_5mm",    "field": "hand_object_physics_contact_5mm_in_mask_frac",  "src": "result"},
    {"key": "contact_10mm",   "field": "hand_object_physics_contact_10mm_in_mask_frac", "src": "result"},
    {"key": "pen_3mm",        "field": "hand_object_physics_penetration_3mm_frame_frac","src": "result"},
    {"key": "pen_5mm",        "field": "hand_object_physics_penetration_5mm_frame_frac","src": "result"},
    {"key": "pen_10mm",       "field": "hand_object_physics_penetration_10mm_frame_frac","src": "result"},
    {"key": "pen_max_mm",     "field": "hand_object_physics_penetration_max_mm",        "src": "result"},
    {"key": "geom_2mm",       "field": "hand_geom_penetration_2mm_frac",                "src": "result"},
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
CACHE = C.EVAL_DIR / "e214_metrics.jsonl"
OUT_TSV = C.EVAL_DIR / "e214_metrics.tsv"


def _num(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return math.nan
    return f if math.isfinite(f) else math.nan


def eval_one(case_id: str, qpos: Path, scene: Path, kin_ref: Path | None,
             mask: Path | None, method: str) -> dict[str, float]:
    row = {"case_id": case_id, "variant": f"e214_{case_id}",
           "object_key": C.object_key_of(case_id), "object_category": "",
           "expected_quality": "e214"}
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
    """One eval job per cem_ok ablation row + one per box023 E173 baseline."""
    rows, _ = C.read_with_fields(C.MANIFEST)
    jobs: list[dict[str, Any]] = []
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
    # box023 full-stack baseline (E173) -- 7 cases, recomputed (not in paper cache).
    for case in C.load_cases():
        if C.object_key_of(case) != "box023":
            continue
        bp = C.baseline_paths(case)
        ri = bp.get("run_inputs", {})
        if bp.get("cem_npz") and ri.get("model_path") and ri.get("data_path"):
            jobs.append({
                "case_id": case, "series": "full__box023", "method": "SPIDER-CEM",
                "qpos": bp["cem_npz"], "scene": ri["model_path"], "kin_ref": ri["data_path"],
                "mask": ri.get("contact_hdmi_mask_path"),
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
        print(f"[{i:03d}/{len(jobs)}] {j['series']:22s} {j['case_id']:30s} {rec['status']}", flush=True)

    # flat tsv
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
