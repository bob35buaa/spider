#!/usr/bin/env python3
"""Build the review-player arm_sweep TSV for the E204/E205/E178 three-arm compare.

Turns the already-scored three_arm_rollout.tsv into the schema the viser review
player expects (review_index.CaseRecord): 3 rows per case (arm in {noPRG, PRG,
G1A2}), each pointing at that arm's CEM rollout + scene for live-qpos playback,
with the 12 numeric gate_pass flags + the top-bar metric columns. NO re-scoring:
metrics come from three_arm_rollout.tsv; paths are reconstructed deterministically.

Registered as exp "E204ARM" (arm_sweep) in review_index.py; launched via
    bash workspace/core4d/scripts/eval/wrappers/review_player.sh E204ARM

Usage:
    .venv/bin/python workspace/core4d/scripts/experiments/E204_E205/build_arm_review_tsv.py
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/experiments/E201"))
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/experiments/E204_E205"))
import funnel_config as FC  # noqa: E402
import e204e205_common as C  # noqa: E402

EVAL_DIR = REPO / "workspace/core4d/results/E204/s6_downstream/eval/three_arm"
ROLLOUT_TSV = EVAL_DIR / "three_arm_rollout.tsv"
OUT_TSV = EVAL_DIR / "e204e205_arm_case_metrics.tsv"
ARM_KEY = {"noPRG": "noprg_e204", "G1A2": "g1a2_e205"}  # PRG handled via E178 glob

# review_index GATE_FIELDS -> (metric field, op, narrow threshold)
GATE_MAP = {
    "fall_gate_pass": ("fall_flag", "fall", 0.0),
    "body_z_gate_pass": ("body_z_err_p95_m", "<=", 0.20),
    "contact_gate_pass": ("hand_object_physics_contact_in_mask_frac", ">=", 0.50),
    "release_gate_pass": ("hand_object_release_false_contact_3mm_frac", "<=", 0.30),
    "hand_penetration_gate_pass": ("hand_object_physics_penetration_3mm_frame_frac", "<=", 0.32),
    "lower_body_gate_pass": ("leg_penetration_frac", "<=", 0.20),
    "root_pos_gate_pass": ("track_root_pos_err_cm_mean", "<=", 20.0),
    "root_ori_gate_pass": ("track_root_ori_err_deg_mean", "<=", 20.0),
    "hand_pos_gate_pass": ("track_eef_pos_err_cm_mean", "<=", 20.0),
    "hand_ori_gate_pass": ("track_eef_ori_err_deg_mean", "<=", 20.0),
    "object_pos_gate_pass": ("track_obj_pos_err_cm_mean", "<=", 20.0),
    "object_ori_gate_pass": ("track_obj_ori_err_deg_mean", "<=", 10.0),
}
METRIC_COLUMNS = (
    "body_z_err_p95_m", "leg_penetration_frac",
    "hand_object_physics_penetration_3mm_frame_frac",
    "hand_object_release_false_contact_3mm_frac",
    "track_root_pos_err_cm_mean", "track_root_ori_err_deg_mean",
    "track_eef_pos_err_cm_mean", "track_eef_ori_err_deg_mean",
    "track_obj_pos_err_cm_mean", "track_obj_ori_err_deg_mean",
    "fall_flag",
)


def prg_rollout(case_id: str) -> Path | None:
    hits = sorted(REPO.glob(
        f"workspace/core4d/results/E178/s6_downstream/cem/full/*{case_id}*/trajectory_mjwp_act.npz"))
    return hits[0] if hits else None


def paths_for(arm: str, case_id: str, source: dict) -> tuple[str, str, str, str]:
    """(outdir_npz, scene_xml, config_act, trajectory) for one arm/case."""
    _tgt, trajectory, _mask = C.base.local_authorities(source)
    if arm == "PRG":
        npz = prg_rollout(case_id)
        scene = C.e178_scene_path(case_id)
    else:
        npz = C.result_npz(ARM_KEY[arm], case_id, "full")
        scene = C.arm_scene_path(ARM_KEY[arm], case_id)
    if npz is None or not npz.is_file():
        return "", str(scene), "", str(trajectory)
    cfg = npz.parent / "config_act.yaml"
    return (str(npz.relative_to(REPO)), str(scene.relative_to(REPO)),
            str(cfg.relative_to(REPO)) if cfg.is_file() else "", str(trajectory.relative_to(REPO)))


def _finite(v: str) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else math.nan
    except (TypeError, ValueError):
        return math.nan


def gate_pass(gfield: str, op: str, thr: float, row: dict) -> str:
    if op == "fall":
        return str(str(row.get("fall_flag", "")).strip().lower() not in ("true", "1"))
    return str(FC.passes(op, _finite(row.get(gfield, "")), thr))


def main() -> int:
    if not ROLLOUT_TSV.is_file():
        print(f"missing {ROLLOUT_TSV}; run eval_E204E205_arm_ablation.py first", file=sys.stderr)
        return 2
    src_by_case = {s["case_id"]: s for s in C.load_sources()}
    with ROLLOUT_TSV.open(encoding="utf-8") as fh:
        rollout = list(csv.DictReader(fh, delimiter="\t"))

    cols = (["arm", "case_id", "variant", "object_key", "retarget_variant_id",
             "numeric_release_pass", "numeric_failure_modes", "status",
             "outdir_npz", "scene_xml", "config_act", "trajectory", "video"]
            + list(GATE_MAP) + list(METRIC_COLUMNS))
    out_rows = []
    for r in rollout:
        arm, cid = r["arm"], r["case_id"]
        src = src_by_case.get(cid)
        if src is None:
            continue
        outdir_npz, scene_xml, config_act, trajectory = paths_for(arm, cid, src)
        task = C.task_of(cid)
        rec = {
            "arm": arm, "case_id": cid, "variant": "orig",
            "object_key": r["object_key"],
            "retarget_variant_id": "omnirt_v2" if "omnirt_v2" in task else "omnirt_v1",
            "numeric_release_pass": r.get("narrow_pass", ""),
            "numeric_failure_modes": r.get("narrow_failed", ""),
            "status": f"{arm}_FULL_COMPLETE",
            "outdir_npz": outdir_npz, "scene_xml": scene_xml,
            "config_act": config_act, "trajectory": trajectory, "video": "",
        }
        for gf, (mfield, op, thr) in GATE_MAP.items():
            rec[gf] = gate_pass(mfield, op, thr, r)
        for m in METRIC_COLUMNS:
            rec[m] = r.get(m, "")
        out_rows.append(rec)

    with OUT_TSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        w.writeheader()
        w.writerows(out_rows)
    playable = sum(1 for r in out_rows if r["outdir_npz"] and (REPO / r["outdir_npz"]).is_file())
    print(f"[done] wrote {OUT_TSV.relative_to(REPO)} ({len(out_rows)} rows, {playable} playable)")
    from collections import Counter
    print("  per-arm:", dict(Counter(r["arm"] for r in out_rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
