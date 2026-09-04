#!/usr/bin/env python3
"""Build the review-player arm_sweep TSV for the E206 noPRG/PRG comparison.

Turns the already-scored `e206_two_arm_rollout.tsv` into the schema the viser
review player expects (review_index.CaseRecord): 2 rows per case (noPRG / PRG),
each pointing at that arm's CEM rollout + scene for live-qpos playback, with the
12 numeric gate_pass flags and the top-bar metric columns. **No re-scoring** --
the metrics come straight from the eval TSV, paths are reconstructed.

Registered as exp "E206ARM" (arm_sweep) in review_index.py; launched via
    bash workspace/core4d/scripts/eval/wrappers/review_player.sh E206ARM

Unlike E204ARM, both arms here are fresh E206 rollouts on the same task dir, so
the paths are symmetric -- no special-casing one arm to a reused experiment.

Usage:
    .venv/bin/python .../build_arm_review_tsv.py
"""

from __future__ import annotations

import csv
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "E201"))

import funnel_config as FC  # noqa: E402
import e206_common as C  # noqa: E402

REPO = C.REPO
EVAL_DIR = C.S6_DIR / "eval/two_arm"
ROLLOUT_TSV = EVAL_DIR / "e206_two_arm_rollout.tsv"
OUT_TSV = EVAL_DIR / "e206_arm_case_metrics.tsv"
RENDER_DIR = C.S6_DIR / "render/full"

# review_index GATE_FIELDS -> (metric field, op, threshold).
# Thresholds match E204ARM's so the two review sessions are read on one scale.
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
    "hand_object_physics_contact_in_mask_frac",
    "hand_object_physics_penetration_3mm_frame_frac",
    "hand_object_release_false_contact_3mm_frac",
    "track_root_pos_err_cm_mean", "track_root_ori_err_deg_mean",
    "track_eef_pos_err_cm_mean", "track_eef_ori_err_deg_mean",
    "track_obj_pos_err_cm_mean", "track_obj_ori_err_deg_mean",
    "fall_flag",
)
ARM_LABEL = {"noprg": "noPRG", "prg": "PRG"}


def target_task_by_case() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in ("omnirt_v1", "omnirt_v2"):
        m = C.S3_DIR / f"{name}/ref_fk/stage2b_manifest_{name}_ref_fk.tsv"
        if not m.is_file():
            continue
        for row in C.read_tsv(m):
            if row.get("stage2b_status") == "pass":
                out[row["case_id"]] = row["target_task"]
    return out


def _finite(v: str) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else math.nan
    except (TypeError, ValueError):
        return math.nan


def gate_pass(field: str, op: str, thr: float, row: dict) -> str:
    if op == "fall":
        return str(str(row.get("fall_flag", "")).strip().lower() not in ("true", "1"))
    return str(FC.passes(op, _finite(row.get(field, "")), thr))


def main() -> int:
    if not ROLLOUT_TSV.is_file():
        print(f"missing {ROLLOUT_TSV}; run eval_E206_arm_ablation.py first",
              file=sys.stderr)
        return 2
    tasks = target_task_by_case()
    with ROLLOUT_TSV.open(encoding="utf-8") as fh:
        rollout = list(csv.DictReader(fh, delimiter="\t"))

    cols = (["arm", "case_id", "variant", "object_key", "retarget_variant_id",
             "numeric_release_pass", "numeric_failure_modes", "status",
             "outdir_npz", "scene_xml", "config_act", "trajectory", "video"]
            + list(GATE_MAP) + list(METRIC_COLUMNS))
    out_rows, missing = [], []
    for r in rollout:
        arm, cid = r["arm"], r["case_id"]
        task = tasks.get(cid)
        if task is None:
            missing.append(cid)
            continue
        tdir = C.PROCESSED_ROOT / task
        out_dir = C.arm_out_dir(arm, cid, "full")
        npz = out_dir / "trajectory_mjwp_act.npz"
        cfg = out_dir / "config_act.yaml"
        scene = tdir / f"{C.SCENE_BY_ARM[arm]}.xml"
        traj = tdir / "0/trajectory_kinematic.npz"
        mp4 = RENDER_DIR / f"E206_{cid}_{arm}.mp4"
        rec = {
            "arm": ARM_LABEL[arm], "case_id": cid, "variant": "orig",
            "object_key": r["object_key"],
            "retarget_variant_id": "omnirt_v2" if "omnirt_v2" in task else "omnirt_v1",
            # the player's headline flag = the funnel's narrow caliber, same as E204ARM
            "numeric_release_pass": r.get("narrow_pass", ""),
            "numeric_failure_modes": r.get("narrow_failed", ""),
            "status": f"{ARM_LABEL[arm]}_FULL_COMPLETE",
            "outdir_npz": str(npz.relative_to(REPO)) if npz.is_file() else "",
            "scene_xml": str(scene.relative_to(REPO)),
            "config_act": str(cfg.relative_to(REPO)) if cfg.is_file() else "",
            "trajectory": str(traj.relative_to(REPO)),
            "video": str(mp4.relative_to(REPO)) if mp4.is_file() else "",
        }
        for gf, (field, op, thr) in GATE_MAP.items():
            rec[gf] = gate_pass(field, op, thr, r)
        for m in METRIC_COLUMNS:
            rec[m] = r.get(m, "")
        out_rows.append(rec)

    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        w.writeheader()
        w.writerows(out_rows)
    playable = sum(1 for r in out_rows if r["outdir_npz"])
    with_video = sum(1 for r in out_rows if r["video"])
    print(f"[done] {OUT_TSV.relative_to(REPO)}  rows={len(out_rows)} "
          f"playable={playable} with_video={with_video}")
    print("  per-arm:", dict(Counter(r["arm"] for r in out_rows)))
    if missing:
        print(f"  [warn] {len(missing)} rows had no Stage2b task: {sorted(set(missing))[:5]}")
    return 0 if playable == len(out_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
