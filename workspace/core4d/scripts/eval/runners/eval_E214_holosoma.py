#!/usr/bin/env python3
"""Holosoma-gauge metrics for E214: foot sliding, penetration, contact precision.

Faithful port of the holosoma_retargeting reference evaluator
(Opensource_projects/holosoma/.../evaluation/eval_retargeting.py) so the numbers
are computed with the SAME criteria/thresholds as holosoma. These are reported
SEPARATELY from the existing SPIDER metrics and are always labelled "holosoma".

Data adaptation. holosoma evals InterMimic SMPL *human* demos as the reference.
Our pipeline is CORE4D SMPLX -> OmniRetarget -> SPIDER-CEM, so the faithful
reference is the CORE4D SMPLX human GT, expressed in the rollout scene frame by
``smplx_reference.build_smplx_reference`` (object-anchored similarity + DTW time
map; see that module). We keep every algorithm/threshold identical and substitute:
  * "reference / demo" foot (sticking phase) = SMPLX toe joints (10/11), scene frame.
  * "reference / demo" hand (contact) = true SMPLX wrist joints (20/21), scene frame.
  * "reference / demo" object pose = the human-driven object pose (scene frame).
  * "robot / retarget" foot & hand & object = the rollout NPZ (cem/ablation output).
penetration needs no reference and is unchanged. This is documented per metric below.

Gauge (from eval_retargeting.py):
  penetration_tolerance          = 0.01 m   (strict penetration deeper than 1cm)
  collision_detection_threshold  = 0.10 m   (prefilter + mj_geomDistance margin)
  sliding_threshold              = 0.01     (per-frame |dxy|; NOT x fps -- verbatim)
  contact center threshold       = 0.28 m   (||keypoint in object frame||; the
                                             transform is rotation-invariant so
                                             this is distance to the object centre)

Foot sliding (detect_foot_sliding + extract_foot_sticking_sequence_velocity):
  reference "sticking" per foot = SMPLX toe per-frame |dxy| <= 0.01 (scene frame)
  sliding frame = reference-sticking AND robot toe per-frame |dxy| > 0.01
  foot_sliding_holosoma_frac     = #sliding frames / #sticking frames
  foot_sliding_holosoma_vel_mean = mean of per-frame max(L,R) sliding speed
  (robot foot = <side>_ankle_roll_link body; reference foot = SMPLX toe 10/11.)

Penetration (evaluate_penetration): robot-vs-(object OR floor) strict penetration
  penetration_holosoma_frac        = fraction of frames with any penetration >1cm
  penetration_holosoma_depth_mean_m= mean per-frame max depth over penetrating frames
  penetration_holosoma_depth_max_m = max penetration depth over the sequence

Contact precision (evaluate_contact_precision), hands (L/R SMPLX wrist):
  contact_precision_holosoma = 1 - (frames where the SMPLX GT hand is within X of the
                               reference object SURFACE but the robot hand is NOT
                               within X of the rollout object surface) / N   (swept X)

Output (results/E214/eval/):  e214_holosoma.jsonl  one record per (case, series)

Usage:
    .venv/bin/python workspace/core4d/scripts/eval/runners/eval_E214_holosoma.py --workers 8
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[5]
for _p in ("workspace/core4d/report/0908/code",
           "workspace/core4d/scripts",
           "workspace/core4d/scripts/eval/reports",
           "workspace/core4d/scripts/experiments/E214",
           "workspace/core4d/scripts/eval/runners"):
    sys.path.insert(0, str(REPO / _p))

import gen_paper_results as G  # noqa: E402
G.REPO = REPO  # gen_paper_results computes REPO one level short (parents[4]); fix it.

import e214_common as C  # noqa: E402
from eval_E214_ablation import jobs_from_manifest  # noqa: E402
from smplx_reference import build_smplx_reference  # noqa: E402

CACHE = C.EVAL_DIR / "e214_holosoma.jsonl"
SMPLX_CACHE = C.EVAL_DIR / "smplx_ref"  # per-case SMPLX-GT reference npz cache

# holosoma gauge constants (from eval_retargeting.py, non-multi_boxes)
COLLISION_DETECTION_THRESHOLD = 0.10   # prefilter + mj_geomDistance margin
STICK_THRESHOLD = 0.01                 # reference toe |dxy| <= this => sticking (contact phase)
NQ_ROBOT = 36
ANKLE_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
WRIST_BODIES = ("left_wrist_yaw_link", "right_wrist_yaw_link")

# threshold sweeps (user request)
SLIDE_THRESHOLDS = (0.005, 0.01, 0.02)   # per-frame |dxy| of a stance foot
PEN_THRESHOLDS_M = (0.005, 0.01, 0.02)   # robot-vs-(object|floor) penetration depth (m)
CONTACT_THRESHOLDS_M = (0.005, 0.01, 0.02, 0.05, 0.10)  # hand-to-object-SURFACE distance (m)


def _cm_key(t: float) -> str:  # 0.005->"0.5cm", 0.01->"1cm", 0.10->"10cm"
    return f"contact_precision_holosoma_{t * 100:g}cm"


KEYS = (["foot_sliding_holosoma_vel_mean"]
        + [f"foot_sliding_holosoma_frac_{int(t*1000)}mm" for t in SLIDE_THRESHOLDS]
        + [f"penetration_holosoma_frac_{int(t*1000)}mm" for t in PEN_THRESHOLDS_M]
        + ["penetration_holosoma_depth_max_m"]
        + [_cm_key(t) for t in CONTACT_THRESHOLDS_M])


def build_jobs() -> list[dict[str, str]]:
    """(case, series) -> rollout qpos + shared scene/kin, for every report series."""
    jobs: list[dict[str, str]] = []
    for j in jobs_from_manifest():  # ablations + box023 full-stack
        jobs.append({"case_id": j["case_id"], "series": j["series"],
                     "qpos": str(j["qpos"]), "scene": str(j["scene"]),
                     "kin": str(j["kin_ref"]) if j["kin_ref"] else ""})
    idx = G.SourceIndex()
    for case in C.load_cases():
        if C.object_key_of(case) == "box023":
            continue
        rl = G.find_rl_row(idx, case, G.case_cfg(case)) or {}
        cem = G.resolve(rl.get(G.RL_CEM_NPZ, ""))
        scn = G.resolve(rl.get(G.RL_SCENE_ACT, ""))
        kin = G.resolve(rl.get(G.RL_TRAJECTORY, ""))
        if cem and scn:
            jobs.append({"case_id": case, "series": "full",
                         "qpos": str(cem), "scene": str(scn),
                         "kin": str(kin) if kin else ""})
    return jobs


# ---- holosoma gauge, ported (heavy imports done inside the worker) ----------

def _foot_sliding(mujoco, np, model, run_q, ref_toe_xy):
    """detect_foot_sliding + extract_foot_sticking_sequence_velocity, ported, with a
    sweep over the sliding velocity threshold. The reference "sticking" (contact
    phase) is defined by the SMPLX GT toe (scene frame) per-frame |dxy|, fixed at
    STICK_THRESHOLD as in holosoma; the sweep varies only the robot sliding cutoff.
      ref_toe_xy : (N,2,2) SMPLX toe (L,R) horizontal position in the scene frame.
      vel_mean   = mean per-frame max robot toe speed over sticking frames (thr-free)
      frac[thr]  = #(sticking frames with a stance foot exceeding thr) / #sticking frames
    """
    import mujoco as _mj  # noqa: PLC0415
    ankle_ids = [_mj.mj_name2id(model, _mj.mjtObj.mjOBJ_BODY, n) for n in ANKLE_BODIES]
    if any(a < 0 for a in ankle_ids):
        return math.nan, {t: math.nan for t in SLIDE_THRESHOLDS}
    H = min(len(run_q), len(ref_toe_xy))
    if H < 2:
        return math.nan, {t: math.nan for t in SLIDE_THRESHOLDS}
    data = _mj.MjData(model)

    def ankle_xy(qframe):
        q = qframe[:model.nq] if qframe.shape[0] == model.nq else \
            np.concatenate([qframe[:NQ_ROBOT], np.zeros(model.nq - NQ_ROBOT)])
        data.qpos[:] = q
        data.qvel[:] = 0.0
        _mj.mj_forward(model, data)
        return data.xpos[ankle_ids][:, :2].copy()

    ref_xy = np.asarray(ref_toe_xy[:H])                       # (H,2,2) SMPLX toe
    run_xy = np.array([ankle_xy(run_q[i]) for i in range(H)])  # (H,2,2) robot ankle
    ref_vel = np.linalg.norm(np.diff(ref_xy, axis=0), axis=2)
    ref_vel = np.concatenate([[[STICK_THRESHOLD + 1, STICK_THRESHOLD + 1]], ref_vel], axis=0)
    stick = ref_vel <= STICK_THRESHOLD  # (H,2)
    run_vel = np.linalg.norm(np.diff(run_xy, axis=0), axis=2)
    run_vel = np.concatenate([[[0.0, 0.0]], run_vel], axis=0)  # (H,2)

    stick_any = stick[:, 0] | stick[:, 1]
    num_stick = int(np.sum(stick_any))
    if num_stick == 0:
        return math.nan, {t: math.nan for t in SLIDE_THRESHOLDS}
    per_frame_stick_vel = np.max(run_vel * stick, axis=1)  # (H,) max stance-foot speed
    vel_mean = float(np.mean(per_frame_stick_vel[stick_any]))
    fracs = {}
    for t in SLIDE_THRESHOLDS:
        per_frame_max = np.max(run_vel * (stick & (run_vel > t)), axis=1)
        fracs[t] = float(np.sum(per_frame_max > 0) / num_stick)
    return vel_mean, fracs


def _penetration(mujoco, np, model, run_q):
    """evaluate_penetration, ported: robot vs (object OR floor) strict penetration.

    CORE4D scene_act is a kinematic-replay scene with ALL robot collision geoms
    disabled (contype=conaffinity=0), so holosoma's mj_collision prefilter would
    see nothing. holosoma's robot scene has its collision hull active; we
    replicate that condition by enabling the contact bits on the robot's *named*
    collision geoms (the *_collision / lf,rf foot spheres / lh,rh hand hulls;
    visual meshes stay off) for this pass only, then restore them. object_visual
    stays off, object_collision/floor stay on -- so pairs are robot-vs-object and
    robot-vs-floor, exactly the holosoma gauge."""
    import mujoco as _mj  # noqa: PLC0415
    data = _mj.MjData(model)
    object_bid = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_BODY, "object")
    pelvis_bid = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_BODY, "pelvis")
    floor_gids = {g for g in range(model.ngeom)
                  if model.geom_type[g] == _mj.mjtGeom.mjGEOM_PLANE
                  or "floor" in (_mj.mj_id2name(model, _mj.mjtObj.mjOBJ_GEOM, g) or "").lower()
                  or "ground" in (_mj.mj_id2name(model, _mj.mjtObj.mjOBJ_GEOM, g) or "").lower()}
    obj_gids = {g for g in range(model.ngeom) if model.geom_bodyid[g] == object_bid}

    def _is_robot_body(bid: int) -> bool:
        cur = bid
        while cur > 0:
            if cur == pelvis_bid:
                return True
            cur = int(model.body_parentid[cur])
        return False

    # enable robot collision-hull geoms (named) for this pass; restore afterward
    saved_contype = model.geom_contype.copy()
    saved_conaff = model.geom_conaffinity.copy()
    for g in range(model.ngeom):
        if _is_robot_body(model.geom_bodyid[g]) and _mj.mj_id2name(model, _mj.mjtObj.mjOBJ_GEOM, g):
            model.geom_contype[g] = 1
            model.geom_conaffinity[g] = 1

    def is_obj(g): return g in obj_gids
    def is_ground(g): return g in floor_gids  # noqa: E306

    def masks_ok(g1, g2):
        if model.geom_contype[g1] == 0 and model.geom_conaffinity[g1] == 0:
            return False
        if model.geom_contype[g2] == 0 and model.geom_conaffinity[g2] == 0:
            return False
        if (is_obj(g1) and is_ground(g2)) or (is_obj(g2) and is_ground(g1)):
            return False
        return is_obj(g1) or is_obj(g2) or is_ground(g1) or is_ground(g2)

    saved_margin = model.geom_margin.copy()
    fromto = np.zeros(6, dtype=float)
    per_frame_max_depth: list[float] = []  # deepest robot-vs-(object|floor) penetration per frame
    for q in run_q:
        data.qpos[:] = q[:model.nq]
        _mj.mj_forward(model, data)
        model.geom_margin[:] = COLLISION_DETECTION_THRESHOLD
        _mj.mj_collision(model, data)
        cands = set()
        for k in range(data.ncon):
            c = data.contact[k]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 >= 0 and g2 >= 0:
                cands.add((min(g1, g2), max(g1, g2)))
        model.geom_margin[:] = saved_margin
        best = 0.0
        for g1, g2 in cands:
            if not masks_ok(g1, g2):
                continue
            fromto[:] = 0.0
            dist = _mj.mj_geomDistance(model, data, g1, g2, COLLISION_DETECTION_THRESHOLD, fromto)
            if dist < 0.0:
                best = max(best, -float(dist))
        per_frame_max_depth.append(best)
    model.geom_contype[:] = saved_contype
    model.geom_conaffinity[:] = saved_conaff
    depths = np.asarray(per_frame_max_depth)
    n = max(len(depths), 1)
    fracs = {t: float(np.sum(depths > t) / n) for t in PEN_THRESHOLDS_M}
    depth_max = float(depths.max()) if depths.size else 0.0
    return fracs, depth_max


def _contact_precision(mujoco, np, model, run_q, ref_wrist, ref_obj_pos, ref_obj_quat):
    """evaluate_contact_precision, ported to hand-to-object-SURFACE distance
    (docstring intent: "keypoints <= X from object surface"), swept over X, with
    the SMPLX GT human hand as the reference.

    Per frame, for each hand keypoint (L/R):
      demo_sdf  = min signed distance of the SMPLX GT wrist (scene frame) to the
                  reference-pose object boxes (human-driven object pose)
      robot_sdf = min signed distance of the rollout wrist to the rollout-pose
                  object boxes   (point_object_sdf on the FK'd rollout)
    For threshold X: demo_contact = demo_sdf <= X; robot_contact = robot_sdf <= X.
    A frame is a miss if any hand is demo-contact but not robot-contact.
    precision(X) = 1 - miss_frames / N   (holosoma evaluate_contact_precision formula)
      ref_wrist    : (N,2,3) SMPLX wrist (L,R) in the scene frame.
      ref_obj_pos  : (N,3)   reference object position (scene frame).
      ref_obj_quat : (N,4)   reference object orientation (wxyz, scene frame).
    """
    import mujoco as _mj  # noqa: PLC0415
    from eval.core.core_metrics import (  # noqa: PLC0415
        object_collision_geoms, point_object_sdf, signed_point_box,
    )
    wrist_ids = [_mj.mj_name2id(model, _mj.mjtObj.mjOBJ_BODY, n) for n in WRIST_BODIES]
    object_bid = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_BODY, "object")
    if any(w < 0 for w in wrist_ids) or object_bid < 0:
        return {t: math.nan for t in CONTACT_THRESHOLDS_M}
    try:
        obj_gids = object_collision_geoms(model)
    except ValueError:
        return {t: math.nan for t in CONTACT_THRESHOLDS_M}
    H = min(len(run_q), len(ref_wrist))
    if H == 0:
        return {t: math.nan for t in CONTACT_THRESHOLDS_M}
    data = _mj.MjData(model)

    def ref_object_sdf(point, body_pos, body_quat):
        bmat = np.zeros(9); _mj.mju_quat2Mat(bmat, body_quat); bmat = bmat.reshape(3, 3)
        vals = []
        for gid in obj_gids:
            gm = np.zeros(9); _mj.mju_quat2Mat(gm, model.geom_quat[gid]); gm = gm.reshape(3, 3)
            world_pos = body_pos + bmat @ model.geom_pos[gid]
            world_mat = bmat @ gm
            vals.append(signed_point_box(point, world_pos, world_mat, model.geom_size[gid, :3]))
        return min(vals)

    miss = {t: 0 for t in CONTACT_THRESHOLDS_M}
    for i in range(H):
        # reference: SMPLX GT wrist (scene frame) vs human-driven object pose
        demo_sdf = np.array([ref_object_sdf(ref_wrist[i][h], ref_obj_pos[i], ref_obj_quat[i])
                             for h in range(len(wrist_ids))])
        # robot: FK rollout, object at rollout pose
        data.qpos[:] = run_q[i][:model.nq]; data.qvel[:] = 0.0; _mj.mj_forward(model, data)
        run_wr = data.xpos[wrist_ids].copy()
        robot_sdf = np.array([point_object_sdf(model, data, run_wr[h], obj_gids) for h in range(len(wrist_ids))])
        for t in CONTACT_THRESHOLDS_M:
            demo_c = demo_sdf <= t
            robot_c = robot_sdf <= t
            if np.any(demo_c & (demo_c != robot_c)):
                miss[t] += 1
    return {t: 1.0 - miss[t] / H for t in CONTACT_THRESHOLDS_M}


def _worker(job: dict[str, str]) -> dict[str, Any]:
    import mujoco  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from eval.core.core_metrics import npz_qpos  # noqa: PLC0415

    case, series = job["case_id"], job["series"]
    rec: dict[str, Any] = {"case_id": case, "series": series,
                           "object_key": C.object_key_of(case)}
    try:
        model = mujoco.MjModel.from_xml_path(job["scene"])
        run_q, _ = npz_qpos(Path(job["qpos"]))
        # SMPLX GT reference (scene frame), shared across a case's series (cached).
        ref = None
        if job["kin"]:
            ref_q = np.asarray(np.load(job["kin"], allow_pickle=True)["qpos"], dtype=np.float64)
            if ref_q.ndim == 3:
                ref_q = ref_q[:, 0, :]
            ref = build_smplx_reference(case, ref_q, cache_dir=SMPLX_CACHE)
        m: dict[str, float] = {}
        if ref is not None:
            rec["smplx_gt_align_residual_m"] = ref.align_residual_m
            rec["smplx_gt_status"] = ref.status
            vel_mean, fs_fracs = _foot_sliding(mujoco, np, model, run_q, ref.toe_scene[:, :, :2])
            m["foot_sliding_holosoma_vel_mean"] = vel_mean
            for t, v in fs_fracs.items():
                m[f"foot_sliding_holosoma_frac_{int(t*1000)}mm"] = v
            cp = _contact_precision(mujoco, np, model, run_q, ref.wrist_scene,
                                    ref.obj_pos, ref.obj_quat)
            for t, v in cp.items():
                m[_cm_key(t)] = v
        pen_fracs, pdx = _penetration(mujoco, np, model, run_q)
        for t, v in pen_fracs.items():
            m[f"penetration_holosoma_frac_{int(t*1000)}mm"] = v
        m["penetration_holosoma_depth_max_m"] = pdx
        rec.update(status="ok", metrics={k: (float(v) if v is not None and math.isfinite(float(v)) else None)
                                         for k, v in m.items()})
    except Exception as exc:  # noqa: BLE001
        rec.update(status=f"ERROR:{exc}", metrics={})
    return rec


def load_cache() -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    if CACHE.is_file():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[(r["case_id"], r["series"])] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    C.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    if args.fresh and CACHE.is_file():
        CACHE.unlink()
    cache = load_cache()
    jobs = [j for j in build_jobs() if (j["case_id"], j["series"]) not in cache]

    # Precompute the per-case SMPLX-GT reference single-threaded so pool workers
    # only read the cache (no write race), and report alignment coverage upfront.
    if jobs:
        kin_by_case: dict[str, str] = {}
        for j in build_jobs():
            if j["kin"] and j["case_id"] not in kin_by_case:
                kin_by_case[j["case_id"]] = j["kin"]
        noalign = []
        for case, kin in kin_by_case.items():
            q = np.asarray(np.load(kin, allow_pickle=True)["qpos"], dtype=np.float64)
            r = build_smplx_reference(case, q, cache_dir=SMPLX_CACHE)
            if not r.status.startswith("ok"):
                noalign.append(case)
        print(f"SMPLX-GT reference: {len(kin_by_case)} cases aligned, "
              f"{len(noalign)} NO_GT_ALIGN {noalign}", flush=True)

    print(f"{len(cache)} cached, {len(jobs)} to compute (workers={args.workers})", flush=True)
    if jobs:
        with CACHE.open("a", encoding="utf-8") as fh, \
                mp.Pool(processes=max(1, args.workers)) as pool:
            for i, rec in enumerate(pool.imap_unordered(_worker, jobs), 1):
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                print(f"[{i:03d}/{len(jobs)}] {rec['series']:22s} {rec['case_id']:30s} {rec['status']}", flush=True)
    records = load_cache()
    ok = sum(1 for r in records.values() if r.get("status") == "ok")
    print(f"\nwrote {C.rel(CACHE)} ({ok}/{len(records)} ok)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
