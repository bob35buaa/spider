#!/usr/bin/env python3
"""E071 evaluation: scene_act ctrl mapping fix validation for box023."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
DEBUG_DIR = REPO / "workspace/core4d/scripts/debug"
if str(DEBUG_DIR) not in sys.path:
    sys.path.insert(0, str(DEBUG_DIR))

from diagnose_E068_init_drift import (  # noqa: E402
    _build_config,
    _convert_scene_act_ref,
    _normalize_angle_deg,
)
from spider.io import load_data  # noqa: E402

RESULTS = REPO / "workspace/core4d/results/E071"
DATASET = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object/box023_person1"
RUNS = [("E071W02_box023", "core4d_e071w02_box023")]
EXPECTED_ORIG_YAW = {1: 0.574, 2: 1.075}
FPS = 50.0


def yaw_deg(q):
    return float(R.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz", degrees=True)[2])


def load_ref():
    cfg = _build_config("core4d_e062_box023", "box023_person1", "cpu", 4)
    qpos_ref, qvel_ref, ctrl_ref, _contact, _contact_pos = load_data(cfg, cfg.data_path)
    qpos_ref, _qvel_ref, ctrl_ref = _convert_scene_act_ref(cfg, qpos_ref, qvel_ref, ctrl_ref)
    return qpos_ref.detach().cpu().numpy(), ctrl_ref.detach().cpu().numpy()


def foot_z(qpos):
    model = mujoco.MjModel.from_xml_path(str(DATASET / "scene_act.xml"))
    data = mujoco.MjData(model)
    lf = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    rf = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
    lf_z, rf_z = [], []
    for q in qpos:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        lf_z.append(float(data.site_xpos[lf, 2]))
        rf_z.append(float(data.site_xpos[rf, 2]))
    return np.asarray(lf_z), np.asarray(rf_z)


def body_metrics(qpos, qpos_ref):
    model = mujoco.MjModel.from_xml_path(str(DATASET / "scene_act.xml"))
    data = mujoco.MjData(model)
    obj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    def series(qs):
        obj_pos, pelvis_pos = [], []
        for q in qs:
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            obj_pos.append(data.xpos[obj].copy())
            pelvis_pos.append(data.xpos[pelvis].copy())
        return np.asarray(obj_pos), np.asarray(pelvis_pos)

    obj_pos, pelvis_pos = series(qpos)
    obj_ref, pelvis_ref = series(qpos_ref)
    return obj_pos, obj_ref, pelvis_pos, pelvis_ref


def count_single_foot_runs(lf_z, rf_z):
    single = ((lf_z > 0.05) & (rf_z < 0.02)) | ((rf_z > 0.05) & (lf_z < 0.02))
    runs = 0
    i = 0
    while i < len(single):
        if not single[i]:
            i += 1
            continue
        j = i
        while j < len(single) and single[j]:
            j += 1
        if j - i >= 5:
            runs += 1
        i = j
    return runs


def main():
    qpos_ref, ctrl_ref = load_ref()
    rows = []
    for label, override in RUNS:
        path = RESULTS / f"{label}.npz"
        if not path.is_file():
            rows.append({"label": label, "override": override, "status": "missing"})
            continue
        d = np.load(path)
        qpos_raw = d["qpos"]
        ctrl_raw = d["ctrl"]
        qpos = qpos_raw.reshape(-1, qpos_raw.shape[-1])
        ctrl = ctrl_raw.reshape(-1, ctrl_raw.shape[-1])
        T = min(len(qpos), len(qpos_ref), len(ctrl), len(ctrl_ref))
        qpos = qpos[:T]
        qpos_ref_t = qpos_ref[:T]
        ctrl = ctrl[:T]

        yaw_err_0017 = _normalize_angle_deg(yaw_deg(qpos[0, 3:7]) - yaw_deg(qpos_ref[1, 3:7]))
        yaw_err_0033 = _normalize_angle_deg(yaw_deg(qpos[1, 3:7]) - yaw_deg(qpos_ref[2, 3:7]))
        warmup_steps = min(12, T)
        ctrl_diff = ctrl[:warmup_steps] - ctrl_ref[:warmup_steps]
        robot_ctrl_max = float(np.max(np.abs(ctrl_diff[:, :29])))
        object_ctrl_max = float(np.max(np.abs(ctrl_diff[:, 29:])))

        lf_z, rf_z = foot_z(qpos)
        pre_end = min(int(2.0 * FPS) + 1, T)
        b1 = float(max(lf_z[:pre_end].max(), rf_z[:pre_end].max()))
        b2 = count_single_foot_runs(lf_z, rf_z)
        intent_start, intent_end = 21, min(78, T)
        pelvis_min_intent = float(qpos[intent_start:intent_end, 2].min())
        pelvis_mean_intent = float(qpos[intent_start:intent_end, 2].mean())
        obj_pos, obj_ref, pelvis_pos, pelvis_ref = body_metrics(qpos, qpos_ref_t)
        obj_err = np.linalg.norm(obj_pos - obj_ref, axis=1)
        pelvis_err = np.linalg.norm(pelvis_pos - pelvis_ref, axis=1)
        post2_start = min(int(2.0 * FPS), T)
        post2 = slice(post2_start, T)
        first_obj_fail = np.where(obj_err[post2] > 0.25)[0]
        first_fall = np.where(pelvis_pos[post2, 2] < 0.45)[0]
        first_obj_fail_frame = (
            int(first_obj_fail[0] + post2_start) if len(first_obj_fail) else -1
        )
        first_fall_frame = int(first_fall[0] + post2_start) if len(first_fall) else -1
        rows.append(
            {
                "label": label,
                "override": override,
                "status": "ok",
                "T": T,
                "yaw_err_t0017_deg": yaw_err_0017,
                "yaw_err_t0033_deg": yaw_err_0033,
                "yaw_delta_vs_E070_orig_t0017_deg": yaw_err_0017 - EXPECTED_ORIG_YAW[1],
                "yaw_delta_vs_E070_orig_t0033_deg": yaw_err_0033 - EXPECTED_ORIG_YAW[2],
                "robot_ctrl_max_diff_warmup": robot_ctrl_max,
                "object_ctrl_max_diff_warmup": object_ctrl_max,
                "B1_pre_contact_max_foot_z_m": b1,
                "B2_single_foot_runs": b2,
                "pelvis_min_intent": pelvis_min_intent,
                "pelvis_mean_intent": pelvis_mean_intent,
                "post2_obj_err_max_m": float(obj_err[post2].max()),
                "post2_obj_err_mean_m": float(obj_err[post2].mean()),
                "post2_pelvis_body_z_min_m": float(pelvis_pos[post2, 2].min()),
                "post2_pelvis_err_max_m": float(pelvis_err[post2].max()),
                "first_post2_obj_err_gt_25cm_frame": first_obj_fail_frame,
                "first_post2_obj_err_gt_25cm_time_s": (
                    first_obj_fail_frame / FPS if first_obj_fail_frame >= 0 else -1.0
                ),
                "first_post2_pelvis_z_lt_45cm_frame": first_fall_frame,
                "first_post2_pelvis_z_lt_45cm_time_s": (
                    first_fall_frame / FPS if first_fall_frame >= 0 else -1.0
                ),
            }
        )

    out = RESULTS / "eval_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({k for row in rows for k in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
