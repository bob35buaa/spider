#!/usr/bin/env python3
"""E073 evaluation: eef_offset-consistent dynamic contact targets."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
EVAL_DIR = REPO / "workspace/core4d/scripts/eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
DEBUG_DIR = REPO / "workspace/core4d/scripts/debug"
if str(DEBUG_DIR) not in sys.path:
    sys.path.insert(0, str(DEBUG_DIR))

import eval_E072 as e072  # noqa: E402
from diagnose_E068_init_drift import _normalize_angle_deg  # noqa: E402

RESULTS = REPO / "workspace/core4d/results/E073"
NPZ_PATH = RESULTS / "E073_box023.npz"
SCENE_SNAPSHOT = RESULTS / "scene_snapshot/box023_person1/scene_act.xml"
ACTIVE_SCENE = (
    REPO
    / "example_datasets/processed/core4d/unitree_g1/humanoid_object/box023_person1/scene_act.xml"
)
BASELINE_SUMMARY = REPO / "workspace/core4d/results/E072/diagnosis_summary.json"
FPS = 50.0


def load_scene_model() -> tuple[mujoco.MjModel, Path]:
    try:
        return mujoco.MjModel.from_xml_path(str(SCENE_SNAPSHOT)), SCENE_SNAPSHOT
    except Exception:
        text = SCENE_SNAPSHOT.read_text(encoding="utf-8")
        patched = text.replace(
            "../../../../../example_datasets",
            str(REPO / "example_datasets"),
        )
        patched_path = RESULTS / "_scene_act_abs.xml"
        patched_path.parent.mkdir(parents=True, exist_ok=True)
        patched_path.write_text(patched, encoding="utf-8")
        try:
            return mujoco.MjModel.from_xml_path(str(patched_path)), patched_path
        except Exception:
            return mujoco.MjModel.from_xml_path(str(ACTIVE_SCENE)), ACTIVE_SCENE


def yaw_deg(q_wxyz: np.ndarray) -> float:
    return float(
        R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_euler(
            "xyz", degrees=True
        )[2]
    )


def count_single_foot_runs(lf_z: np.ndarray, rf_z: np.ndarray) -> int:
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


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    data_npz = np.load(NPZ_PATH, allow_pickle=True)
    qpos = e072.flatten_time_major(data_npz["qpos"])
    qvel = e072.flatten_time_major(data_npz["qvel"])
    ctrl = e072.flatten_time_major(data_npz["ctrl"])
    time = e072.flatten_time_major(data_npz["time"]).astype(float)
    opt_steps = np.repeat(data_npz["opt_steps"].reshape(-1), 2)
    qpos_ref, ctrl_ref = e072.load_ref()

    T = min(
        len(qpos),
        len(qvel),
        len(ctrl),
        len(time),
        len(opt_steps),
        len(qpos_ref),
        len(ctrl_ref),
    )
    qpos = qpos[:T]
    qvel = qvel[:T]
    ctrl = ctrl[:T]
    time = time[:T]
    opt_steps = opt_steps[:T]
    qpos_ref = qpos_ref[:T]
    ctrl_ref = ctrl_ref[:T]

    model, scene_used = load_scene_model()
    sim = e072.replay_metrics(model, qpos)
    ref = e072.replay_metrics(model, qpos_ref)

    obj_err = np.linalg.norm(sim["obj_pos"] - ref["obj_pos"], axis=1)
    pelvis_err = np.linalg.norm(sim["pelvis_pos"] - ref["pelvis_pos"], axis=1)
    sim_min_hand_sdf = np.minimum(sim["left_sdf_m"], sim["right_sdf_m"])
    ref_min_hand_sdf = np.minimum(ref["left_sdf_m"], ref["right_sdf_m"])
    sim_total_contacts = sim["left_contact_count"] + sim["right_contact_count"]
    ref_total_contacts = ref["left_contact_count"] + ref["right_contact_count"]
    ctrl_diff = ctrl - ctrl_ref
    robot_ctrl_linf = np.max(np.abs(ctrl_diff[:, :29]), axis=1)
    robot_ctrl_l2 = np.linalg.norm(ctrl_diff[:, :29], axis=1)
    object_ctrl_linf = np.max(np.abs(ctrl_diff[:, 29:]), axis=1)
    object_ctrl_l2 = np.linalg.norm(ctrl_diff[:, 29:], axis=1)
    qpos_abs_max = np.max(np.abs(qpos - qpos_ref), axis=1)

    rows = []
    for i in range(T):
        rows.append(
            {
                "frame": i,
                "eval_time_s": float(i / FPS),
                "npz_time_s": float(time[i]),
                "opt_step": int(opt_steps[i]),
                "obj_err_m": float(obj_err[i]),
                "pelvis_err_m": float(pelvis_err[i]),
                "pelvis_z_qpos_m": float(qpos[i, 2]),
                "sim_pelvis_z_m": float(sim["pelvis_pos"][i, 2]),
                "ref_pelvis_z_m": float(ref["pelvis_pos"][i, 2]),
                "sim_left_sdf_m": float(sim["left_sdf_m"][i]),
                "sim_right_sdf_m": float(sim["right_sdf_m"][i]),
                "sim_min_hand_sdf_m": float(sim_min_hand_sdf[i]),
                "ref_left_sdf_m": float(ref["left_sdf_m"][i]),
                "ref_right_sdf_m": float(ref["right_sdf_m"][i]),
                "ref_min_hand_sdf_m": float(ref_min_hand_sdf[i]),
                "sim_left_contact_count": int(sim["left_contact_count"][i]),
                "sim_right_contact_count": int(sim["right_contact_count"][i]),
                "sim_total_contact_count": int(sim_total_contacts[i]),
                "ref_left_contact_count": int(ref["left_contact_count"][i]),
                "ref_right_contact_count": int(ref["right_contact_count"][i]),
                "ref_total_contact_count": int(ref_total_contacts[i]),
                "sim_left_foot_z_m": float(sim["left_foot_z"][i]),
                "sim_right_foot_z_m": float(sim["right_foot_z"][i]),
                "ref_left_foot_z_m": float(ref["left_foot_z"][i]),
                "ref_right_foot_z_m": float(ref["right_foot_z"][i]),
                "robot_ctrl_linf": float(robot_ctrl_linf[i]),
                "robot_ctrl_l2": float(robot_ctrl_l2[i]),
                "object_ctrl_linf": float(object_ctrl_linf[i]),
                "object_ctrl_l2": float(object_ctrl_l2[i]),
                "qpos_max_abs_diff_vs_ref": float(qpos_abs_max[i]),
                "qvel_max_abs": float(np.max(np.abs(qvel[i]))),
            }
        )

    timeseries_path = RESULTS / "timeseries.csv"
    with timeseries_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    start = int(2.0 * FPS)
    end = min(int(3.6 * FPS) + 1, T)
    post = slice(start, end)
    frame_window = slice(100, min(146, T))
    yaw_err_0017 = _normalize_angle_deg(yaw_deg(qpos[0, 3:7]) - yaw_deg(qpos_ref[1, 3:7]))
    yaw_err_0033 = _normalize_angle_deg(yaw_deg(qpos[1, 3:7]) - yaw_deg(qpos_ref[2, 3:7]))
    pre_end = min(int(2.0 * FPS) + 1, T)
    b1 = float(max(sim["left_foot_z"][:pre_end].max(), sim["right_foot_z"][:pre_end].max()))

    summary: dict[str, float | int | str] = {
        "npz_path": str(NPZ_PATH.relative_to(REPO)),
        "scene_used": str(scene_used.relative_to(REPO)),
        "T": T,
        "yaw_err_t0017_deg": float(yaw_err_0017),
        "yaw_err_t0033_deg": float(yaw_err_0033),
        "B1_pre_contact_max_foot_z_m": b1,
        "B2_single_foot_runs": count_single_foot_runs(sim["left_foot_z"], sim["right_foot_z"]),
        "first_obj_err_gt_25cm_frame": e072.first_after(obj_err > 0.25, start),
        "first_sim_zero_contact_frame": e072.first_after(sim_total_contacts == 0, start),
        "first_sim_min_hand_sdf_gt_10cm_frame": e072.first_after(sim_min_hand_sdf > 0.10, start),
        "first_pelvis_z_lt_45cm_frame": e072.first_after(sim["pelvis_pos"][:, 2] < 0.45, start),
        "first_robot_ctrl_linf_gt_0p5_frame": e072.first_after(robot_ctrl_linf > 0.50, start),
        "post2_obj_err_max_m": float(obj_err[post].max()),
        "post2_obj_err_mean_m": float(obj_err[post].mean()),
        "post2_sim_contact_frames_pct": float((sim_total_contacts[post] > 0).mean() * 100.0),
        "post2_ref_contact_frames_pct": float((ref_total_contacts[post] > 0).mean() * 100.0),
        "post2_sim_min_hand_sdf_mean_m": float(sim_min_hand_sdf[post].mean()),
        "post2_sim_min_hand_sdf_max_m": float(sim_min_hand_sdf[post].max()),
        "post2_pelvis_z_min_m": float(sim["pelvis_pos"][post, 2].min()),
        "post2_robot_ctrl_linf_max": float(robot_ctrl_linf[post].max()),
        "post2_object_ctrl_linf_max": float(object_ctrl_linf[post].max()),
        "frame100_145_contact_frames_pct": float((sim_total_contacts[frame_window] > 0).mean() * 100.0),
        "frame100_145_ref_contact_frames_pct": float((ref_total_contacts[frame_window] > 0).mean() * 100.0),
        "frame100_145_obj_err_max_m": float(obj_err[frame_window].max()),
        "time_note": "eval_time_s uses frame/50 to match E071/E072 metrics; npz_time_s is also present in CSV.",
    }
    for key in list(summary.keys()):
        if key.endswith("_frame"):
            frame = int(summary[key])
            summary[key.replace("_frame", "_eval_time_s")] = e072.eval_time_or_neg(frame)
            summary[key.replace("_frame", "_npz_time_s")] = e072.npz_time_or_neg(frame, time)

    if BASELINE_SUMMARY.is_file():
        base = json.loads(BASELINE_SUMMARY.read_text(encoding="utf-8"))
        summary["baseline_E071_post2_sim_contact_frames_pct"] = base.get(
            "post2_sim_contact_frames_pct", ""
        )
        summary["baseline_E071_post2_obj_err_max_m"] = base.get("post2_obj_err_max_m", "")
        summary["baseline_E071_first_pelvis_z_lt_45cm_frame"] = base.get(
            "first_pelvis_z_lt_45cm_frame", ""
        )

    summary_path = RESULTS / "eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with (RESULTS / "eval_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    e072.plot_timeline(rows, RESULTS / "plots/post2_failure_timeline.png")
    print(f"Wrote {timeseries_path}")
    print(f"Wrote {summary_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
