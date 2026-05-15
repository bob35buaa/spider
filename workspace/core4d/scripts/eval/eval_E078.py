#!/usr/bin/env python3
"""E078 evaluation for CORE4D 3cm per-hand contact mask experiments."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

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
import eval_E073 as e073  # noqa: E402
from diagnose_E068_init_drift import (  # noqa: E402
    _build_config,
    _convert_scene_act_ref,
    _normalize_angle_deg,
)
from spider.io import load_data  # noqa: E402

RESULTS = REPO / "workspace/core4d/results/E078"
FPS = 50.0

VARIANTS = {
    "E078A": {
        "name": "E078A_box023_p1",
        "case": "box023_person1",
        "override": "core4d_e078a_box023_p1_3cm",
    },
    "E078B": {
        "name": "E078B_box023_p2",
        "case": "box023_person2",
        "override": "core4d_e078b_box023_p2_3cm",
    },
}


def load_scene_model(case: str) -> tuple[mujoco.MjModel, Path]:
    snapshot = RESULTS / "scene_snapshot" / case / "scene_act.xml"
    active = (
        REPO
        / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
        / case
        / "scene_act.xml"
    )
    if snapshot.is_file():
        try:
            return mujoco.MjModel.from_xml_path(str(snapshot)), snapshot
        except Exception:
            text = snapshot.read_text(encoding="utf-8")
            patched = text.replace(
                "../../../../../example_datasets",
                str(REPO / "example_datasets"),
            )
            patched_path = RESULTS / f"_scene_act_abs_{case}.xml"
            patched_path.parent.mkdir(parents=True, exist_ok=True)
            patched_path.write_text(patched, encoding="utf-8")
            try:
                return mujoco.MjModel.from_xml_path(str(patched_path)), patched_path
            except Exception:
                pass
    return mujoco.MjModel.from_xml_path(str(active)), active


def load_ref(override: str, case: str) -> tuple[np.ndarray, np.ndarray]:
    cfg = _build_config(override, case, "cpu", 4)
    qpos_ref, qvel_ref, ctrl_ref, _contact, _contact_pos = load_data(cfg, cfg.data_path)
    qpos_ref, _qvel_ref, ctrl_ref = _convert_scene_act_ref(
        cfg, qpos_ref, qvel_ref, ctrl_ref
    )
    return qpos_ref.detach().cpu().numpy(), ctrl_ref.detach().cpu().numpy()


def resize_mask(mask_np: np.ndarray, target_len: int) -> np.ndarray:
    if mask_np.shape[0] == target_len:
        return mask_np.astype(np.float32)
    idx = np.round(np.linspace(0, mask_np.shape[0] - 1, target_len)).astype(np.int64)
    return mask_np[idx].astype(np.float32)


def contact_mask_stats(override: str, case: str, target_len: int) -> dict[str, object]:
    cfg = _build_config(override, case, "cpu", 4)
    if cfg.contact_hdmi_mask_source != "core4d_3cm":
        return {}
    path = Path(cfg.contact_hdmi_mask_path)
    if not path.is_absolute():
        path = REPO / path
    data = np.load(path, allow_pickle=True)
    axis = cfg.contact_hdmi_mask_time_axis
    if axis == "auto":
        if (
            "spider_contact_mask_3cm" in data
            and data["spider_contact_mask_3cm"].shape[0] == target_len
        ):
            axis = "spider"
        elif (
            "eval_contact_mask_3cm" in data
            and data["eval_contact_mask_3cm"].shape[0] == target_len
        ):
            axis = "eval"
        else:
            axis = "eval" if "eval_contact_mask_3cm" in data else "spider"
    key = f"{axis}_contact_mask_3cm"
    raw = data[key]
    mask = resize_mask(raw[:, int(cfg.contact_hdmi_mask_person_idx), :], target_len)
    frame_window = slice(115, min(131, target_len))
    return {
        "mask_path": str(path.relative_to(REPO)),
        "mask_key": key,
        "mask_person_idx": int(cfg.contact_hdmi_mask_person_idx),
        "mask_original_len": int(raw.shape[0]),
        "mask_target_len": int(target_len),
        "mask_left_active_pct": float(mask[:, 0].mean() * 100.0),
        "mask_right_active_pct": float(mask[:, 1].mean() * 100.0),
        "mask_frame115_130_left_active_pct": float(mask[frame_window, 0].mean() * 100.0),
        "mask_frame115_130_right_active_pct": float(mask[frame_window, 1].mean() * 100.0),
    }


def actuator_index(model: mujoco.MjModel, name: str) -> int:
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name))


def foot_xy(model: mujoco.MjModel, qpos: np.ndarray) -> dict[str, np.ndarray]:
    data = mujoco.MjData(model)
    left_foot = e072.name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    right_foot = e072.name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
    out = {
        "left": np.zeros((len(qpos), 2), dtype=float),
        "right": np.zeros((len(qpos), 2), dtype=float),
    }
    for t, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        out["left"][t] = data.site_xpos[left_foot, :2]
        out["right"][t] = data.site_xpos[right_foot, :2]
    return out


def frame_time(summary: dict[str, object], key: str, time: np.ndarray) -> None:
    frame = int(summary[key])
    summary[key.replace("_frame", "_eval_time_s")] = e072.eval_time_or_neg(frame)
    summary[key.replace("_frame", "_npz_time_s")] = e072.npz_time_or_neg(frame, time)


def evaluate_variant(variant: str) -> dict[str, object]:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown E078 variant: {variant}")
    meta = VARIANTS[variant]
    name = str(meta["name"])
    case = str(meta["case"])
    override = str(meta["override"])
    npz_path = RESULTS / f"{name}.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)

    model, scene_used = load_scene_model(case)
    data_npz = np.load(npz_path, allow_pickle=True)
    qpos = e072.flatten_time_major(data_npz["qpos"])
    qvel = e072.flatten_time_major(data_npz["qvel"])
    ctrl = e072.flatten_time_major(data_npz["ctrl"])
    time = e072.flatten_time_major(data_npz["time"]).astype(float)
    opt_steps = np.repeat(data_npz["opt_steps"].reshape(-1), 2)
    qpos_ref, ctrl_ref = load_ref(override, case)

    T = min(
        len(qpos),
        len(qvel),
        len(ctrl),
        len(time),
        len(opt_steps),
        len(qpos_ref),
        len(ctrl_ref),
    )
    qpos, qvel, ctrl, time, opt_steps = qpos[:T], qvel[:T], ctrl[:T], time[:T], opt_steps[:T]
    qpos_ref, ctrl_ref = qpos_ref[:T], ctrl_ref[:T]

    sim = e072.replay_metrics(model, qpos)
    ref = e072.replay_metrics(model, qpos_ref)
    sim_foot_xy = foot_xy(model, qpos)
    ref_foot_xy = foot_xy(model, qpos_ref)

    obj_err = np.linalg.norm(sim["obj_pos"] - ref["obj_pos"], axis=1)
    pelvis_err = np.linalg.norm(sim["pelvis_pos"] - ref["pelvis_pos"], axis=1)
    sim_min_hand_sdf = np.minimum(sim["left_sdf_m"], sim["right_sdf_m"])
    ref_min_hand_sdf = np.minimum(ref["left_sdf_m"], ref["right_sdf_m"])
    sim_total_contacts = sim["left_contact_count"] + sim["right_contact_count"]
    ref_total_contacts = ref["left_contact_count"] + ref["right_contact_count"]
    ctrl_diff = ctrl - ctrl_ref
    robot_ctrl_linf = np.max(np.abs(ctrl_diff[:, :29]), axis=1)
    robot_ctrl_l2 = np.linalg.norm(ctrl_diff[:, :29], axis=1)
    if ctrl_diff.shape[1] > 29:
        object_ctrl_linf = np.max(np.abs(ctrl_diff[:, 29:]), axis=1)
        object_ctrl_l2 = np.linalg.norm(ctrl_diff[:, 29:], axis=1)
    else:
        object_ctrl_linf = np.zeros(T)
        object_ctrl_l2 = np.zeros(T)

    rhip_idx = actuator_index(model, "right_hip_pitch_joint")
    rhip_diff = ctrl_diff[:, rhip_idx] if 0 <= rhip_idx < ctrl_diff.shape[1] else np.zeros(T)
    sim_rstep = np.zeros(T)
    ref_rstep = np.zeros(T)
    sim_rstep[1:] = np.linalg.norm(np.diff(sim_foot_xy["right"], axis=0), axis=1)
    ref_rstep[1:] = np.linalg.norm(np.diff(ref_foot_xy["right"], axis=0), axis=1)

    rows = []
    for i in range(T):
        rows.append(
            {
                "variant": variant,
                "case": case,
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
                "sim_right_foot_xy_step_m": float(sim_rstep[i]),
                "ref_right_foot_xy_step_m": float(ref_rstep[i]),
                "right_hip_pitch_ctrl_diff": float(rhip_diff[i]),
                "robot_ctrl_linf": float(robot_ctrl_linf[i]),
                "robot_ctrl_l2": float(robot_ctrl_l2[i]),
                "object_ctrl_linf": float(object_ctrl_linf[i]),
                "object_ctrl_l2": float(object_ctrl_l2[i]),
                "qpos_max_abs_diff_vs_ref": float(np.max(np.abs(qpos[i] - qpos_ref[i]))),
                "qvel_max_abs": float(np.max(np.abs(qvel[i]))),
            }
        )

    timeseries_path = RESULTS / f"timeseries_{variant}.csv"
    with timeseries_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    start = int(2.0 * FPS)
    end = min(int(3.6 * FPS) + 1, T)
    post = slice(start, end)
    frame100_145 = slice(100, min(146, T))
    frame115_130 = slice(115, min(131, T))
    frame119_125_steps = slice(120, min(126, T))
    frame120_124 = slice(120, min(125, T))
    pre_end = min(int(2.0 * FPS) + 1, T)

    summary: dict[str, object] = {
        "variant": variant,
        "case": case,
        "override": override,
        "npz_path": str(npz_path.relative_to(REPO)),
        "scene_used": str(scene_used.relative_to(REPO)),
        "T": T,
        "yaw_err_t0017_deg": float(_normalize_angle_deg(e073.yaw_deg(qpos[0, 3:7]) - e073.yaw_deg(qpos_ref[1, 3:7]))),
        "yaw_err_t0033_deg": float(_normalize_angle_deg(e073.yaw_deg(qpos[1, 3:7]) - e073.yaw_deg(qpos_ref[2, 3:7]))),
        "B1_pre_contact_max_foot_z_m": float(max(sim["left_foot_z"][:pre_end].max(), sim["right_foot_z"][:pre_end].max())),
        "B2_single_foot_runs": e073.count_single_foot_runs(sim["left_foot_z"], sim["right_foot_z"]),
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
        "frame100_145_contact_frames_pct": float((sim_total_contacts[frame100_145] > 0).mean() * 100.0),
        "frame100_145_ref_contact_frames_pct": float((ref_total_contacts[frame100_145] > 0).mean() * 100.0),
        "frame100_145_obj_err_max_m": float(obj_err[frame100_145].max()),
        "frame115_130_sim_right_foot_xy_step_sum_m": float(sim_rstep[frame115_130].sum()),
        "frame115_130_ref_right_foot_xy_step_sum_m": float(ref_rstep[frame115_130].sum()),
        "frame119_125_sim_right_foot_xy_step_sum_m": float(sim_rstep[frame119_125_steps].sum()),
        "frame119_125_ref_right_foot_xy_step_sum_m": float(ref_rstep[frame119_125_steps].sum()),
        "frame119_125_sim_right_foot_xy_step_max_m": float(sim_rstep[frame119_125_steps].max()),
        "frame119_125_ref_right_foot_xy_step_max_m": float(ref_rstep[frame119_125_steps].max()),
        "frame120_124_right_hip_pitch_ctrl_diff_mean": float(rhip_diff[frame120_124].mean()),
        "frame120_124_right_hip_pitch_ctrl_diff_abs_max": float(np.abs(rhip_diff[frame120_124]).max()),
    }
    summary.update(contact_mask_stats(override, case, T))
    for key in [k for k in summary if k.endswith("_frame")]:
        frame_time(summary, key, time)

    summary_json = RESULTS / f"eval_summary_{variant}.json"
    summary_csv = RESULTS / f"eval_summary_{variant}.csv"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    e072.plot_timeline(rows, RESULTS / "plots" / f"{variant}_post2_failure_timeline.png")
    return summary


def main() -> None:
    variants = sys.argv[1:] or ["E078A", "E078B"]
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "plots").mkdir(parents=True, exist_ok=True)
    summaries = []
    for variant in variants:
        try:
            summaries.append(evaluate_variant(variant))
        except FileNotFoundError as exc:
            print(f"[SKIP] {exc}")

    if not summaries:
        raise SystemExit("No E078 variant results found.")

    keys = sorted({k for row in summaries for k in row.keys()})
    comparison = RESULTS / "comparison.csv"
    with comparison.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Wrote {comparison}")
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
