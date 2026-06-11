#!/usr/bin/env python3
"""E074 evaluation for post-2s hold/contact experiments."""

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
from diagnose_E068_init_drift import _normalize_angle_deg  # noqa: E402

RESULTS = REPO / "workspace/core4d/results/E074"
BASELINE = REPO / "workspace/core4d/results/E073/eval_summary.json"
SCENE_SNAPSHOT = RESULTS / "scene_snapshot/box023_person1/scene_act.xml"
ACTIVE_SCENE = (
    REPO
    / "example_datasets/processed/core4d/unitree_g1/humanoid_object/box023_person1/scene_act.xml"
)
FPS = 50.0


def load_scene_model() -> tuple[mujoco.MjModel, Path]:
    if SCENE_SNAPSHOT.is_file():
        try:
            return mujoco.MjModel.from_xml_path(str(SCENE_SNAPSHOT)), SCENE_SNAPSHOT
        except Exception:
            text = SCENE_SNAPSHOT.read_text(encoding="utf-8")
            patched = text.replace(
                "../../../../../example_datasets",
                str(REPO / "example_datasets"),
            )
            patched_path = RESULTS / "_scene_act_abs.xml"
            patched_path.write_text(patched, encoding="utf-8")
            try:
                return mujoco.MjModel.from_xml_path(str(patched_path)), patched_path
            except Exception:
                pass
    return mujoco.MjModel.from_xml_path(str(ACTIVE_SCENE)), ACTIVE_SCENE


def scalar_frame_time(summary: dict[str, object], key: str, time: np.ndarray) -> None:
    frame = int(summary[key])
    summary[key.replace("_frame", "_eval_time_s")] = e072.eval_time_or_neg(frame)
    summary[key.replace("_frame", "_npz_time_s")] = e072.npz_time_or_neg(frame, time)


def evaluate_variant(variant: str, model: mujoco.MjModel, scene_used: Path) -> dict[str, object]:
    name = f"{variant}_box023"
    npz_path = RESULTS / f"{name}.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)

    data_npz = np.load(npz_path, allow_pickle=True)
    qpos = e072.flatten_time_major(data_npz["qpos"])
    qvel = e072.flatten_time_major(data_npz["qvel"])
    ctrl = e072.flatten_time_major(data_npz["ctrl"])
    time = e072.flatten_time_major(data_npz["time"]).astype(float)
    opt_steps = np.repeat(data_npz["opt_steps"].reshape(-1), 2)
    qpos_ref, ctrl_ref = e072.load_ref()

    T = min(len(qpos), len(qvel), len(ctrl), len(time), len(opt_steps), len(qpos_ref), len(ctrl_ref))
    qpos, qvel, ctrl, time, opt_steps = qpos[:T], qvel[:T], ctrl[:T], time[:T], opt_steps[:T]
    qpos_ref, ctrl_ref = qpos_ref[:T], ctrl_ref[:T]

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
    if ctrl_diff.shape[1] > 29:
        object_ctrl_linf = np.max(np.abs(ctrl_diff[:, 29:]), axis=1)
        object_ctrl_l2 = np.linalg.norm(ctrl_diff[:, 29:], axis=1)
    else:
        object_ctrl_linf = np.zeros(T)
        object_ctrl_l2 = np.zeros(T)

    rows = []
    for i in range(T):
        rows.append(
            {
                "variant": variant,
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
    frame_window = slice(100, min(146, T))
    pre_end = min(int(2.0 * FPS) + 1, T)

    summary: dict[str, object] = {
        "variant": variant,
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
        "frame100_145_contact_frames_pct": float((sim_total_contacts[frame_window] > 0).mean() * 100.0),
        "frame100_145_ref_contact_frames_pct": float((ref_total_contacts[frame_window] > 0).mean() * 100.0),
        "frame100_145_obj_err_max_m": float(obj_err[frame_window].max()),
    }
    for key in [k for k in summary if k.endswith("_frame")]:
        scalar_frame_time(summary, key, time)

    if BASELINE.is_file():
        base = json.loads(BASELINE.read_text(encoding="utf-8"))
        for key in (
            "yaw_err_t0017_deg",
            "yaw_err_t0033_deg",
            "B1_pre_contact_max_foot_z_m",
            "first_sim_zero_contact_frame",
            "post2_sim_contact_frames_pct",
            "post2_obj_err_max_m",
            "post2_pelvis_z_min_m",
            "frame100_145_contact_frames_pct",
        ):
            summary[f"E073_{key}"] = base.get(key, "")

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
    variants = sys.argv[1:] or ["E074A", "E074C"]
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "plots").mkdir(parents=True, exist_ok=True)
    model, scene_used = load_scene_model()
    summaries = []
    for variant in variants:
        try:
            summaries.append(evaluate_variant(variant, model, scene_used))
        except FileNotFoundError as exc:
            print(f"[SKIP] {exc}")

    if summaries:
        keys = sorted({k for row in summaries for k in row.keys()})
        comparison = RESULTS / "comparison.csv"
        with comparison.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(summaries)
        print(f"Wrote {comparison}")
        print(json.dumps(summaries, indent=2, sort_keys=True))
    else:
        raise SystemExit("No E074 variant results found.")


if __name__ == "__main__":
    main()
