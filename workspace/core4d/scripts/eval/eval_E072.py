#!/usr/bin/env python3
"""E072 replay diagnosis for box023 post-2s hold/place failure."""

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
DEBUG_DIR = REPO / "workspace/core4d/scripts/debug"
if str(DEBUG_DIR) not in sys.path:
    sys.path.insert(0, str(DEBUG_DIR))

from diagnose_E068_init_drift import _build_config, _convert_scene_act_ref  # noqa: E402
from spider.io import load_data  # noqa: E402

RESULTS_E071 = REPO / "workspace/core4d/results/E071"
RESULTS_E072 = REPO / "workspace/core4d/results/E072"
NPZ_PATH = RESULTS_E071 / "E071W02_box023.npz"
SCENE_SNAPSHOT = RESULTS_E071 / "scene_snapshot/box023_person1/scene_act.xml"
ACTIVE_SCENE = (
    REPO
    / "example_datasets/processed/core4d/unitree_g1/humanoid_object/box023_person1/scene_act.xml"
)
FPS = 50.0
POST2_START_S = 2.0
POST2_END_S = 3.6
HAND_GAP_FAIL_M = 0.10
OBJ_FAIL_M = 0.25
FALL_PELVIS_Z_M = 0.45
ROBOT_CTRL_JUMP_RAD = 0.50
OBJECT_CTRL_JUMP = 0.10


def flatten_time_major(x: np.ndarray) -> np.ndarray:
    if x.ndim >= 3:
        return x.reshape(-1, x.shape[-1])
    if x.ndim == 2 and x.shape[1] == 1:
        return x.reshape(-1)
    if x.ndim == 2:
        return x.reshape(-1)
    return x


def load_scene_model() -> tuple[mujoco.MjModel, Path]:
    try:
        return mujoco.MjModel.from_xml_path(str(SCENE_SNAPSHOT)), SCENE_SNAPSHOT
    except Exception as exc:
        # Snapshot XML keeps the original relative mesh path, which may not be
        # valid from results/. Patch only the mesh prefix into an E072-local copy.
        text = SCENE_SNAPSHOT.read_text(encoding="utf-8")
        patched = text.replace(
            "../../../../../example_datasets",
            str(REPO / "example_datasets"),
        )
        patched_path = RESULTS_E072 / "_scene_act_abs.xml"
        RESULTS_E072.mkdir(parents=True, exist_ok=True)
        patched_path.write_text(patched, encoding="utf-8")
        try:
            return mujoco.MjModel.from_xml_path(str(patched_path)), patched_path
        except Exception:
            print(
                f"WARNING: failed to load snapshot scene ({exc}); falling back to active scene",
                file=sys.stderr,
            )
            return mujoco.MjModel.from_xml_path(str(ACTIVE_SCENE)), ACTIVE_SCENE


def load_ref() -> tuple[np.ndarray, np.ndarray]:
    cfg = _build_config("core4d_e062_box023", "box023_person1", "cpu", 4)
    qpos_ref, qvel_ref, ctrl_ref, _contact, _contact_pos = load_data(cfg, cfg.data_path)
    qpos_ref, _qvel_ref, ctrl_ref = _convert_scene_act_ref(
        cfg, qpos_ref, qvel_ref, ctrl_ref
    )
    return qpos_ref.detach().cpu().numpy(), ctrl_ref.detach().cpu().numpy()


def name2id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise ValueError(f"Missing MuJoCo {obj_type} name: {name}")
    return idx


def box_surface_distance(point: np.ndarray, obj_pos: np.ndarray, obj_mat: np.ndarray, half: np.ndarray) -> float:
    local = obj_mat.T @ (point - obj_pos)
    clamped = np.clip(local, -half, half)
    return float(np.linalg.norm(local - clamped))


def hand_object_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    left_geom: int,
    right_geom: int,
    object_geom: int,
) -> tuple[int, int, float, float]:
    left_count = 0
    right_count = 0
    left_min_dist = np.inf
    right_min_dist = np.inf
    for i in range(data.ncon):
        con = data.contact[i]
        pair = {int(con.geom1), int(con.geom2)}
        if object_geom not in pair:
            continue
        if left_geom in pair:
            left_count += 1
            left_min_dist = min(left_min_dist, float(con.dist))
        if right_geom in pair:
            right_count += 1
            right_min_dist = min(right_min_dist, float(con.dist))
    if not np.isfinite(left_min_dist):
        left_min_dist = np.nan
    if not np.isfinite(right_min_dist):
        right_min_dist = np.nan
    return left_count, right_count, left_min_dist, right_min_dist


def object_floor_contacts(data: mujoco.MjData, object_geom: int, floor_geom: int) -> int:
    count = 0
    for i in range(data.ncon):
        con = data.contact[i]
        pair = {int(con.geom1), int(con.geom2)}
        if object_geom in pair and floor_geom in pair:
            count += 1
    return count


def replay_metrics(model: mujoco.MjModel, qpos: np.ndarray) -> dict[str, np.ndarray]:
    data = mujoco.MjData(model)
    left_palm = name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_palm")
    right_palm = name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_palm")
    left_foot = name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    right_foot = name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
    pelvis = name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    obj = name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    left_geom = name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "lh")
    right_geom = name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "rh")
    object_geom = name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_collision")
    floor_geom = name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    half = model.geom_size[object_geom, :3].copy()

    T = len(qpos)
    out = {
        "obj_pos": np.zeros((T, 3)),
        "pelvis_pos": np.zeros((T, 3)),
        "left_palm_pos": np.zeros((T, 3)),
        "right_palm_pos": np.zeros((T, 3)),
        "left_foot_z": np.zeros(T),
        "right_foot_z": np.zeros(T),
        "left_sdf_m": np.zeros(T),
        "right_sdf_m": np.zeros(T),
        "left_contact_count": np.zeros(T, dtype=int),
        "right_contact_count": np.zeros(T, dtype=int),
        "left_contact_min_dist": np.full(T, np.nan),
        "right_contact_min_dist": np.full(T, np.nan),
        "object_floor_contact_count": np.zeros(T, dtype=int),
        "total_contact_count": np.zeros(T, dtype=int),
    }

    for t, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        obj_pos = data.xpos[obj].copy()
        obj_mat = data.xmat[obj].reshape(3, 3).copy()
        left_pos = data.site_xpos[left_palm].copy()
        right_pos = data.site_xpos[right_palm].copy()
        lc, rc, lmin, rmin = hand_object_contacts(
            model, data, left_geom, right_geom, object_geom
        )

        out["obj_pos"][t] = obj_pos
        out["pelvis_pos"][t] = data.xpos[pelvis].copy()
        out["left_palm_pos"][t] = left_pos
        out["right_palm_pos"][t] = right_pos
        out["left_foot_z"][t] = data.site_xpos[left_foot, 2]
        out["right_foot_z"][t] = data.site_xpos[right_foot, 2]
        out["left_sdf_m"][t] = box_surface_distance(left_pos, obj_pos, obj_mat, half)
        out["right_sdf_m"][t] = box_surface_distance(right_pos, obj_pos, obj_mat, half)
        out["left_contact_count"][t] = lc
        out["right_contact_count"][t] = rc
        out["left_contact_min_dist"][t] = lmin
        out["right_contact_min_dist"][t] = rmin
        out["object_floor_contact_count"][t] = object_floor_contacts(
            data, object_geom, floor_geom
        )
        out["total_contact_count"][t] = data.ncon
    return out


def first_after(mask: np.ndarray, start: int) -> int:
    idx = np.where(mask[start:])[0]
    return int(idx[0] + start) if len(idx) else -1


def eval_time_or_neg(frame: int) -> float:
    return float(frame / FPS) if frame >= 0 else -1.0


def npz_time_or_neg(frame: int, time: np.ndarray) -> float:
    return float(time[frame]) if frame >= 0 else -1.0


def classify(summary: dict[str, float | int]) -> str:
    hold_t = float(summary["first_sim_min_hand_sdf_gt_10cm_eval_time_s"])
    obj_t = float(summary["first_obj_err_gt_25cm_eval_time_s"])
    fall_t = float(summary["first_pelvis_z_lt_45cm_eval_time_s"])
    ctrl_t = float(summary["first_robot_ctrl_linf_gt_0p5_eval_time_s"])
    ref_hold_t = float(summary["first_ref_min_hand_sdf_gt_10cm_eval_time_s"])

    if ref_hold_t >= 0 and ref_hold_t <= 2.4:
        return "ref_hand_geometry_degrades_early"
    if hold_t >= 0 and (fall_t < 0 or hold_t <= fall_t - 0.3):
        return "hold_contact_first"
    if ctrl_t >= 0 and (hold_t < 0 or ctrl_t <= hold_t + 0.1) and ctrl_t <= 2.4:
        return "ctrl_divergence_first"
    if fall_t >= 0 and (obj_t < 0 or fall_t <= obj_t + 0.3):
        return "stability_first_or_mixed"
    return "mixed_unclear"


def plot_timeline(rows: list[dict[str, float | int | str]], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time = np.array([float(r["eval_time_s"]) for r in rows])
    sim_min = np.array([float(r["sim_min_hand_sdf_m"]) for r in rows])
    ref_min = np.array([float(r["ref_min_hand_sdf_m"]) for r in rows])
    obj_err = np.array([float(r["obj_err_m"]) for r in rows])
    pelvis_z = np.array([float(r["sim_pelvis_z_m"]) for r in rows])
    robot_ctrl = np.array([float(r["robot_ctrl_linf"]) for r in rows])

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    for ax in axes:
        ax.axvspan(2.0, 3.6, color="#f4c542", alpha=0.12)
        ax.grid(True, alpha=0.3)

    axes[0].plot(time, sim_min * 100, label="sim min hand SDF", color="#d62728")
    axes[0].plot(time, ref_min * 100, label="ref min hand SDF", color="#1f77b4")
    axes[0].axhline(HAND_GAP_FAIL_M * 100, color="black", ls="--", lw=1)
    axes[0].set_ylabel("hand-box gap (cm)")
    axes[0].legend(loc="upper left")

    axes[1].plot(time, obj_err * 100, color="#9467bd")
    axes[1].axhline(OBJ_FAIL_M * 100, color="black", ls="--", lw=1)
    axes[1].set_ylabel("obj err (cm)")

    axes[2].plot(time, pelvis_z, color="#2ca02c")
    axes[2].axhline(FALL_PELVIS_Z_M, color="black", ls="--", lw=1)
    axes[2].set_ylabel("pelvis z (m)")

    axes[3].plot(time, robot_ctrl, color="#ff7f0e")
    axes[3].axhline(ROBOT_CTRL_JUMP_RAD, color="black", ls="--", lw=1)
    axes[3].set_ylabel("robot ctrl Linf")
    axes[3].set_xlabel("time (s)")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RESULTS_E072.mkdir(parents=True, exist_ok=True)
    data_npz = np.load(NPZ_PATH, allow_pickle=True)
    qpos = flatten_time_major(data_npz["qpos"])
    qvel = flatten_time_major(data_npz["qvel"])
    ctrl = flatten_time_major(data_npz["ctrl"])
    time = flatten_time_major(data_npz["time"]).astype(float)
    opt_steps = np.repeat(data_npz["opt_steps"].reshape(-1), 2)
    qpos_ref, ctrl_ref = load_ref()

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
    sim = replay_metrics(model, qpos)
    ref = replay_metrics(model, qpos_ref)

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

    rows: list[dict[str, float | int | str]] = []
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
                "sim_object_x_m": float(sim["obj_pos"][i, 0]),
                "sim_object_y_m": float(sim["obj_pos"][i, 1]),
                "sim_object_z_m": float(sim["obj_pos"][i, 2]),
                "ref_object_x_m": float(ref["obj_pos"][i, 0]),
                "ref_object_y_m": float(ref["obj_pos"][i, 1]),
                "ref_object_z_m": float(ref["obj_pos"][i, 2]),
                "sim_left_sdf_m": float(sim["left_sdf_m"][i]),
                "sim_right_sdf_m": float(sim["right_sdf_m"][i]),
                "sim_min_hand_sdf_m": float(sim_min_hand_sdf[i]),
                "ref_left_sdf_m": float(ref["left_sdf_m"][i]),
                "ref_right_sdf_m": float(ref["right_sdf_m"][i]),
                "ref_min_hand_sdf_m": float(ref_min_hand_sdf[i]),
                "sim_left_contact_count": int(sim["left_contact_count"][i]),
                "sim_right_contact_count": int(sim["right_contact_count"][i]),
                "sim_total_contact_count": int(sim_total_contacts[i]),
                "sim_object_floor_contact_count": int(sim["object_floor_contact_count"][i]),
                "sim_all_contact_count": int(sim["total_contact_count"][i]),
                "ref_left_contact_count": int(ref["left_contact_count"][i]),
                "ref_right_contact_count": int(ref["right_contact_count"][i]),
                "ref_total_contact_count": int(ref_total_contacts[i]),
                "ref_object_floor_contact_count": int(ref["object_floor_contact_count"][i]),
                "ref_all_contact_count": int(ref["total_contact_count"][i]),
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

    timeseries_path = RESULTS_E072 / "timeseries.csv"
    with timeseries_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    start = int(POST2_START_S * FPS)
    end = min(int(POST2_END_S * FPS) + 1, T)
    post = slice(start, end)
    summary: dict[str, float | int | str] = {
        "npz_path": str(NPZ_PATH.relative_to(REPO)),
        "scene_used": str(scene_used.relative_to(REPO)),
        "T": T,
        "post_window_start_s": POST2_START_S,
        "post_window_end_s": POST2_END_S,
        "time_note": "eval_time_s uses frame/50 to match prior E071 metrics; npz_time_s is also written because saved rollout time advances at about 60 Hz.",
        "first_sim_min_hand_sdf_gt_10cm_frame": first_after(
            sim_min_hand_sdf > HAND_GAP_FAIL_M, start
        ),
        "first_ref_min_hand_sdf_gt_10cm_frame": first_after(
            ref_min_hand_sdf > HAND_GAP_FAIL_M, start
        ),
        "first_obj_err_gt_25cm_frame": first_after(obj_err > OBJ_FAIL_M, start),
        "first_pelvis_z_lt_45cm_frame": first_after(
            sim["pelvis_pos"][:, 2] < FALL_PELVIS_Z_M, start
        ),
        "first_sim_zero_contact_frame": first_after(sim_total_contacts == 0, start),
        "first_ref_zero_contact_frame": first_after(ref_total_contacts == 0, start),
        "first_robot_ctrl_linf_gt_0p5_frame": first_after(
            robot_ctrl_linf > ROBOT_CTRL_JUMP_RAD, start
        ),
        "first_object_ctrl_linf_gt_0p1_frame": first_after(
            object_ctrl_linf > OBJECT_CTRL_JUMP, start
        ),
        "post2_sim_min_hand_sdf_mean_m": float(sim_min_hand_sdf[post].mean()),
        "post2_sim_min_hand_sdf_max_m": float(sim_min_hand_sdf[post].max()),
        "post2_ref_min_hand_sdf_mean_m": float(ref_min_hand_sdf[post].mean()),
        "post2_ref_min_hand_sdf_max_m": float(ref_min_hand_sdf[post].max()),
        "post2_obj_err_mean_m": float(obj_err[post].mean()),
        "post2_obj_err_max_m": float(obj_err[post].max()),
        "post2_pelvis_z_min_m": float(sim["pelvis_pos"][post, 2].min()),
        "post2_sim_contact_frames_pct": float((sim_total_contacts[post] > 0).mean() * 100.0),
        "post2_ref_contact_frames_pct": float((ref_total_contacts[post] > 0).mean() * 100.0),
        "post2_robot_ctrl_linf_max": float(robot_ctrl_linf[post].max()),
        "post2_object_ctrl_linf_max": float(object_ctrl_linf[post].max()),
    }
    for key in list(summary.keys()):
        if key.endswith("_frame"):
            summary[key.replace("_frame", "_eval_time_s")] = eval_time_or_neg(
                int(summary[key])
            )
            summary[key.replace("_frame", "_npz_time_s")] = npz_time_or_neg(
                int(summary[key]), time
            )
    summary["failure_classification"] = classify(summary)

    summary_path = RESULTS_E072 / "diagnosis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    contact_summary_path = RESULTS_E072 / "contact_summary.csv"
    contact_rows = [
        {
            "series": "sim",
            "post2_min_hand_sdf_mean_m": summary["post2_sim_min_hand_sdf_mean_m"],
            "post2_min_hand_sdf_max_m": summary["post2_sim_min_hand_sdf_max_m"],
            "post2_contact_frames_pct": summary["post2_sim_contact_frames_pct"],
        },
        {
            "series": "ref",
            "post2_min_hand_sdf_mean_m": summary["post2_ref_min_hand_sdf_mean_m"],
            "post2_min_hand_sdf_max_m": summary["post2_ref_min_hand_sdf_max_m"],
            "post2_contact_frames_pct": summary["post2_ref_contact_frames_pct"],
        },
    ]
    with contact_summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(contact_rows[0].keys()))
        writer.writeheader()
        writer.writerows(contact_rows)

    plot_timeline(rows, RESULTS_E072 / "plots/post2_failure_timeline.png")

    print(f"Wrote {timeseries_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {contact_summary_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
