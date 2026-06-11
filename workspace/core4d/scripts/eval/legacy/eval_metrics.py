#!/usr/bin/env python3
"""Comprehensive retargeting evaluation metrics.

Computes metrics from DynaRetarget, OmniRetarget, Harmanoid, and SPIDER
for dual_humanoid_object box-carrying retargeting results.

Usage:
    python workspace/core4d/scripts/eval/eval_metrics.py \
        --results workspace/core4d/results/E017d_box025_dual_soft2connect.npz \
                  workspace/core4d/results/E018d2_box025_dual_taskspace_basepos15.npz \
        [--scene <path/to/scene.xml>] [--ref <path/to/trajectory_kinematic_dual.npz>]
        [--csv workspace/core4d/results/metrics.csv]

Metrics computed:
  SPIDER:       obj_pos_err, obj_quat_err, success (pos<10cm & rot<0.5rad)
  DynaRetarget: success (pos<10cm & rot<25°), smoothness (normalized joint acc)
  Harmanoid:    g-MPJPE (mm), E_Acc (rad/s²), E_Vel (rad/s)
  OmniRetarget: foot_skating_duration, foot_skating_max_vel, penetration (contact depth)
  Physical:     R1/R2 stability, obj_z stats
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R


# ─── helpers ────────────────────────────────────────────────────────────────────

def quat_angular_distance(q1_wxyz: np.ndarray, q2_wxyz: np.ndarray) -> float:
    """Angular distance between two wxyz quaternions, in radians."""
    r1 = R.from_quat([q1_wxyz[1], q1_wxyz[2], q1_wxyz[3], q1_wxyz[0]])
    r2 = R.from_quat([q2_wxyz[1], q2_wxyz[2], q2_wxyz[3], q2_wxyz[0]])
    return (r1.inv() * r2).magnitude()


def compute_body_xpos(
    model: mujoco.MjModel, data: mujoco.MjData, qpos_seq: np.ndarray
) -> np.ndarray:
    """FK: (T, nq) → (T, nbody, 3)."""
    T = qpos_seq.shape[0]
    xpos = np.zeros((T, model.nbody, 3))
    for t in range(T):
        data.qpos[:] = qpos_seq[t]
        mujoco.mj_kinematics(model, data)
        xpos[t] = data.xpos.copy()
    return xpos


def interpolate_ref(qpos_ref: np.ndarray, ref_dt: float, sim_dt: float, T_target: int) -> np.ndarray:
    """Interpolate reference trajectory to match simulation timestep."""
    T_ref = qpos_ref.shape[0]
    t_ref = np.arange(T_ref) * ref_dt
    t_sim = np.arange(T_target) * sim_dt
    interp_fn = interp1d(t_ref, qpos_ref, axis=0, bounds_error=False, fill_value=qpos_ref[-1])
    return interp_fn(np.clip(t_sim, 0, t_ref[-1]))


# ─── metric functions ───────────────────────────────────────────────────────────

def metric_object_tracking(qpos_sim: np.ndarray, qpos_ref: np.ndarray) -> dict:
    """Object position and rotation tracking error (SPIDER/DynaRetarget)."""
    # qpos layout: ...obj_pos(3)...obj_quat(4) at indices 72:79
    obj_pos_sim = qpos_sim[:, 72:75]
    obj_pos_ref = qpos_ref[:, 72:75]
    obj_quat_sim = qpos_sim[:, 75:79]  # wxyz
    obj_quat_ref = qpos_ref[:, 75:79]

    pos_err_per_frame = np.linalg.norm(obj_pos_sim - obj_pos_ref, axis=1)
    rot_err_per_frame = np.array([
        quat_angular_distance(obj_quat_sim[t], obj_quat_ref[t])
        for t in range(len(qpos_sim))
    ])

    return {
        "obj_pos_err_mean_m": pos_err_per_frame.mean(),
        "obj_pos_err_max_m": pos_err_per_frame.max(),
        "obj_rot_err_mean_deg": np.degrees(rot_err_per_frame.mean()),
        "obj_rot_err_max_deg": np.degrees(rot_err_per_frame.max()),
        "success_spider": pos_err_per_frame.mean() < 0.10 and rot_err_per_frame.mean() < 0.5,
        "success_dynaretarget": pos_err_per_frame.mean() < 0.10 and np.degrees(rot_err_per_frame.mean()) < 25.0,
    }


def metric_body_mpjpe(
    xpos_sim: np.ndarray, xpos_ref: np.ndarray, body_ids: list[int]
) -> dict:
    """Global MPJPE (Harmanoid g-MPJPE) on selected bodies, in mm."""
    # xpos: (T, nbody, 3)
    err = np.linalg.norm(xpos_sim[:, body_ids] - xpos_ref[:, body_ids], axis=-1)  # (T, K)
    per_frame = err.mean(axis=1)  # mean over bodies
    return {
        "g_mpjpe_mm": per_frame.mean() * 1000,
        "g_mpjpe_max_mm": per_frame.max() * 1000,
    }


def metric_joint_dynamics(qpos_sim: np.ndarray, qpos_ref: np.ndarray, dt: float) -> dict:
    """E_Acc, E_Vel (Harmanoid), smoothness (DynaRetarget)."""
    # R1 joints: indices 7:36 (29 joints)
    joints_sim = qpos_sim[:, 7:36]
    joints_ref = qpos_ref[:, 7:36]

    # Velocity
    vel_sim = np.diff(joints_sim, axis=0) / dt
    vel_ref = np.diff(joints_ref, axis=0) / dt
    e_vel = np.linalg.norm(vel_sim - vel_ref, axis=1).mean()

    # Acceleration
    acc_sim = np.diff(joints_sim, n=2, axis=0) / (dt ** 2)
    acc_ref = np.diff(joints_ref, n=2, axis=0) / (dt ** 2)
    e_acc = np.linalg.norm(acc_sim - acc_ref, axis=1).mean()

    # DynaRetarget smoothness: total acceleration of sim, normalized by ref
    smoothness_sim = np.abs(acc_sim).sum()
    smoothness_ref = np.abs(acc_ref).sum()
    smoothness_normalized = smoothness_sim / max(smoothness_ref, 1e-6)

    return {
        "e_vel_rad_s": e_vel,
        "e_acc_rad_s2": e_acc,
        "smoothness_norm": smoothness_normalized,
    }


def metric_foot_skating(
    xpos_sim: np.ndarray, foot_ids: list[int], dt: float,
    ground_z_thresh: float = 0.10, skate_vel_thresh: float = 0.01
) -> dict:
    """Foot skating (OmniRetarget): duration and max velocity when foot on ground."""
    foot_pos = xpos_sim[:, foot_ids]  # (T, nfeet, 3)
    foot_vel = np.diff(foot_pos, axis=0) / dt  # (T-1, nfeet, 3)
    foot_speed_xy = np.linalg.norm(foot_vel[:, :, :2], axis=-1)  # (T-1, nfeet)

    on_ground = foot_pos[:-1, :, 2] < ground_z_thresh  # (T-1, nfeet)
    skating = on_ground & (foot_speed_xy > skate_vel_thresh)

    total_ground_frames = on_ground.sum()
    skating_frames = skating.sum()
    duration = skating_frames / max(total_ground_frames, 1)
    max_vel = (foot_speed_xy * on_ground).max() * 100  # cm/s

    return {
        "foot_skate_duration": duration,
        "foot_skate_max_vel_cm_s": max_vel,
    }


def metric_stability(qpos_sim: np.ndarray) -> dict:
    """Robot pelvis stability."""
    T = qpos_sim.shape[0]
    r1_pz = qpos_sim[:, 2]
    r2_pz = qpos_sim[:, 38]
    return {
        "r1_stable_pct": (r1_pz >= 0.50).sum() / T * 100,
        "r2_stable_pct": (r2_pz >= 0.50).sum() / T * 100,
        "r1_pz_min": r1_pz.min(),
        "r2_pz_min": r2_pz.min(),
    }


def metric_object_height(qpos_sim: np.ndarray, qpos_ref: np.ndarray) -> dict:
    """Object z trajectory quality."""
    T = qpos_sim.shape[0]
    obj_z = qpos_sim[:, 74]
    obj_z_ref = qpos_ref[:, 74]
    return {
        "obj_z_max": obj_z.max(),
        "obj_z_mean": obj_z.mean(),
        "obj_z_above_040_pct": (obj_z > 0.40).sum() / T * 100,
        "obj_z_ref_max": obj_z_ref.max(),
    }


# ─── main evaluation ────────────────────────────────────────────────────────────

def evaluate_single(
    npz_path: str,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_ref_raw: np.ndarray,
    body_ids_mpjpe: list[int],
    foot_ids: list[int],
    ref_dt: float = 0.0333,
    sim_dt: float = 0.01667,
) -> dict:
    """Evaluate a single result NPZ against reference."""
    run_data = np.load(npz_path)
    qpos_run = run_data["qpos"]
    if qpos_run.ndim == 3:
        qpos_run = qpos_run.reshape(-1, qpos_run.shape[-1])
    T = qpos_run.shape[0]

    # Interpolate reference
    qpos_ref = interpolate_ref(qpos_ref_raw, ref_dt, sim_dt, T)

    # FK for body positions
    xpos_sim = compute_body_xpos(model, data, qpos_run)
    xpos_ref = compute_body_xpos(model, data, qpos_ref)

    # Compute all metrics
    metrics = {"run": Path(npz_path).stem, "T_frames": T}
    metrics.update(metric_object_tracking(qpos_run, qpos_ref))
    metrics.update(metric_body_mpjpe(xpos_sim, xpos_ref, body_ids_mpjpe))
    metrics.update(metric_joint_dynamics(qpos_run, qpos_ref, sim_dt))
    metrics.update(metric_foot_skating(xpos_sim, foot_ids, sim_dt))
    metrics.update(metric_stability(qpos_run))
    metrics.update(metric_object_height(qpos_run, qpos_ref))
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Comprehensive retargeting evaluation")
    parser.add_argument("--results", nargs="+", required=True, help="NPZ result files to evaluate")
    parser.add_argument("--scene", default="example_datasets/processed/core4d/unitree_g1/dual_humanoid_object/box025_person1/scene_dual_robot_connect2.xml")
    parser.add_argument("--ref", default="example_datasets/processed/core4d/unitree_g1/dual_humanoid_object/box025_person1/0/trajectory_kinematic_dual.npz")
    parser.add_argument("--ref-dt", type=float, default=0.0333)
    parser.add_argument("--sim-dt", type=float, default=0.01667)
    parser.add_argument("--csv", default="", help="Output CSV path (optional)")
    args = parser.parse_args()

    # Load model
    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)

    # Load reference
    ref_data = np.load(args.ref)
    qpos_ref_raw = ref_data["qpos"]

    # Resolve body IDs
    mpjpe_names = [
        "pelvis", "torso_link",
        "left_wrist_yaw_link", "right_wrist_yaw_link",
        "left_ankle_roll_link", "right_ankle_roll_link",
        "r2_pelvis", "r2_torso_link",
        "r2_left_wrist_yaw_link", "r2_right_wrist_yaw_link",
        "r2_left_ankle_roll_link", "r2_right_ankle_roll_link",
    ]
    foot_names = [
        "left_ankle_roll_link", "right_ankle_roll_link",
        "r2_left_ankle_roll_link", "r2_right_ankle_roll_link",
    ]
    body_ids_mpjpe = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in mpjpe_names]
    foot_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in foot_names]

    # Evaluate each result
    all_metrics = []
    for npz_path in args.results:
        if not Path(npz_path).exists():
            print(f"SKIP (not found): {npz_path}", file=sys.stderr)
            continue
        try:
            m = evaluate_single(npz_path, model, data, qpos_ref_raw, body_ids_mpjpe, foot_ids, args.ref_dt, args.sim_dt)
            all_metrics.append(m)
        except Exception as e:
            print(f"ERROR: {npz_path}: {e}", file=sys.stderr)

    if not all_metrics:
        print("No results to report.", file=sys.stderr)
        return

    # Print formatted table
    # Select key columns for display
    display_cols = [
        ("run", "Run", "s"),
        ("obj_pos_err_mean_m", "ObjPos(m)", ".3f"),
        ("obj_rot_err_mean_deg", "ObjRot(°)", ".1f"),
        ("g_mpjpe_mm", "gMPJPE(mm)", ".0f"),
        ("e_acc_rad_s2", "E_Acc", ".0f"),
        ("e_vel_rad_s", "E_Vel", ".1f"),
        ("smoothness_norm", "Smooth", ".2f"),
        ("foot_skate_duration", "FtSkate", ".3f"),
        ("foot_skate_max_vel_cm_s", "FtVel(cm/s)", ".0f"),
        ("r1_stable_pct", "R1%", ".0f"),
        ("r2_stable_pct", "R2%", ".0f"),
        ("obj_z_max", "ObjZmax", ".3f"),
        ("obj_z_above_040_pct", "Oz>0.4%", ".0f"),
        ("success_dynaretarget", "Succ", "s"),
    ]

    # Header
    header_parts = []
    for key, label, _ in display_cols:
        header_parts.append(f"{label:>12}")
    print(" | ".join(header_parts))
    print("-" * (15 * len(display_cols)))

    # Rows
    for m in all_metrics:
        row_parts = []
        for key, label, fmt in display_cols:
            val = m.get(key, "")
            if fmt == "s":
                if isinstance(val, bool):
                    val = "✅" if val else "❌"
                row_parts.append(f"{str(val):>12}")
            else:
                row_parts.append(f"{val:>12{fmt}}")
        print(" | ".join(row_parts))

    # CSV output
    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
            writer.writeheader()
            writer.writerows(all_metrics)
        print(f"\nCSV saved: {csv_path}")


if __name__ == "__main__":
    main()
