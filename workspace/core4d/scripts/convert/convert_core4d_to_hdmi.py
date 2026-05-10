#!/usr/bin/env python3
"""Convert CORE4D trajectory to HDMI motion.npz format.

HDMI expects: body-level world coordinates at 50fps for 28 robot bodies + 1 object body.
CORE4D has: MuJoCo qpos at 30fps with freejoint pelvis + 29 joints + freejoint object.

Conversion: FK per frame → body world coords → upsample 30→50fps → finite-diff velocities.

Usage:
    uv run workspace/core4d/scripts/convert/convert_core4d_to_hdmi.py \
        --case box023_person1 --output-dir /path/to/output
"""

import argparse
import json
import os

import mujoco
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

BASE = "example_datasets/processed/core4d/unitree_g1/humanoid_object"

# HDMI body names (28 robot bodies + object, NO waist_yaw/waist_roll)
HDMI_BODY_NAMES = [
    "pelvis",
    "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
    "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
    "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
    "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link",
    "left_elbow_link", "left_wrist_roll_link", "left_wrist_pitch_link", "left_wrist_yaw_link",
    "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link",
    "right_elbow_link", "right_wrist_roll_link", "right_wrist_pitch_link", "right_wrist_yaw_link",
]

HDMI_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

CORE4D_FPS = 30.0
HDMI_FPS = 50.0
OBJECT_BODY_NAME = "suitcase"  # HDMI convention


def run_fk(model: mujoco.MjModel, qpos_all: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run forward kinematics for all frames. Returns (T, nbody, 3) pos and (T, nbody, 4) quat."""
    data = mujoco.MjData(model)
    T = qpos_all.shape[0]
    nbody = model.nbody
    all_pos = np.zeros((T, nbody, 3))
    all_quat = np.zeros((T, nbody, 4))

    for t in range(T):
        data.qpos[:] = qpos_all[t]
        mujoco.mj_forward(model, data)
        all_pos[t] = data.xpos.copy()
        all_quat[t] = data.xquat.copy()  # wxyz

    return all_pos, all_quat


def build_body_index_map(model: mujoco.MjModel) -> list[int]:
    """Map HDMI body names → MuJoCo body indices."""
    all_names = HDMI_BODY_NAMES + [OBJECT_BODY_NAME]
    mj_name_to_id = {}
    for i in range(model.nbody):
        name = model.body(i).name
        mj_name_to_id[name] = i
    # Map "suitcase" to "object" (CORE4D naming)
    mj_name_to_id["suitcase"] = mj_name_to_id.get("object", -1)

    indices = []
    for name in all_names:
        idx = mj_name_to_id.get(name, -1)
        if idx < 0:
            raise ValueError(f"Body '{name}' not found in MuJoCo model. Available: {list(mj_name_to_id.keys())}")
        indices.append(idx)
    return indices


def upsample_pos(pos: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    """Upsample position arrays linearly. (T_src, ...) → (T_dst, ...)."""
    T_src = pos.shape[0]
    t_src = np.arange(T_src) / src_fps
    T_dst = int(np.round(t_src[-1] * dst_fps)) + 1
    t_dst = np.arange(T_dst) / dst_fps

    orig_shape = pos.shape[1:]
    pos_flat = pos.reshape(T_src, -1)
    interp_fn = interp1d(t_src, pos_flat, axis=0, kind="linear", fill_value="extrapolate")
    result = interp_fn(t_dst).reshape(T_dst, *orig_shape)
    return result


def upsample_quat(quat_wxyz: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    """Upsample quaternion arrays with SLERP. (T_src, N, 4) wxyz → (T_dst, N, 4) wxyz."""
    T_src, N, _ = quat_wxyz.shape
    t_src = np.arange(T_src) / src_fps
    T_dst = int(np.round(t_src[-1] * dst_fps)) + 1
    t_dst = np.arange(T_dst) / dst_fps

    result = np.zeros((T_dst, N, 4))
    for b in range(N):
        # Convert wxyz → xyzw for scipy
        q_xyzw = quat_wxyz[:, b, [1, 2, 3, 0]]
        rotations = Rotation.from_quat(q_xyzw)
        slerp = Slerp(t_src, rotations)
        interp_rot = slerp(t_dst)
        q_interp_xyzw = interp_rot.as_quat()
        # Convert xyzw → wxyz
        result[:, b] = q_interp_xyzw[:, [3, 0, 1, 2]]

    return result


def finite_diff_vel(pos: np.ndarray, dt: float) -> np.ndarray:
    """Compute velocity by central finite difference. (T, ...) → (T, ...)."""
    vel = np.zeros_like(pos)
    vel[1:-1] = (pos[2:] - pos[:-2]) / (2 * dt)
    vel[0] = (pos[1] - pos[0]) / dt
    vel[-1] = (pos[-1] - pos[-2]) / dt
    return vel


def quat_angular_vel(quat_wxyz: np.ndarray, dt: float) -> np.ndarray:
    """Compute angular velocity from quaternion sequence. (T, N, 4) wxyz → (T, N, 3)."""
    T, N, _ = quat_wxyz.shape
    ang_vel = np.zeros((T, N, 3))

    for b in range(N):
        q_xyzw = quat_wxyz[:, b, [1, 2, 3, 0]]
        for t in range(1, T):
            r_prev = Rotation.from_quat(q_xyzw[t - 1])
            r_curr = Rotation.from_quat(q_xyzw[t])
            r_diff = r_prev.inv() * r_curr
            rotvec = r_diff.as_rotvec()
            ang_vel[t, b] = rotvec / dt
        ang_vel[0, b] = ang_vel[1, b]  # copy first frame

    return ang_vel


def convert(case: str, output_dir: str) -> str:
    """Convert one CORE4D case to HDMI motion format."""
    scene_path = os.path.join(BASE, case, "scene.xml")
    traj_path = os.path.join(BASE, case, "0", "trajectory_kinematic.npz")

    model = mujoco.MjModel.from_xml_path(scene_path)
    traj = np.load(traj_path)
    qpos = traj["qpos"]  # (T, 43)
    contact = traj["contact"]  # (T, 2) per-hand

    T_src = qpos.shape[0]
    print(f"  Source: {T_src} frames at {CORE4D_FPS}fps, qpos shape {qpos.shape}")

    # Step 1: FK at source fps
    body_idx_map = build_body_index_map(model)
    all_pos, all_quat = run_fk(model, qpos)

    # Select HDMI bodies (28 robot + 1 object)
    body_pos = all_pos[:, body_idx_map]  # (T_src, 29, 3)
    body_quat = all_quat[:, body_idx_map]  # (T_src, 29, 4) wxyz

    # Step 2: Extract joint positions (29 robot joints from qpos)
    joint_pos = qpos[:, 7:36]  # skip freejoint pelvis (7), take 29 joints
    joint_vel_src = finite_diff_vel(joint_pos, 1.0 / CORE4D_FPS)

    # Step 3: Upsample to HDMI fps
    body_pos_50 = upsample_pos(body_pos, CORE4D_FPS, HDMI_FPS)
    body_quat_50 = upsample_quat(body_quat, CORE4D_FPS, HDMI_FPS)
    joint_pos_50 = upsample_pos(joint_pos, CORE4D_FPS, HDMI_FPS)

    T_dst = body_pos_50.shape[0]
    dt = 1.0 / HDMI_FPS

    # Step 4: Compute velocities at 50fps
    body_lin_vel = finite_diff_vel(body_pos_50, dt)
    body_ang_vel = quat_angular_vel(body_quat_50, dt)
    joint_vel_50 = finite_diff_vel(joint_pos_50, dt)

    # Step 5: Upsample contact
    contact_any = (contact[:, 0].astype(bool) | contact[:, 1].astype(bool)).astype(np.float32)
    contact_50 = upsample_pos(contact_any[:, None], CORE4D_FPS, HDMI_FPS)  # (T_dst, 1)
    object_contact = (contact_50 > 0.5).astype(np.float32)

    print(f"  Output: {T_dst} frames at {HDMI_FPS}fps")
    print(f"  body_pos: {body_pos_50.shape}, joint_pos: {joint_pos_50.shape}, contact: {object_contact.shape}")

    # Step 6: Save
    os.makedirs(output_dir, exist_ok=True)
    np.savez(
        os.path.join(output_dir, "motion.npz"),
        body_pos_w=body_pos_50.astype(np.float32),
        body_quat_w=body_quat_50.astype(np.float32),
        body_lin_vel_w=body_lin_vel.astype(np.float32),
        body_ang_vel_w=body_ang_vel.astype(np.float32),
        joint_pos=joint_pos_50.astype(np.float32),
        joint_vel=joint_vel_50.astype(np.float32),
        object_contact=object_contact.astype(np.float32),
    )

    body_names = HDMI_BODY_NAMES + [OBJECT_BODY_NAME]
    meta = {
        "body_names": body_names,
        "joint_names": HDMI_JOINT_NAMES,
        "fps": HDMI_FPS,
    }
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=4)

    return output_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="box023_person1", help="CORE4D case name")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: auto)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"/home/ubuntu/Workspace/HDMI/data/motion/g1/core4d/{args.case}"

    print(f"Converting {args.case} → HDMI format")
    out = convert(args.case, args.output_dir)
    print(f"  Saved to {out}")


if __name__ == "__main__":
    main()
