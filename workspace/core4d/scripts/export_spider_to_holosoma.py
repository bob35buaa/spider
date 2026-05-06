"""Export SPIDER physics-retargeted robot motion + kinematic object trajectory
to Holosoma RL training format.

Takes:
- Robot body from SPIDER MJWP output (physically compliant, no foot slide/penetration)
- Object trajectory from kinematic reference (correct lift trajectory)

Produces Holosoma MotionLoader-compatible .npz with all required fields.

Usage:
    python workspace/core4d/scripts/export_spider_to_holosoma.py \
        --spider-npz .../trajectory_mjwp.npz \
        --kinematic-npz .../trajectory_kinematic.npz \
        --scene-xml .../scene.xml \
        --output-npz .../spider_retargeted_holosoma.npz
"""

import argparse

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def resample_to_fps(data, src_dt, tgt_fps):
    """Resample data from src_dt timestep to target fps using linear interpolation."""
    tgt_dt = 1.0 / tgt_fps
    src_times = np.arange(data.shape[0]) * src_dt
    tgt_times = np.arange(0, src_times[-1], tgt_dt)
    if data.ndim == 1:
        return np.interp(tgt_times, src_times, data)
    result = np.zeros((len(tgt_times), *data.shape[1:]))
    for idx in np.ndindex(data.shape[1:]):
        result[(slice(None),) + idx] = np.interp(
            tgt_times, src_times, data[(slice(None),) + idx]
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spider-npz", required=True, help="SPIDER trajectory_mjwp.npz")
    parser.add_argument("--kinematic-npz", required=True, help="Kinematic reference trajectory_kinematic.npz")
    parser.add_argument("--scene-xml", required=True, help="MuJoCo scene XML (freejoint version)")
    parser.add_argument("--output-npz", required=True, help="Output Holosoma-format .npz")
    parser.add_argument("--tgt-fps", type=int, default=50, help="Target FPS (Holosoma standard)")
    parser.add_argument("--object-name", default="Box025", help="Object name for body_names")
    args = parser.parse_args()

    # Load scene
    model = mujoco.MjModel.from_xml_path(args.scene_xml)
    data = mujoco.MjData(model)

    # Load SPIDER output: (N_ctrl, substeps, nq)
    spider = np.load(args.spider_npz)
    spider_qpos = spider["qpos"]  # (42, 6, 43)
    spider_qvel = spider["qvel"]  # (42, 6, 41)

    # Flatten to simulation timesteps
    n_ctrl, n_sub, nq = spider_qpos.shape
    spider_qpos_flat = spider_qpos.reshape(-1, nq)  # (252, 43)
    spider_qvel_flat = spider_qvel.reshape(-1, spider_qvel.shape[-1])

    # Load kinematic reference (original 30fps)
    kin = np.load(args.kinematic_npz)
    kin_qpos = kin["qpos"]  # (124, 43)
    kin_fps = 30  # CORE4D fps

    # sim_dt = 1/60 for SPIDER
    spider_dt = 1.0 / 60.0
    T_spider = spider_qpos_flat.shape[0]

    # Build hybrid qpos: robot from SPIDER, object from kinematic reference
    # First, resample kinematic reference to match SPIDER timesteps (60fps)
    kin_times = np.arange(kin_qpos.shape[0]) / kin_fps
    spider_times = np.arange(T_spider) * spider_dt

    # Interpolate kinematic object trajectory to 60fps
    kin_obj_pos = kin_qpos[:, 36:39]  # (124, 3)
    kin_obj_quat = kin_qpos[:, 39:43]  # (124, 4) wxyz
    obj_pos_60fps = np.zeros((T_spider, 3))
    for d in range(3):
        obj_pos_60fps[:, d] = np.interp(spider_times, kin_times, kin_obj_pos[:, d])

    # Interpolate quaternions via slerp
    rotations = Rotation.from_quat(
        kin_obj_quat[:, [1, 2, 3, 0]]  # wxyz → xyzw for scipy
    )
    slerp = Slerp(kin_times, rotations)
    obj_rot_60fps = slerp(np.clip(spider_times, kin_times[0], kin_times[-1]))
    obj_quat_60fps_xyzw = obj_rot_60fps.as_quat()  # xyzw
    obj_quat_60fps_wxyz = obj_quat_60fps_xyzw[:, [3, 0, 1, 2]]  # wxyz

    # Build hybrid qpos
    hybrid_qpos = spider_qpos_flat.copy()
    hybrid_qpos[:, 36:39] = obj_pos_60fps
    hybrid_qpos[:, 39:43] = obj_quat_60fps_wxyz

    # Forward kinematics to extract body positions/orientations
    nbody = model.nbody
    body_names = []
    for i in range(nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        body_names.append(name if name else f"body_{i}")

    # Rename object body to match Holosoma convention
    for i, name in enumerate(body_names):
        if name == "object":
            body_names[i] = f"{args.object_name}_link"

    body_pos_w = np.zeros((T_spider, nbody, 3))
    body_quat_w = np.zeros((T_spider, nbody, 4))  # wxyz

    for t in range(T_spider):
        data.qpos[:] = hybrid_qpos[t]
        data.qvel[:] = spider_qvel_flat[t] if t < spider_qvel_flat.shape[0] else 0
        mujoco.mj_forward(model, data)
        body_pos_w[t] = data.xpos[:nbody].copy()
        # MuJoCo xquat is wxyz
        body_quat_w[t] = data.xquat[:nbody].copy()

    # Compute velocities via finite differences
    body_lin_vel_w = np.zeros_like(body_pos_w)
    body_ang_vel_w = np.zeros_like(body_pos_w)
    for t in range(1, T_spider):
        body_lin_vel_w[t] = (body_pos_w[t] - body_pos_w[t - 1]) / spider_dt
        # Angular velocity from quaternion difference (simplified)
        for b in range(nbody):
            r0 = Rotation.from_quat(body_quat_w[t - 1, b, [1, 2, 3, 0]])
            r1 = Rotation.from_quat(body_quat_w[t, b, [1, 2, 3, 0]])
            dr = r0.inv() * r1
            body_ang_vel_w[t, b] = dr.as_rotvec() / spider_dt

    # Build joint arrays (with pelvis prefix as Holosoma expects)
    # joint_pos: (T, 36) = pelvis(7) + 29 joints
    joint_pos = hybrid_qpos[:, :36]  # pelvis freejoint + 29 joints
    # joint_vel: (T, 35) = pelvis_vel(6) + 29 joints
    joint_vel = spider_qvel_flat[:, :35]

    # Object arrays
    object_pos_w = obj_pos_60fps
    object_quat_w = obj_quat_60fps_wxyz
    object_lin_vel_w = np.zeros_like(object_pos_w)
    object_ang_vel_w = np.zeros_like(object_pos_w)
    for t in range(1, T_spider):
        object_lin_vel_w[t] = (object_pos_w[t] - object_pos_w[t - 1]) / spider_dt
    for t in range(1, T_spider):
        r0 = Rotation.from_quat(object_quat_w[t - 1, [1, 2, 3, 0]])
        r1 = Rotation.from_quat(object_quat_w[t, [1, 2, 3, 0]])
        dr = r0.inv() * r1
        object_ang_vel_w[t] = dr.as_rotvec() / spider_dt

    # Joint names (29 G1 joints)
    joint_names = []
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name and name != "floating_base_joint" and name != "object_joint":
            joint_names.append(name)

    # Resample everything to target fps (50Hz for Holosoma)
    tgt_dt = 1.0 / args.tgt_fps
    tgt_times = np.arange(0, (T_spider - 1) * spider_dt, tgt_dt)
    T_out = len(tgt_times)

    def resample(arr):
        src_times_local = np.arange(arr.shape[0]) * spider_dt
        shape_out = (T_out,) + arr.shape[1:]
        out = np.zeros(shape_out)
        for idx in np.ndindex(arr.shape[1:]):
            sl = (slice(None),) + idx
            out[sl] = np.interp(tgt_times, src_times_local, arr[sl])
        return out

    # Resample all arrays
    body_pos_w_out = resample(body_pos_w)
    body_quat_w_out = resample(body_quat_w)  # Note: linear interp on quats is approximate
    body_lin_vel_w_out = resample(body_lin_vel_w)
    body_ang_vel_w_out = resample(body_ang_vel_w)
    joint_pos_out = resample(joint_pos)
    joint_vel_out = resample(joint_vel)
    object_pos_w_out = resample(object_pos_w)
    object_quat_w_out = resample(object_quat_w)
    object_lin_vel_w_out = resample(object_lin_vel_w)
    object_ang_vel_w_out = resample(object_ang_vel_w)

    # Save
    np.savez(
        args.output_npz,
        fps=np.array([args.tgt_fps]),
        body_pos_w=body_pos_w_out,
        body_quat_w=body_quat_w_out,
        body_lin_vel_w=body_lin_vel_w_out,
        body_ang_vel_w=body_ang_vel_w_out,
        joint_pos=joint_pos_out,
        joint_vel=joint_vel_out,
        body_names=np.array(body_names),
        joint_names=np.array(joint_names),
        object_pos_w=object_pos_w_out,
        object_quat_w=object_quat_w_out,
        object_lin_vel_w=object_lin_vel_w_out,
        object_ang_vel_w=object_ang_vel_w_out,
    )

    print(f"Exported {args.output_npz}")
    print(f"  T={T_out}, fps={args.tgt_fps}")
    print(f"  body_pos_w: {body_pos_w_out.shape}")
    print(f"  joint_pos: {joint_pos_out.shape}")
    print(f"  body_names: {len(body_names)} ({body_names[0]}...{body_names[-1]})")
    print(f"  joint_names: {len(joint_names)}")
    print(f"  object_pos_w: {object_pos_w_out.shape}")


if __name__ == "__main__":
    main()
