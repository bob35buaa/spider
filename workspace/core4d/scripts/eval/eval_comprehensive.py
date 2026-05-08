#!/usr/bin/env python3
"""Unified comprehensive evaluation per workspace/core4d/docs/eval_metrics.md.

Reports ALL metrics at multiple thresholds. Uses separate FK models for sim (scene_act.xml)
and ref (scene.xml) to handle nq mismatch.

Usage:
  uv run workspace/core4d/scripts/eval/eval_comprehensive.py <task> <sim_npz> [--ref <ref_npz>]

If sim_npz has shape (T, 2, nq), channel 0=sim, channel 1=ref (internal).
If --ref is given, uses external kinematic ref with freejoint model for FK.
"""
import sys
import argparse
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R


def quat_angle(q1_wxyz, q2_wxyz):
    """Angle between two wxyz quaternions in degrees."""
    r1 = R.from_quat([q1_wxyz[1], q1_wxyz[2], q1_wxyz[3], q1_wxyz[0]])
    r2 = R.from_quat([q2_wxyz[1], q2_wxyz[2], q2_wxyz[3], q2_wxyz[0]])
    return np.degrees((r1.inv() * r2).magnitude())


def surface_distance(hand_pos, obj_pos, obj_mat, half_ext):
    """Hand-to-object surface distance (axis-aligned bbox in object frame)."""
    local = obj_mat.T @ (hand_pos - obj_pos)
    clamped = np.clip(local, -half_ext, half_ext)
    return np.linalg.norm(local - clamped)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", help="e.g. desk005_person2")
    parser.add_argument("sim_npz", help="sim trajectory npz")
    parser.add_argument("--ref", default=None, help="external ref npz (freejoint format)")
    args = parser.parse_args()

    task = args.task
    base_dir = f"example_datasets/processed/core4d/unitree_g1/humanoid_object/{task}"

    half_ext_map = {
        "desk005_person2": np.array([0.20, 0.37, 0.40]),
        "box025_person1": np.array([0.305, 0.305, 0.446]),
        "bucket010_person1": np.array([0.20, 0.20, 0.20]),
        "chair022_person1": np.array([0.25, 0.25, 0.40]),
    }
    obj_half = half_ext_map.get(task, np.array([0.20, 0.20, 0.20]))

    # Load sim
    d_sim = np.load(args.sim_npz)
    qpos_all = d_sim["qpos"]
    if qpos_all.ndim == 3 and qpos_all.shape[1] == 2:
        qpos_sim = qpos_all[:, 0, :]
    else:
        qpos_sim = qpos_all.reshape(-1, qpos_all.shape[-1])

    # Load ref
    if args.ref:
        ref_data = np.load(args.ref)
        qpos_ref = ref_data["qpos"]
        if qpos_ref.ndim == 3:
            qpos_ref = qpos_ref.reshape(-1, qpos_ref.shape[-1])
        use_freejoint_ref = (qpos_ref.shape[1] != qpos_sim.shape[1])
    else:
        # Use internal ref channel
        if qpos_all.ndim == 3 and qpos_all.shape[1] == 2:
            qpos_ref = qpos_all[:, 1, :]
            use_freejoint_ref = False
        else:
            print("ERROR: no ref available. Provide --ref or use (T,2,nq) npz.")
            return

    # Load models
    m_sim = mujoco.MjModel.from_xml_path(f"{base_dir}/scene_act.xml")
    d_s = mujoco.MjData(m_sim)
    if use_freejoint_ref:
        m_ref = mujoco.MjModel.from_xml_path(f"{base_dir}/scene.xml")
        d_r = mujoco.MjData(m_ref)
    else:
        m_ref = m_sim
        d_r = mujoco.MjData(m_ref)

    # Body IDs
    pelvis_id = 1
    obj_id = mujoco.mj_name2id(m_sim, mujoco.mjtObj.mjOBJ_BODY, "object")
    lw_id = mujoco.mj_name2id(m_sim, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
    rw_id = mujoco.mj_name2id(m_sim, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
    la_id = mujoco.mj_name2id(m_sim, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    ra_id = mujoco.mj_name2id(m_sim, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
    robot_body_ids = list(range(1, obj_id))
    eef_ids = [lw_id, rw_id]
    foot_ids = [la_id, ra_id]

    T = min(qpos_sim.shape[0], qpos_ref.shape[0])
    nq_robot = m_sim.nq - 6  # 36 for G1 (7 base + 29 joints)
    dt = 0.0333  # per saved frame

    # Per-frame metrics
    mpkpe_l, root_pos_l, root_ori_l = [], [], []
    eef_pos_l, eef_ori_l = [], []
    joint_err_l = []
    obj_pos_l, obj_ori_l = [], []
    pelvis_z_l = []
    surf_dist_l = []
    ref_surf_dist_l = []  # ref hand-obj distance (for desired contact detection)
    foot_z_l = []
    foot_pos_prev = None

    foot_skating_l = []

    for t in range(T):
        # FK sim
        d_s.qpos[:] = qpos_sim[t]
        mujoco.mj_kinematics(m_sim, d_s)

        # FK ref
        if use_freejoint_ref:
            d_r.qpos[:] = qpos_ref[t]
        else:
            d_r.qpos[:] = qpos_ref[t]
        mujoco.mj_kinematics(m_ref, d_r)

        # A. Body Tracking
        kp_err = [np.linalg.norm(d_s.xpos[b] - d_r.xpos[b]) for b in robot_body_ids]
        mpkpe_l.append(np.mean(kp_err))

        root_pos_l.append(np.linalg.norm(d_s.xpos[pelvis_id] - d_r.xpos[pelvis_id]))
        root_ori_l.append(quat_angle(d_s.xquat[pelvis_id], d_r.xquat[pelvis_id]))

        ep, eo = [], []
        for eid in eef_ids:
            ep.append(np.linalg.norm(d_s.xpos[eid] - d_r.xpos[eid]))
            eo.append(quat_angle(d_s.xquat[eid], d_r.xquat[eid]))
        eef_pos_l.append(np.mean(ep))
        eef_ori_l.append(np.mean(eo))

        # Joint error (robot joints only: qpos[7:36])
        j_sim = qpos_sim[t, 7:36]
        j_ref = qpos_ref[t, 7:36] if qpos_ref.shape[1] >= 36 else qpos_ref[t, 7:min(36, qpos_ref.shape[1])]
        joint_err_l.append(np.mean(np.abs(j_sim[:len(j_ref)] - j_ref)))

        # C. Object Tracking
        obj_pos_l.append(np.linalg.norm(d_s.xpos[obj_id] - d_r.xpos[obj_id]))
        obj_ori_l.append(quat_angle(d_s.xquat[obj_id], d_r.xquat[obj_id]))

        # D. Stability
        pelvis_z_l.append(d_s.xpos[pelvis_id, 2])
        foot_z = min(d_s.xpos[la_id, 2], d_s.xpos[ra_id, 2])
        foot_z_l.append(foot_z)

        # Foot skating
        foot_pos = np.array([d_s.xpos[la_id, :2], d_s.xpos[ra_id, :2]])
        if foot_pos_prev is not None:
            for fi in range(2):
                fz = d_s.xpos[foot_ids[fi], 2]
                if fz < 0.05:  # on ground
                    vel_xy = np.linalg.norm(foot_pos[fi] - foot_pos_prev[fi]) / dt
                    foot_skating_l.append(vel_xy > 0.10)
                else:
                    foot_skating_l.append(False)  # not on ground, not skating
        foot_pos_prev = foot_pos.copy()

        # E. Contact
        obj_pos_s = d_s.xpos[obj_id]
        obj_mat_s = d_s.xmat[obj_id].reshape(3, 3)
        sd = min(surface_distance(d_s.xpos[eid], obj_pos_s, obj_mat_s, obj_half) for eid in eef_ids)
        surf_dist_l.append(sd)

        # Ref contact (for desired contact detection)
        obj_pos_r = d_r.xpos[obj_id]
        obj_mat_r = d_r.xmat[obj_id].reshape(3, 3)
        sd_ref = min(surface_distance(d_r.xpos[eid], obj_pos_r, obj_mat_r, obj_half) for eid in eef_ids)
        ref_surf_dist_l.append(sd_ref)

    # Convert
    mpkpe = np.array(mpkpe_l) * 100
    root_pos = np.array(root_pos_l) * 100
    root_ori = np.array(root_ori_l)
    eef_pos = np.array(eef_pos_l) * 100
    eef_ori = np.array(eef_ori_l)
    joint_err = np.degrees(np.array(joint_err_l))
    obj_pos = np.array(obj_pos_l) * 100
    obj_ori = np.array(obj_ori_l)
    pelvis_z = np.array(pelvis_z_l)
    surf_dist = np.array(surf_dist_l)
    ref_surf = np.array(ref_surf_dist_l)
    foot_z = np.array(foot_z_l)

    # Smoothness
    joints_seq = qpos_sim[:T, 7:36]
    if T > 2:
        jacc = np.diff(joints_seq, n=2, axis=0) / (dt ** 2)
        smoothness = np.mean(np.abs(jacc))
    else:
        smoothness = 0.0

    # Sustained contact (longest consecutive <threshold)
    def longest_run(mask):
        mx, cur = 0, 0
        for v in mask:
            cur = cur + 1 if v else 0
            mx = max(mx, cur)
        return mx

    # Desired contact frames (ref hand-obj < 15cm)
    desired_contact = ref_surf < 0.15

    # Print
    print(f"\n{'='*72}")
    print(f"  COMPREHENSIVE EVAL: {task}")
    print(f"  File: {args.sim_npz.split('/')[-1]}")
    print(f"  T={T} frames ({T*dt:.1f}s)")
    print(f"{'='*72}")

    print(f"\n  A. BODY TRACKING")
    print(f"  {'─'*40}")
    print(f"  MPKPE (all bodies):     {mpkpe.mean():7.2f} ± {mpkpe.std():.2f} cm")
    print(f"  Joint Angle Error:      {joint_err.mean():7.2f} ± {joint_err.std():.2f} deg")
    print(f"  EEF Position Error:     {eef_pos.mean():7.2f} ± {eef_pos.std():.2f} cm")
    print(f"  EEF Orientation Error:  {eef_ori.mean():7.2f} ± {eef_ori.std():.2f} deg")

    print(f"\n  B. ROOT TRACKING")
    print(f"  {'─'*40}")
    print(f"  Root Position Error:    {root_pos.mean():7.2f} ± {root_pos.std():.2f} cm")
    print(f"  Root Orientation Error: {root_ori.mean():7.2f} ± {root_ori.std():.2f} deg")

    print(f"\n  C. OBJECT TRACKING")
    print(f"  {'─'*40}")
    print(f"  Obj Position Error:     {obj_pos.mean():7.2f} ± {obj_pos.std():.2f} cm")
    print(f"  Obj Orientation Error:  {obj_ori.mean():7.2f} ± {obj_ori.std():.2f} deg")

    print(f"\n  D. PHYSICAL PLAUSIBILITY")
    print(f"  {'─'*40}")
    print(f"  Pelvis z: min={pelvis_z.min():.3f}m, mean={pelvis_z.mean():.3f}m")
    print(f"  Stability:")
    for th in [0.70, 0.65, 0.60, 0.55]:
        print(f"    >{th:.2f}m: {100*(pelvis_z>th).mean():.1f}%")
    # Longest unstable
    unstable = pelvis_z < 0.60
    max_unstable = longest_run(unstable)
    print(f"  Longest unstable (<0.60m): {max_unstable} frames ({max_unstable*dt:.2f}s)")
    # Penetration
    pen_ratio = (foot_z < -0.01).mean() * 100
    print(f"  Penetration (foot<-1cm): {pen_ratio:.1f}%")
    # Foot skating
    if foot_skating_l:
        skating_ratio = np.mean(foot_skating_l) * 100
        print(f"  Foot Skating (>10cm/s on ground): {skating_ratio:.1f}%")

    print(f"\n  E. INTERACTION QUALITY")
    print(f"  {'─'*40}")
    print(f"  Mean hand-obj surface dist: {surf_dist.mean()*100:.2f} cm")
    print(f"  Contact (sim hand-obj distance < threshold):")
    for th in [0.15, 0.10, 0.05, 0.03, 0.01]:
        pct = (surf_dist < th).mean() * 100
        sus = longest_run(surf_dist < th)
        print(f"    <{th*100:2.0f}cm: {pct:5.1f}% (sustained: {sus} frames / {sus*dt:.2f}s)")

    # Contact preservation (only during desired contact frames)
    n_desired = desired_contact.sum()
    if n_desired > 0:
        print(f"  Contact Preservation (ref hand-obj < 15cm → {n_desired}/{T} desired frames):")
        for th in [0.10, 0.05, 0.03, 0.01]:
            pres = (surf_dist[desired_contact] < th).mean() * 100 if n_desired > 0 else 0
            print(f"    sim <{th*100:2.0f}cm | desired: {pres:.1f}%")
    else:
        print(f"  Contact Preservation: N/A (no desired contact frames in ref)")

    print(f"\n  F. SMOOTHNESS")
    print(f"  {'─'*40}")
    print(f"  Mean |joint acceleration|: {smoothness:.1f} rad/s²")

    # Summary line
    print(f"\n  {'─'*40}")
    print(f"  SUMMARY: MPKPE={mpkpe.mean():.1f}cm | Joint={joint_err.mean():.1f}° | "
          f"ObjPos={obj_pos.mean():.1f}cm | Stable>{0.60}={100*(pelvis_z>0.60).mean():.0f}% | "
          f"Contact<10cm={100*(surf_dist<0.10).mean():.0f}%")
    print()


if __name__ == "__main__":
    main()
