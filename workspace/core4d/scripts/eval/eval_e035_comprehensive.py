#!/usr/bin/env python3
"""E035 comprehensive evaluation — paper-standard metrics (SPIDER Table 4 + DynaRetarget Table V).

Metrics:
- Body Tracking: MPKPE (mean per-keypoint pos error, cm), joint angle error (deg), body ori error (deg)
- Root Tracking: root pos error (cm), root ori error (deg)
- Object Tracking: obj pos error (cm), obj ori error (deg)
- Stability: pelvis_z multi-threshold, foot-ground contact, longest unstable stretch
- Contact Quality: hand-obj surface dist, sustained contact ratio (consecutive frames <10cm)
- Smoothness: joint acceleration (jerk proxy), qvel magnitude
"""
import sys
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R


def quat_angle_diff(q1_wxyz, q2_wxyz):
    """Compute angle between two quaternions in degrees. q: (4,) wxyz."""
    # Convert wxyz to xyzw for scipy
    r1 = R.from_quat([q1_wxyz[1], q1_wxyz[2], q1_wxyz[3], q1_wxyz[0]])
    r2 = R.from_quat([q2_wxyz[1], q2_wxyz[2], q2_wxyz[3], q2_wxyz[0]])
    diff = r1.inv() * r2
    return np.degrees(diff.magnitude())


def evaluate_comprehensive(npz_path, ref_npz_path, task=None):
    """Full paper-standard evaluation."""
    # Auto-detect task
    if task is None:
        for t in ["desk005_person2", "box025_person1", "bucket010_person1", "chair022_person1"]:
            if t in npz_path or t in ref_npz_path:
                task = t
                break
    if task is None:
        task = "desk005_person2"

    # Object half-extents
    half_ext_map = {
        "desk005_person2": np.array([0.20, 0.37, 0.40]),
        "box025_person1": np.array([0.305, 0.305, 0.446]),
        "bucket010_person1": np.array([0.20, 0.20, 0.20]),
        "chair022_person1": np.array([0.25, 0.25, 0.40]),
    }
    obj_half = half_ext_map.get(task, np.array([0.20, 0.20, 0.20]))

    # Load sim and ref data
    d_sim = np.load(npz_path)
    qpos_all = d_sim["qpos"]
    # Handle (T, 2, nq) format: channel 0 = sim, channel 1 = ref (in model nq format)
    if qpos_all.ndim == 3 and qpos_all.shape[1] == 2:
        qpos_sim = qpos_all[:, 0, :]
        qpos_ref_model = qpos_all[:, 1, :]  # ref already in model nq format
        use_internal_ref = True
    else:
        qpos_sim = qpos_all.reshape(-1, qpos_all.shape[-1])
        use_internal_ref = False

    # Load external ref if needed (for cross-checking or if no internal ref)
    if ref_npz_path and not use_internal_ref:
        d_ref = np.load(ref_npz_path)
        qpos_ref_raw = d_ref["qpos"]
        if qpos_ref_raw.ndim == 3:
            qpos_ref_raw = qpos_ref_raw.reshape(-1, qpos_ref_raw.shape[-1])
    elif use_internal_ref:
        qpos_ref_raw = None  # use qpos_ref_model directly
    else:
        print("ERROR: no ref data available")
        return

    T_sim = qpos_sim.shape[0]
    T = T_sim

    # Load model
    xml = f"example_datasets/processed/core4d/unitree_g1/humanoid_object/{task}/scene_act.xml"
    m = mujoco.MjModel.from_xml_path(xml)
    data_sim = mujoco.MjData(m)
    data_ref = mujoco.MjData(m)

    # Body IDs
    pelvis_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    left_wrist_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
    right_wrist_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
    left_ankle_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    right_ankle_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
    obj_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")

    # All robot body IDs (1 to obj_id-1)
    robot_body_ids = list(range(1, obj_id))
    eef_ids = [left_wrist_id, right_wrist_id]
    foot_ids = [left_ankle_id, right_ankle_id]

    # Collect per-frame metrics
    mpkpe_list = []       # per-keypoint position error (all robot bodies)
    root_pos_err_list = []
    root_ori_err_list = []
    joint_err_list = []   # joint angle differences (rad -> deg)
    eef_pos_err_list = [] # end-effector position error
    eef_ori_err_list = [] # end-effector orientation error
    obj_pos_err_list = []
    obj_ori_err_list = []
    pelvis_z_list = []
    foot_z_list = []      # min foot height (for ground contact check)
    surf_dist_list = []   # hand-object surface distance
    qvel_norm_list = []   # joint velocity magnitude (smoothness)

    nq_robot = m.nq - 6  # robot qpos (excluding object 6-DOF)

    for t in range(T):
        # FK for sim
        data_sim.qpos[:] = qpos_sim[t]
        mujoco.mj_kinematics(m, data_sim)

        # FK for ref — use internal ref (already in model nq format)
        if use_internal_ref:
            data_ref.qpos[:] = qpos_ref_model[t]
        else:
            # External ref may have different nq (freejoint=43 vs model=42)
            if qpos_ref_raw.shape[1] == m.nq:
                data_ref.qpos[:] = qpos_ref_raw[t]
            else:
                data_ref.qpos[:nq_robot] = qpos_ref_raw[t, :nq_robot]
                data_ref.qpos[nq_robot:] = qpos_sim[t, nq_robot:]
        mujoco.mj_kinematics(m, data_ref)

        # --- MPKPE: mean per-keypoint position error ---
        kp_errors = []
        for bid in robot_body_ids:
            err = np.linalg.norm(data_sim.xpos[bid] - data_ref.xpos[bid])
            kp_errors.append(err)
        mpkpe_list.append(np.mean(kp_errors))

        # --- Root tracking ---
        root_pos_err = np.linalg.norm(data_sim.xpos[pelvis_id] - data_ref.xpos[pelvis_id])
        root_pos_err_list.append(root_pos_err)
        root_ori_err = quat_angle_diff(data_sim.xquat[pelvis_id], data_ref.xquat[pelvis_id])
        root_ori_err_list.append(root_ori_err)

        # --- Joint angle error ---
        # Compare joint angles (skip first 7 = freejoint base)
        nq_joints_sim = nq_robot - 7  # robot joints excluding pelvis freejoint
        if nq_joints_sim > 0:
            j_sim = qpos_sim[t, 7:nq_robot]
            if use_internal_ref:
                j_ref = qpos_ref_model[t, 7:nq_robot]
            elif qpos_ref_raw is not None and qpos_ref_raw.shape[1] >= nq_robot:
                j_ref = qpos_ref_raw[t, 7:nq_robot]
            else:
                j_ref = j_sim
            joint_err_list.append(np.mean(np.abs(j_sim - j_ref)))

        # --- End-effector tracking ---
        eef_pos_errs = []
        eef_ori_errs = []
        for eid in eef_ids:
            eef_pos_errs.append(np.linalg.norm(data_sim.xpos[eid] - data_ref.xpos[eid]))
            eef_ori_errs.append(quat_angle_diff(data_sim.xquat[eid], data_ref.xquat[eid]))
        eef_pos_err_list.append(np.mean(eef_pos_errs))
        eef_ori_err_list.append(np.mean(eef_ori_errs))

        # --- Object tracking ---
        obj_pos_err = np.linalg.norm(data_sim.xpos[obj_id] - data_ref.xpos[obj_id])
        obj_pos_err_list.append(obj_pos_err)
        obj_ori_err = quat_angle_diff(data_sim.xquat[obj_id], data_ref.xquat[obj_id])
        obj_ori_err_list.append(obj_ori_err)

        # --- Stability ---
        pelvis_z_list.append(data_sim.xpos[pelvis_id, 2])
        foot_z = min(data_sim.xpos[left_ankle_id, 2], data_sim.xpos[right_ankle_id, 2])
        foot_z_list.append(foot_z)

        # --- Contact: hand-object surface distance ---
        obj_pos = data_sim.xpos[obj_id]
        obj_mat = data_sim.xmat[obj_id].reshape(3, 3)
        dists = []
        for hid in eef_ids:
            local = obj_mat.T @ (data_sim.xpos[hid] - obj_pos)
            clamped = np.clip(local, -obj_half, obj_half)
            dists.append(np.linalg.norm(local - clamped))
        surf_dist_list.append(min(dists))

    # Convert to arrays
    mpkpe = np.array(mpkpe_list) * 100  # m -> cm
    root_pos_err = np.array(root_pos_err_list) * 100  # m -> cm
    root_ori_err = np.array(root_ori_err_list)  # degrees
    joint_err = np.degrees(np.array(joint_err_list)) if joint_err_list else np.array([0.0])  # rad -> deg
    eef_pos_err = np.array(eef_pos_err_list) * 100  # m -> cm
    eef_ori_err = np.array(eef_ori_err_list)  # degrees
    obj_pos_err = np.array(obj_pos_err_list) * 100  # m -> cm
    obj_ori_err = np.array(obj_ori_err_list)  # degrees
    pelvis_z = np.array(pelvis_z_list)
    surf_dist = np.array(surf_dist_list)

    # Smoothness: joint acceleration (finite diff of qpos joints)
    if T > 2:
        joints_sim = qpos_sim[:T, 7:nq_robot]
        jvel = np.diff(joints_sim, axis=0) / 0.0167  # approx joint velocity
        jacc = np.diff(jvel, axis=0) / 0.0167  # approx joint acceleration
        smoothness = np.mean(np.abs(jacc))  # mean absolute joint acceleration (rad/s^2)
    else:
        smoothness = 0.0

    # Sustained contact: longest consecutive stretch with surf_dist < 0.10m
    contact_10 = surf_dist < 0.10
    max_sustained = 0
    current = 0
    for c in contact_10:
        if c:
            current += 1
            max_sustained = max(max_sustained, current)
        else:
            current = 0

    # Print results
    print(f"\n{'='*70}")
    print(f" {task} — {npz_path.split('/')[-1]}")
    print(f" T={T} frames ({T*0.0167:.1f}s)")
    print(f"{'='*70}")

    print(f"\n Body Tracking (SPIDER Table 4 style):")
    print(f"   MPKPE (all bodies):     {mpkpe.mean():.2f} ± {mpkpe.std():.2f} cm")
    print(f"   Joint Angle Error:      {joint_err.mean():.2f} ± {joint_err.std():.2f} deg")
    print(f"   EEF Position Error:     {eef_pos_err.mean():.2f} ± {eef_pos_err.std():.2f} cm")
    print(f"   EEF Orientation Error:  {eef_ori_err.mean():.2f} ± {eef_ori_err.std():.2f} deg")

    print(f"\n Root Tracking:")
    print(f"   Root Position Error:    {root_pos_err.mean():.2f} ± {root_pos_err.std():.2f} cm")
    print(f"   Root Orientation Error: {root_ori_err.mean():.2f} ± {root_ori_err.std():.2f} deg")

    print(f"\n Object Tracking:")
    print(f"   Object Position Error:  {obj_pos_err.mean():.2f} ± {obj_pos_err.std():.2f} cm")
    print(f"   Object Orientation Error: {obj_ori_err.mean():.2f} ± {obj_ori_err.std():.2f} deg")

    print(f"\n Stability:")
    print(f"   Pelvis z: min={pelvis_z.min():.3f}m, mean={pelvis_z.mean():.3f}m")
    print(f"   >0.70m: {100*(pelvis_z>0.70).mean():.1f}%  >0.65m: {100*(pelvis_z>0.65).mean():.1f}%  >0.60m: {100*(pelvis_z>0.60).mean():.1f}%")
    # Longest unstable stretch
    unstable = pelvis_z < 0.60
    max_unstable = 0
    cur = 0
    for u in unstable:
        if u:
            cur += 1
            max_unstable = max(max_unstable, cur)
        else:
            cur = 0
    print(f"   Longest unstable (<0.60m): {max_unstable} frames ({max_unstable*0.0167:.2f}s)")

    print(f"\n Contact Quality:")
    print(f"   Mean hand-obj surface dist: {surf_dist.mean()*100:.2f} cm")
    print(f"   <10cm: {100*(surf_dist<0.10).mean():.1f}%  <5cm: {100*(surf_dist<0.05).mean():.1f}%  <3cm: {100*(surf_dist<0.03).mean():.1f}%")
    print(f"   Longest sustained contact (<10cm): {max_sustained} frames ({max_sustained*0.0167:.2f}s)")
    total_contact_time = contact_10.sum() * 0.0167
    print(f"   Total contact time (<10cm): {total_contact_time:.2f}s / {T*0.0167:.1f}s")

    print(f"\n Smoothness:")
    print(f"   Mean |joint acceleration|: {smoothness:.1f} rad/s²")

    print()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: eval_e035_comprehensive.py <sim_npz> <ref_npz> [task]")
        print("  ref_npz: trajectory_kinematic.npz (freejoint format)")
        sys.exit(1)
    sim_path = sys.argv[1]
    ref_path = sys.argv[2]
    task = sys.argv[3] if len(sys.argv) > 3 else None
    evaluate_comprehensive(sim_path, ref_path, task)
