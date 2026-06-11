#!/usr/bin/env python3
"""E033 multi-threshold contact evaluation for desk005."""
import sys
import numpy as np
import mujoco

def evaluate(npz_path, task="desk005_person2"):
    obj_half = np.array([0.20, 0.37, 0.40])  # desk005
    carry_start_t, carry_end_t = 0.3, 3.6  # carrying phase in seconds
    dt = 1.0 / 30.0  # ref dt

    d = np.load(npz_path)
    qpos = d["qpos"].reshape(-1, d["qpos"].shape[-1])
    T = qpos.shape[0]
    sim_dt = d.get("time", None)

    xml = f"example_datasets/processed/core4d/unitree_g1/humanoid_object/{task}/scene_act.xml"
    m = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(m)
    left_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
    right_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
    obj_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")

    surf_dists = []
    for t in range(T):
        data.qpos[:] = qpos[t]
        mujoco.mj_kinematics(m, data)
        obj_pos = data.xpos[obj_id]
        obj_mat = data.xmat[obj_id].reshape(3, 3)
        dists = []
        for hid in [left_id, right_id]:
            local = obj_mat.T @ (data.xpos[hid] - obj_pos)
            clamped = np.clip(local, -obj_half, obj_half)
            dists.append(np.linalg.norm(local - clamped))
        surf_dists.append(min(dists))

    surf_dists = np.array(surf_dists)
    pelvis_z = qpos[:, 2]
    obj_disp = np.linalg.norm(qpos[-1, -6:-3] - qpos[0, -6:-3])

    # Carrying phase frames (based on sim time)
    # Each sim step = sim_dt (0.0167s), carrying phase = 0.3-3.6s
    carry_start_f = int(carry_start_t / 0.0167)
    carry_end_f = min(int(carry_end_t / 0.0167), T - 1)
    carry_mask = np.zeros(T, dtype=bool)
    carry_mask[carry_start_f:carry_end_f] = True

    thresholds = [0.15, 0.10, 0.05, 0.03, 0.01]

    print(f"=== {npz_path} ===")
    print(f"  T={T} frames, obj_disp={obj_disp:.2f}m")
    print(f"  Stable (pelvis>0.55): {(pelvis_z > 0.55).mean() * 100:.1f}%")
    print(f"  Pelvis z: min={pelvis_z.min():.3f}, mean={pelvis_z.mean():.3f}")
    print(f"  Carrying phase: frames {carry_start_f}-{carry_end_f} ({carry_mask.sum()} frames)")
    print(f"  Contact (full sequence):")
    for th in thresholds:
        pct = (surf_dists < th).mean() * 100
        print(f"    <{th*100:.0f}cm: {pct:.1f}%")
    print(f"  Contact (carrying phase only):")
    carry_dists = surf_dists[carry_mask]
    for th in thresholds:
        pct = (carry_dists < th).mean() * 100
        print(f"    <{th*100:.0f}cm: {pct:.1f}%")
    print(f"  Stable during carry: {(pelvis_z[carry_mask] > 0.55).mean() * 100:.1f}%")
    print(f"  Mean surf dist (carry): {carry_dists.mean():.4f}m")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: eval_e033_desk005.py <npz_path> [npz_path2 ...]")
        sys.exit(1)
    for path in sys.argv[1:]:
        evaluate(path)
