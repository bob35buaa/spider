#!/usr/bin/env python3
"""E034 rigorous evaluation — multi-threshold stability + contact + temporal analysis."""
import sys
import numpy as np
import mujoco


def evaluate(npz_path, task=None):
    """Evaluate with multiple stability thresholds and temporal breakdown."""
    d = np.load(npz_path)
    qpos = d["qpos"].reshape(-1, d["qpos"].shape[-1])
    T = qpos.shape[0]

    # Auto-detect task from path
    if task is None:
        for t in ["desk005_person2", "box025_person1", "bucket010_person1", "chair022_person1"]:
            if t in npz_path:
                task = t
                break
    if task is None:
        task = "desk005_person2"

    # Object half-extents (approximate, for surface distance)
    half_ext_map = {
        "desk005_person2": np.array([0.20, 0.37, 0.40]),
        "box025_person1": np.array([0.305, 0.305, 0.446]),
        "bucket010_person1": np.array([0.20, 0.20, 0.20]),
        "chair022_person1": np.array([0.25, 0.25, 0.40]),
    }
    obj_half = half_ext_map.get(task, np.array([0.20, 0.20, 0.20]))

    xml = f"example_datasets/processed/core4d/unitree_g1/humanoid_object/{task}/scene_act.xml"
    m = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(m)
    pelvis_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    left_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
    right_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
    obj_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")

    pelvis_z_list = []
    surf_dists = []
    for t in range(T):
        data.qpos[:] = qpos[t]
        mujoco.mj_kinematics(m, data)
        # Use xpos (FK result), not qpos[2]
        pelvis_z_list.append(data.xpos[pelvis_id, 2])
        # Surface distance
        obj_pos = data.xpos[obj_id]
        obj_mat = data.xmat[obj_id].reshape(3, 3)
        dists = []
        for hid in [left_id, right_id]:
            local = obj_mat.T @ (data.xpos[hid] - obj_pos)
            clamped = np.clip(local, -obj_half, obj_half)
            dists.append(np.linalg.norm(local - clamped))
        surf_dists.append(min(dists))

    pelvis_z = np.array(pelvis_z_list)
    surf_dists = np.array(surf_dists)
    obj_disp = np.linalg.norm(qpos[-1, -6:-3] - qpos[0, -6:-3])

    # Time segments (each frame ≈ 0.0167s for sim_dt=0.0167)
    frame_dt = 0.0167
    seg_size = T // 5

    print(f"=== {task} — {npz_path.split('/')[-1]} ===")
    print(f"  T={T} frames ({T * frame_dt:.1f}s), obj_disp={obj_disp:.2f}m")
    print()

    # Multi-threshold stability (using xpos pelvis_z from FK)
    print("  Stability (pelvis xpos z from FK):")
    print(f"    min={pelvis_z.min():.3f}m, mean={pelvis_z.mean():.3f}m, max={pelvis_z.max():.3f}m")
    for th in [0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40]:
        pct = (pelvis_z > th).mean() * 100
        print(f"    >{th:.2f}m: {pct:.1f}%")

    # Temporal stability breakdown (5 segments)
    print()
    print("  Temporal breakdown (pelvis_z by segment):")
    for i in range(5):
        s = i * seg_size
        e = min((i + 1) * seg_size, T)
        seg_z = pelvis_z[s:e]
        t_start = s * frame_dt
        t_end = e * frame_dt
        print(f"    [{t_start:.1f}-{t_end:.1f}s] min={seg_z.min():.3f} mean={seg_z.mean():.3f} >0.60={100*(seg_z>0.60).mean():.0f}% >0.55={100*(seg_z>0.55).mean():.0f}%")

    # Find worst moments
    worst_idx = pelvis_z.argmin()
    print(f"    Worst frame: {worst_idx} (t={worst_idx*frame_dt:.2f}s, z={pelvis_z[worst_idx]:.3f}m)")
    # Find longest consecutive unstable stretch (pelvis < 0.60)
    unstable = pelvis_z < 0.60
    if unstable.any():
        max_run = 0
        current_run = 0
        run_start = 0
        best_start = 0
        for i in range(T):
            if unstable[i]:
                if current_run == 0:
                    run_start = i
                current_run += 1
                if current_run > max_run:
                    max_run = current_run
                    best_start = run_start
            else:
                current_run = 0
        print(f"    Longest unstable (<0.60m) stretch: {max_run} frames ({max_run*frame_dt:.2f}s) starting at frame {best_start} (t={best_start*frame_dt:.2f}s)")
    else:
        print("    No unstable (<0.60m) frames")

    # Contact metrics (focus on <10cm and tighter)
    print()
    print("  Contact (hand-to-object-surface distance):")
    print(f"    mean={surf_dists.mean():.4f}m, min={surf_dists.min():.4f}m")
    for th in [0.10, 0.05, 0.03, 0.01]:
        pct = (surf_dists < th).mean() * 100
        print(f"    <{th*100:.0f}cm: {pct:.1f}%")

    # Temporal contact breakdown
    print()
    print("  Temporal contact breakdown:")
    for i in range(5):
        s = i * seg_size
        e = min((i + 1) * seg_size, T)
        seg_d = surf_dists[s:e]
        t_start = s * frame_dt
        t_end = e * frame_dt
        pct10 = 100 * (seg_d < 0.10).mean()
        pct5 = 100 * (seg_d < 0.05).mean()
        print(f"    [{t_start:.1f}-{t_end:.1f}s] mean_surf={seg_d.mean():.3f}m <10cm={pct10:.0f}% <5cm={pct5:.0f}%")

    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: eval_e034_rigorous.py <npz_path> [npz_path2 ...]")
        sys.exit(1)
    for path in sys.argv[1:]:
        evaluate(path)
