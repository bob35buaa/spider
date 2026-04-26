"""Compare kinematic (input) vs physics (SPIDER output) trajectories.

Usage:
    python compare_kin_vs_phys.py --scene <scene.xml> --kin <trajectory_kinematic.npz> --phys <trajectory_mjwp.npz>

Example:
    python workspace/v3_spider/scripts/compare_kin_vs_phys.py \
        --scene /mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/spider/example_datasets/processed/omomo/unitree_g1/humanoid_object/move_largebox/scene.xml \
        --kin /mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/spider/example_datasets/processed/omomo/unitree_g1/humanoid_object/move_largebox/0/trajectory_kinematic.npz \
        --phys /mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/spider/example_datasets/processed/omomo/unitree_g1/humanoid_object/move_largebox/0/trajectory_mjwp.npz
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np


def get_body_pos(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> np.ndarray:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return data.xpos[body_id].copy()


def quat_to_mat(quat: np.ndarray) -> np.ndarray:
    """Convert wxyz quaternion to 3x3 rotation matrix."""
    w, x, y, z = quat
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def world_to_box_local(point: np.ndarray, box_pos: np.ndarray, box_quat: np.ndarray) -> np.ndarray:
    """Transform world point to box-local coordinates."""
    R = quat_to_mat(box_quat)
    return R.T @ (point - box_pos)


def analyze_trajectory(
    model: mujoco.MjModel,
    qpos_seq: np.ndarray,
    label: str,
) -> dict:
    """Analyze a trajectory and return per-frame metrics."""
    data = mujoco.MjData(model)
    T = qpos_seq.shape[0]
    nq = model.nq

    # Find body IDs
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    # Try to find hand bodies
    hand_names_l = ["left_rubber_hand", "left_wrist_roll_rubber_hand", "left_wrist_roll_link"]
    hand_names_r = ["right_rubber_hand", "right_wrist_roll_rubber_hand", "right_wrist_roll_link"]
    left_hand_id = -1
    right_hand_id = -1
    for name in hand_names_l:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            left_hand_id = bid
            break
    for name in hand_names_r:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            right_hand_id = bid
            break

    # Find object body (last freejoint body, or named)
    obj_names = ["largebox", "Box025", "object"]
    obj_id = -1
    for name in obj_names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            obj_id = bid
            break

    metrics = {
        "pelvis_z": [],
        "obj_pos": [],
        "obj_quat": [],
        "left_hand_pos": [],
        "right_hand_pos": [],
        "left_hand_obj_dist": [],
        "right_hand_obj_dist": [],
        "left_hand_box_local": [],
        "right_hand_box_local": [],
    }

    for t in range(T):
        data.qpos[:nq] = qpos_seq[t, :nq]
        mujoco.mj_forward(model, data)

        metrics["pelvis_z"].append(data.xpos[pelvis_id][2])

        # Object pose from qpos (last 7 dims)
        obj_pos = qpos_seq[t, -7:-4]
        obj_quat = qpos_seq[t, -4:]  # wxyz
        metrics["obj_pos"].append(obj_pos.copy())
        metrics["obj_quat"].append(obj_quat.copy())

        if left_hand_id >= 0:
            lh_pos = data.xpos[left_hand_id].copy()
            metrics["left_hand_pos"].append(lh_pos)
            metrics["left_hand_obj_dist"].append(np.linalg.norm(lh_pos - obj_pos))
            metrics["left_hand_box_local"].append(world_to_box_local(lh_pos, obj_pos, obj_quat))

        if right_hand_id >= 0:
            rh_pos = data.xpos[right_hand_id].copy()
            metrics["right_hand_pos"].append(rh_pos)
            metrics["right_hand_obj_dist"].append(np.linalg.norm(rh_pos - obj_pos))
            metrics["right_hand_box_local"].append(world_to_box_local(rh_pos, obj_pos, obj_quat))

    # Convert to arrays
    for k, v in metrics.items():
        if v:
            metrics[k] = np.array(v)

    return metrics


def flatten_mjwp_qpos(phys_data: dict) -> np.ndarray:
    """Flatten trajectory_mjwp.npz 3D qpos to 2D [T, nq].

    SPIDER saves qpos as [N_steps, ctrl_steps, nq].
    We take the last ctrl_step of each N_step as the "result" state.
    """
    qpos = phys_data["qpos"]
    if qpos.ndim == 3:
        # Take last control step per planning step
        return qpos[:, -1, :]
    return qpos


def print_comparison(kin_metrics: dict, phys_metrics: dict) -> None:
    """Print side-by-side comparison."""
    T_kin = len(kin_metrics["pelvis_z"])
    T_phys = len(phys_metrics["pelvis_z"])

    print(f"\n{'='*70}")
    print(f"  Trajectory Comparison: Kinematic (T={T_kin}) vs Physics (T={T_phys})")
    print(f"{'='*70}")

    # Object tracking error
    T_min = min(T_kin, T_phys)
    # Resample if lengths differ
    if T_kin != T_phys:
        kin_idx = np.linspace(0, T_kin - 1, T_min).astype(int)
        phys_idx = np.arange(T_phys) if T_phys == T_min else np.linspace(0, T_phys - 1, T_min).astype(int)
    else:
        kin_idx = phys_idx = np.arange(T_min)

    kin_obj = kin_metrics["obj_pos"][kin_idx]
    phys_obj = phys_metrics["obj_pos"][phys_idx]
    obj_pos_err = np.linalg.norm(kin_obj - phys_obj, axis=-1)

    print(f"\n--- Object Tracking (Physics vs Kinematic ref) ---")
    print(f"  Position error: mean={obj_pos_err.mean():.4f}, max={obj_pos_err.max():.4f}")

    # Pelvis comparison
    print(f"\n--- Robot Pelvis Z ---")
    print(f"  Kinematic: mean={np.mean(kin_metrics['pelvis_z']):.3f}, "
          f"min={np.min(kin_metrics['pelvis_z']):.3f}, max={np.max(kin_metrics['pelvis_z']):.3f}")
    print(f"  Physics:   mean={np.mean(phys_metrics['pelvis_z']):.3f}, "
          f"min={np.min(phys_metrics['pelvis_z']):.3f}, max={np.max(phys_metrics['pelvis_z']):.3f}")

    # Object Z
    kin_obj_z = kin_metrics["obj_pos"][:, 2]
    phys_obj_z = phys_metrics["obj_pos"][:, 2]
    print(f"\n--- Object Z ---")
    print(f"  Kinematic: mean={np.mean(kin_obj_z):.3f}, min={np.min(kin_obj_z):.3f}, max={np.max(kin_obj_z):.3f}")
    print(f"  Physics:   mean={np.mean(phys_obj_z):.3f}, min={np.min(phys_obj_z):.3f}, max={np.max(phys_obj_z):.3f}")

    # Hand-object distance
    for side in ["left", "right"]:
        key_dist = f"{side}_hand_obj_dist"
        key_local = f"{side}_hand_box_local"
        if isinstance(kin_metrics[key_dist], np.ndarray) and len(kin_metrics[key_dist]) > 0:
            print(f"\n--- {side.capitalize()} Hand-Object Distance ---")
            print(f"  Kinematic: mean={np.mean(kin_metrics[key_dist]):.3f}m")
            print(f"  Physics:   mean={np.mean(phys_metrics[key_dist]):.3f}m")

            print(f"\n--- {side.capitalize()} Hand Box-Local Coords (mean) ---")
            kin_local = np.mean(kin_metrics[key_local], axis=0)
            phys_local = np.mean(phys_metrics[key_local], axis=0)
            print(f"  Kinematic: x={kin_local[0]:.3f}, y={kin_local[1]:.3f}, z={kin_local[2]:.3f}")
            print(f"  Physics:   x={phys_local[0]:.3f}, y={phys_local[1]:.3f}, z={phys_local[2]:.3f}")

    # Side grip analysis
    print(f"\n--- Side Grip Analysis ---")
    for side in ["left", "right"]:
        key_local = f"{side}_hand_box_local"
        if isinstance(kin_metrics[key_local], np.ndarray) and len(kin_metrics[key_local]) > 0:
            for label, m in [("Kinematic", kin_metrics), ("Physics", phys_metrics)]:
                local_coords = m[key_local]
                # Side grip: |z| < |x| and |z| < |y| (hand on side, not top/bottom)
                abs_coords = np.abs(local_coords)
                side_mask = (abs_coords[:, 2] < abs_coords[:, 0]) | (abs_coords[:, 2] < abs_coords[:, 1])
                pct = 100.0 * np.mean(side_mask)
                print(f"  {label} {side}: {pct:.1f}% side grip, box-local z mean={np.mean(local_coords[:, 2]):.3f}")

    # Per-step summary (sample a few frames)
    print(f"\n--- Sample Frames (Physics) ---")
    sample_steps = np.linspace(0, T_phys - 1, min(5, T_phys)).astype(int)
    print(f"  {'Step':>5} {'PelvisZ':>8} {'ObjZ':>6} {'L-dist':>7} {'R-dist':>7} {'L-local-z':>10} {'R-local-z':>10}")
    for s in sample_steps:
        row = f"  {s:5d} {phys_metrics['pelvis_z'][s]:8.3f} {phys_metrics['obj_pos'][s][2]:6.3f}"
        if isinstance(phys_metrics["left_hand_obj_dist"], np.ndarray) and len(phys_metrics["left_hand_obj_dist"]) > s:
            row += f" {phys_metrics['left_hand_obj_dist'][s]:7.3f}"
            row += f" {phys_metrics['right_hand_obj_dist'][s]:7.3f}"
            row += f" {phys_metrics['left_hand_box_local'][s][2]:10.3f}"
            row += f" {phys_metrics['right_hand_box_local'][s][2]:10.3f}"
        print(row)

    print(f"\n{'='*70}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare kinematic vs physics trajectories")
    parser.add_argument("--scene", type=str, required=True, help="Path to scene.xml")
    parser.add_argument("--kin", type=str, required=True, help="Path to trajectory_kinematic.npz")
    parser.add_argument("--phys", type=str, required=True, help="Path to trajectory_mjwp.npz")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.scene)
    print(f"Scene: nq={model.nq}, nv={model.nv}, nu={model.nu}, nbody={model.nbody}")

    # Print body names for debugging
    print("Bodies:", [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)])

    # Load kinematic
    kin_data = np.load(args.kin)
    kin_qpos = kin_data["qpos"]
    print(f"\nKinematic: qpos shape={kin_qpos.shape}")

    # Load physics
    phys_data = dict(np.load(args.phys))
    phys_qpos = flatten_mjwp_qpos(phys_data)
    print(f"Physics:   qpos shape={phys_qpos.shape}")

    # Analyze
    kin_metrics = analyze_trajectory(model, kin_qpos, "Kinematic")
    phys_metrics = analyze_trajectory(model, phys_qpos, "Physics")

    print_comparison(kin_metrics, phys_metrics)


if __name__ == "__main__":
    main()
