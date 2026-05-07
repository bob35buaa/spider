"""Export hybrid trajectory: physical robot body + kinematic object from reference.

For each case:
1. Load body-only SPIDER retargeting result (trajectory_mjwp.npz)
2. Load kinematic reference (trajectory_kinematic.npz)
3. Replace object qpos in SPIDER output with reference object qpos
4. Save as hybrid trajectory

Usage:
    python workspace/core4d/scripts/export/export_hybrid.py
"""

import os
import numpy as np
import mujoco


CASES = {
    "box025_person1": "example_datasets/processed/core4d/unitree_g1/humanoid_object/box025_person1",
    "bucket010_person1": "example_datasets/processed/core4d/unitree_g1/humanoid_object/bucket010_person1",
    "chair022_person1": "example_datasets/processed/core4d/unitree_g1/humanoid_object/chair022_person1",
    "desk005_person2": "example_datasets/processed/core4d/unitree_g1/humanoid_object/desk005_person2",
}

OUTPUT_DIR = "workspace/core4d/results/E030_hybrid_export"


def export_hybrid(case_name: str, case_dir: str) -> dict:
    """Export hybrid trajectory for one case."""
    spider_npz_path = f"workspace/core4d/results/E030_bodyonly/{case_name}.npz"
    kin_npz_path = f"{case_dir}/0/trajectory_kinematic.npz"
    scene_xml_path = f"{case_dir}/scene.xml"

    if not os.path.exists(spider_npz_path):
        return {"error": f"SPIDER NPZ not found: {spider_npz_path}"}

    # Load SPIDER result
    spider_data = np.load(spider_npz_path)
    qpos_spider = spider_data["qpos"]  # (n_mpc, horizon, nq=43)

    # Load kinematic reference
    kin_data = np.load(kin_npz_path)
    qpos_ref = kin_data["qpos"]  # (T_ref, nq=43)

    # Extract commit trajectory from SPIDER (first step of each MPC window)
    # Shape: (n_mpc, nq)
    qpos_commit = qpos_spider[:, 0, :].copy()
    n_mpc = qpos_commit.shape[0]

    # Map MPC steps to ref frames
    # Each MPC window covers `horizon * sim_dt` seconds
    # sim_dt = 0.0167, horizon steps = 24 per MPC window
    # So each MPC step = 24 * 0.0167 = 0.4s
    # ref_dt = 0.0333 → ref frames per MPC step = 0.4 / 0.0333 ≈ 12
    mpc_dt = 24 * 0.0167  # seconds per MPC window
    ref_dt = 0.0333
    ref_frames_per_mpc = mpc_dt / ref_dt

    # For each MPC step, get the corresponding ref frame
    hybrid_qpos = qpos_commit.copy()
    T_ref = qpos_ref.shape[0]

    for i in range(n_mpc):
        ref_idx = min(int(i * ref_frames_per_mpc), T_ref - 1)
        # Replace object state (last 7 dof: pos_xyz + quat_wxyz)
        hybrid_qpos[i, -7:] = qpos_ref[ref_idx, -7:]

    # Compute metrics
    pelvis_z = hybrid_qpos[:, 2]
    pelvis_z_min = pelvis_z.min()
    pelvis_stable = bool((pelvis_z > 0.55).all())

    # Object tracking (should be perfect since we replaced with ref)
    obj_pos_hybrid = hybrid_qpos[:, -7:-4]
    obj_pos_ref_sampled = np.array([
        qpos_ref[min(int(i * ref_frames_per_mpc), T_ref - 1), -7:-4]
        for i in range(n_mpc)
    ])
    obj_pos_err = np.linalg.norm(obj_pos_hybrid - obj_pos_ref_sampled, axis=1).mean()

    # Save hybrid trajectory
    output_path = os.path.join(OUTPUT_DIR, f"{case_name}_hybrid.npz")
    np.savez(
        output_path,
        qpos=hybrid_qpos,  # (n_mpc, nq) — robot from physics, object from ref
        qpos_robot_only=qpos_commit[:, :-7],  # robot joints only
        qpos_object_ref=hybrid_qpos[:, -7:],  # object from ref
        pelvis_z=pelvis_z,
        time=np.arange(n_mpc) * mpc_dt,
    )

    return {
        "n_frames": n_mpc,
        "pelvis_z_min": pelvis_z_min,
        "pelvis_stable": pelvis_stable,
        "obj_pos_err": obj_pos_err,
        "output_path": output_path,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"{'Case':<22} {'Frames':>6} {'Pelvis_z_min':>12} {'Stable':>8} {'Obj_pos_err':>11}")
    print("-" * 65)

    for case_name, case_dir in CASES.items():
        result = export_hybrid(case_name, case_dir)
        if "error" in result:
            print(f"{case_name:<22} ERROR: {result['error']}")
            continue
        print(
            f"{case_name:<22} {result['n_frames']:>6} "
            f"{result['pelvis_z_min']:>12.3f} "
            f"{'YES' if result['pelvis_stable'] else 'NO':>8} "
            f"{result['obj_pos_err']:>11.4f}"
        )

    print(f"\nOutput directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
