"""E051a+b: Verify HDMI data pipeline correctness + Euler convention error analysis.

Checks:
1. motion.npz FK consistency (body_pos_w vs trajectory_kinematic + scene.xml FK)
2. Initial frame alignment (motion.npz frame 0 vs scene XML body pos)
3. Euler convention error quantification (xyz vs XYZ vs best per-object)

Usage:
    uv run workspace/core4d/scripts/eval/verify_hdmi_data.py
"""

import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

# Paths
SPIDER_ROOT = Path(__file__).resolve().parents[4]
CORE4D_BASE = SPIDER_ROOT / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
HDMI_MOTION_BASE = Path("/home/ubuntu/Workspace/HDMI/data/motion/g1/core4d")
HDMI_SCENE_BASE = SPIDER_ROOT / "example_datasets/processed/hdmi/unitree_g1/humanoid_object"

CASES = ["box023_person1", "box025_person1"]
HDMI_TASK_MAP = {"box023_person1": "move_box023", "box025_person1": "move_box025"}

# All 6 intrinsic conventions
EULER_CONVENTIONS = ["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"]


def geodesic_angle(q1_wxyz: np.ndarray, q2_wxyz: np.ndarray) -> float:
    """Compute geodesic angle between two quaternions (wxyz) in degrees."""
    q1_xyzw = np.array([q1_wxyz[1], q1_wxyz[2], q1_wxyz[3], q1_wxyz[0]])
    q2_xyzw = np.array([q2_wxyz[1], q2_wxyz[2], q2_wxyz[3], q2_wxyz[0]])
    r1 = R.from_quat(q1_xyzw)
    r2 = R.from_quat(q2_xyzw)
    diff = r1.inv() * r2
    return np.degrees(diff.magnitude())


def verify_case(case: str):
    """Run all verifications for a single case."""
    print(f"\n{'='*70}")
    print(f"  CASE: {case}")
    print(f"{'='*70}")

    # Load data
    kin_path = CORE4D_BASE / case / "0" / "trajectory_kinematic.npz"
    scene_path = CORE4D_BASE / case / "scene.xml"
    motion_path = HDMI_MOTION_BASE / case / "motion.npz"
    meta_path = HDMI_MOTION_BASE / case / "meta.json"
    hdmi_task = HDMI_TASK_MAP[case]
    hdmi_scene_path = HDMI_SCENE_BASE / hdmi_task / "scene" / "mjlab scene.xml"

    if not motion_path.exists():
        print(f"  [SKIP] motion.npz not found: {motion_path}")
        return
    if not kin_path.exists():
        print(f"  [SKIP] trajectory_kinematic.npz not found: {kin_path}")
        return

    kin = np.load(kin_path)
    motion = np.load(motion_path)
    with open(meta_path) as f:
        meta = json.load(f)

    qpos_kin = kin["qpos"]  # (T_30, 43)
    T_30 = qpos_kin.shape[0]
    T_50 = motion["body_pos_w"].shape[0]

    print(f"\n  --- E051a: Data Consistency ---")
    print(f"  Kinematic frames: {T_30} @ 30fps = {T_30/30:.2f}s")
    print(f"  Motion frames: {T_50} @ 50fps = {T_50/50:.2f}s")
    expected_T50 = round((T_30 - 1) / 30.0 * 50.0) + 1
    print(f"  Expected T_50: {expected_T50} (actual: {T_50}, diff: {T_50 - expected_T50})")

    # Check 1: Joint positions frame 0
    joint_pos_motion = motion["joint_pos"][0]  # (29,)
    joint_pos_kin = qpos_kin[0, 7:36]  # (29,)
    joint_diff = np.abs(joint_pos_motion - joint_pos_kin)
    print(f"\n  Joint pos (frame 0): max_diff={np.degrees(joint_diff.max()):.3f}°, "
          f"mean_diff={np.degrees(joint_diff.mean()):.3f}°")
    if joint_diff.max() > 0.01:
        print(f"  [WARN] Joint position mismatch > 0.01 rad!")

    # Check 2: FK reconstruction
    if scene_path.exists():
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        data = mujoco.MjData(model)
        data.qpos[:] = qpos_kin[0]
        mujoco.mj_forward(model, data)

        # Object body FK
        obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
        fk_obj_pos = data.xpos[obj_body_id].copy()
        fk_obj_quat = data.xquat[obj_body_id].copy()  # wxyz

        # Motion data object pos (index 28 = suitcase)
        motion_obj_pos = motion["body_pos_w"][0, 28]
        motion_obj_quat = motion["body_quat_w"][0, 28]  # wxyz

        pos_diff = np.linalg.norm(fk_obj_pos - motion_obj_pos)
        quat_diff = geodesic_angle(fk_obj_quat, motion_obj_quat)
        print(f"\n  Object FK vs motion.npz (frame 0):")
        print(f"    Position diff: {pos_diff*100:.3f} cm")
        print(f"    Rotation diff: {quat_diff:.3f}°")
        if pos_diff > 0.001:
            print(f"    [WARN] Position mismatch > 1mm!")
            print(f"    FK pos:     {fk_obj_pos}")
            print(f"    Motion pos: {motion_obj_pos}")

        # Check 3: Scene XML body pos vs motion frame 0
        if hdmi_scene_path.exists():
            hdmi_model = mujoco.MjModel.from_xml_path(str(hdmi_scene_path))
            suitcase_id = mujoco.mj_name2id(hdmi_model, mujoco.mjtObj.mjOBJ_BODY, "suitcase")
            if suitcase_id >= 0:
                body_default_pos = hdmi_model.body_pos[suitcase_id].copy()
                offset = np.linalg.norm(motion_obj_pos - body_default_pos)
                print(f"\n  Scene XML body_pos vs motion frame 0:")
                print(f"    Scene XML body pos: {body_default_pos}")
                print(f"    Motion frame 0 pos: {motion_obj_pos}")
                print(f"    Offset: {offset*100:.3f} cm")
                if offset > 0.01:
                    print(f"    [WARN] Initial position offset > 1cm!")
            else:
                print(f"    [WARN] 'suitcase' body not found in HDMI scene!")

        # Check robot body FK
        pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        fk_pelvis_pos = data.xpos[pelvis_id].copy()
        motion_pelvis_pos = motion["body_pos_w"][0, 0]  # pelvis = index 0
        pelvis_diff = np.linalg.norm(fk_pelvis_pos - motion_pelvis_pos)
        print(f"\n  Pelvis FK vs motion.npz (frame 0):")
        print(f"    Position diff: {pelvis_diff*100:.3f} cm")

        # Full body FK check (sample mid-frame)
        mid_frame_30 = T_30 // 2
        mid_frame_50 = int(mid_frame_30 * 50 / 30)
        data.qpos[:] = qpos_kin[mid_frame_30]
        mujoco.mj_forward(model, data)
        fk_obj_pos_mid = data.xpos[obj_body_id].copy()
        motion_obj_pos_mid = motion["body_pos_w"][mid_frame_50, 28]
        mid_pos_diff = np.linalg.norm(fk_obj_pos_mid - motion_obj_pos_mid)
        print(f"\n  Object FK vs motion.npz (mid-frame {mid_frame_30}/{mid_frame_50}):")
        print(f"    Position diff: {mid_pos_diff*100:.3f} cm")

    # =============================================
    # E051b: Euler Convention Error Quantification
    # =============================================
    print(f"\n  --- E051b: Euler Convention Analysis ---")

    # Extract object quaternions from kinematic trajectory
    obj_quats_wxyz = qpos_kin[:, 39:43]  # (T, 4) wxyz
    obj_quats_xyzw = np.stack([
        obj_quats_wxyz[:, 1], obj_quats_wxyz[:, 2],
        obj_quats_wxyz[:, 3], obj_quats_wxyz[:, 0]
    ], axis=-1)

    rotations = R.from_quat(obj_quats_xyzw)

    # Analyze each convention
    print(f"\n  Convention | Max Middle Angle | Mean Middle Angle | Max Recon Error")
    print(f"  {'-'*75}")

    best_conv = None
    best_max_middle = 999.0
    results = {}

    for conv in EULER_CONVENTIONS:
        euler_angles = rotations.as_euler(conv)  # intrinsic (uppercase)
        middle_angles = np.abs(euler_angles[:, 1])  # middle axis
        max_middle = np.degrees(middle_angles.max())
        mean_middle = np.degrees(middle_angles.mean())

        # Reconstruction error: euler → quat → compare
        recon_rotations = R.from_euler(conv, euler_angles)
        recon_quats = recon_rotations.as_quat()  # xyzw
        # Geodesic distance between original and reconstructed
        diff_rots = rotations.inv() * recon_rotations
        recon_errors = np.degrees(diff_rots.magnitude())
        max_recon_error = recon_errors.max()

        gimbal_flag = " ⚠️ GIMBAL" if max_middle > 80 else ""
        print(f"  {conv:>6s}   | {max_middle:>15.1f}° | {mean_middle:>16.1f}° | "
              f"{max_recon_error:>14.4f}°{gimbal_flag}")

        results[conv] = {
            "max_middle": max_middle,
            "mean_middle": mean_middle,
            "max_recon_error": max_recon_error,
        }

        if max_middle < best_max_middle:
            best_max_middle = max_middle
            best_conv = conv

    print(f"\n  Best convention: {best_conv} (max middle angle = {best_max_middle:.1f}°)")

    # Compare current HDMI code ("xyz" extrinsic = "ZYX" intrinsic) vs correct
    euler_xyz_ext = rotations.as_euler("xyz")  # extrinsic xyz = intrinsic ZYX
    euler_XYZ_int = rotations.as_euler("XYZ")  # intrinsic XYZ (matches joint order)

    # The error from using extrinsic xyz when joints expect intrinsic XYZ:
    # MuJoCo applies: R_x(e[0]) @ R_y(e[1]) @ R_z(e[2]) intrinsically
    # But euler_xyz gives angles for: first rotate x, then y, then z extrinsically
    # = intrinsic ZYX: R_z(e[2]) @ R_y(e[1]) @ R_x(e[0])
    # So the actual rotation achieved by MuJoCo when given extrinsic xyz values:
    rot_actual_mujoco = R.from_euler("XYZ", euler_xyz_ext)  # MuJoCo interprets as intrinsic XYZ
    rot_intended = R.from_euler("xyz", euler_xyz_ext)  # What the code intended (extrinsic xyz)

    diff_convention = rot_intended.inv() * rot_actual_mujoco
    conv_errors = np.degrees(diff_convention.magnitude())

    print(f"\n  Euler Convention Mismatch (extrinsic 'xyz' given to intrinsic XYZ joints):")
    print(f"    Max error:  {conv_errors.max():.1f}°")
    print(f"    Mean error: {conv_errors.mean():.1f}°")
    print(f"    Min error:  {conv_errors.min():.1f}°")

    if conv_errors.max() < 5.0:
        print(f"    → Euler mismatch is SMALL (<5°) — NOT the cause of object tracking failure!")
    else:
        print(f"    → Euler mismatch is LARGE (>{conv_errors.max():.0f}°) — this IS causing object drift!")

    # Check scene_act_meta.json for reference
    meta_act_path = CORE4D_BASE / case / "scene_act_meta.json"
    if meta_act_path.exists():
        with open(meta_act_path) as f:
            act_meta = json.load(f)
        print(f"\n  scene_act_meta.json convention: {act_meta.get('euler_convention', 'N/A')}")

    return results


def main():
    print("=" * 70)
    print("  E051: HDMI Data Pipeline Verification")
    print("=" * 70)

    for case in CASES:
        verify_case(case)

    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print("""
  Key Questions Answered:
  1. Is motion.npz FK correct? (body_pos_w matches scene.xml + qpos FK)
  2. Is initial alignment correct? (motion frame 0 = scene XML body pos)
  3. Does euler convention matter for box023? (expect: NO, < 1° diff)
  4. Does euler convention matter for box025? (expect: YES, >> 10° diff)
  5. What's the best euler convention per case?
    """)


if __name__ == "__main__":
    main()
