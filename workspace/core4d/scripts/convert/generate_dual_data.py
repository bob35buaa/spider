#!/usr/bin/env python3
"""Generate dual-robot trajectory + scene XMLs for all CORE4D cases.

Combines person1 and person2 holosoma retarget data into SPIDER dual format:
- trajectory_kinematic_dual.npz: qpos(T,79), qvel(T,76), ctrl(T,58), contact(T,4), contact_pos(T,4,3)
- scene_dual_robot.xml: two G1 robots + shared object
- scene_dual_robot_connect2.xml: + soft 2-connect weld constraints

Usage:
    python workspace/core4d/scripts/convert/generate_dual_data.py
"""

import os
import shutil
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

HOLOSOMA_DIR = "/home/ubuntu/Workspace/holosoma/workspace/v2/results/retarget_replace_batch_trimmed"
SPIDER_DIR = "example_datasets/processed/core4d/unitree_g1"

CASES = {
    "bucket010_person1": {
        "p1_file": "20231003_2-059-person1-bucket010_with_obj_original.npz",
        "p2_file": "20231003_2-059-person2-bucket010_with_obj_original.npz",
        "task": "bucket010_person1",
    },
    "chair022_person1": {
        "p1_file": "20231020-084-person1-chair022_with_obj_original.npz",
        "p2_file": "20231020-084-person2-chair022_with_obj_original.npz",
        "task": "chair022_person1",
    },
    "desk005_person2": {
        "p2_file": "20231023-030-person2-desk005_with_obj_original.npz",
        "p1_file": None,  # desk005 has no person1 in holosoma
        "task": "desk005_person2",
    },
}


def compute_qvel(qpos: np.ndarray, dt: float) -> np.ndarray:
    """Compute joint velocities from qpos via finite differences.

    For freejoint (7 qpos → 6 qvel): differentiate pos, convert quat diff to angular vel.
    For hinge joints: simple finite difference.
    """
    T, nq = qpos.shape
    # For dual: nq=79 → nv=76 (two freejoints with 7→6 each, plus object freejoint 7→6)
    # Robot1: qpos[0:7] base (7→6), qpos[7:36] joints (29→29) = 35 qvel
    # Robot2: qpos[36:43] base (7→6), qpos[43:72] joints (29→29) = 35 qvel
    # Object: qpos[72:79] freejoint (7→6) = 6 qvel
    # Total: 35 + 35 + 6 = 76

    nv = nq - 3  # 3 quaternion DOFs become angular velocity (3 freejoints × 1 less)
    qvel = np.zeros((T, nv), dtype=np.float64)

    for t in range(T - 1):
        # Robot1 base pos
        qvel[t, 0:3] = (qpos[t + 1, 0:3] - qpos[t, 0:3]) / dt
        # Robot1 base angular vel (from quaternion diff)
        q1 = qpos[t, 3:7]
        q2 = qpos[t + 1, 3:7]
        r1 = R.from_quat([q1[1], q1[2], q1[3], q1[0]])  # wxyz → xyzw
        r2 = R.from_quat([q2[1], q2[2], q2[3], q2[0]])
        dr = r2 * r1.inv()
        qvel[t, 3:6] = dr.as_rotvec() / dt
        # Robot1 joints
        qvel[t, 6:35] = (qpos[t + 1, 7:36] - qpos[t, 7:36]) / dt

        # Robot2 base pos
        qvel[t, 35:38] = (qpos[t + 1, 36:39] - qpos[t, 36:39]) / dt
        # Robot2 base angular vel
        q1 = qpos[t, 39:43]
        q2 = qpos[t + 1, 39:43]
        r1 = R.from_quat([q1[1], q1[2], q1[3], q1[0]])
        r2 = R.from_quat([q2[1], q2[2], q2[3], q2[0]])
        dr = r2 * r1.inv()
        qvel[t, 38:41] = dr.as_rotvec() / dt
        # Robot2 joints
        qvel[t, 41:70] = (qpos[t + 1, 43:72] - qpos[t, 43:72]) / dt

        # Object pos
        qvel[t, 70:73] = (qpos[t + 1, 72:75] - qpos[t, 72:75]) / dt
        # Object angular vel
        q1 = qpos[t, 75:79]
        q2 = qpos[t + 1, 75:79]
        r1 = R.from_quat([q1[1], q1[2], q1[3], q1[0]])
        r2 = R.from_quat([q2[1], q2[2], q2[3], q2[0]])
        dr = r2 * r1.inv()
        qvel[t, 73:76] = dr.as_rotvec() / dt

    # Last frame: copy previous
    qvel[-1] = qvel[-2] if T > 1 else qvel[0]
    return qvel


def compute_contact_pos(qpos: np.ndarray, scene_xml: str) -> np.ndarray:
    """Compute 4-hand world positions using MuJoCo FK.

    Hands: R1_left_wrist, R1_right_wrist, R2_left_wrist, R2_right_wrist
    """
    model = mujoco.MjModel.from_xml_path(scene_xml)
    data = mujoco.MjData(model)

    hand_names = [
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "r2_left_wrist_yaw_link",
        "r2_right_wrist_yaw_link",
    ]
    hand_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in hand_names]
    missing = [n for n, i in zip(hand_names, hand_ids) if i == -1]
    if missing:
        print(f"  WARNING: bodies not found: {missing}")
        return np.zeros((qpos.shape[0], 4, 3))

    T = qpos.shape[0]
    contact_pos = np.zeros((T, 4, 3))
    for t in range(T):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        for h, bid in enumerate(hand_ids):
            contact_pos[t, h] = data.xpos[bid].copy()

    return contact_pos


def generate_scene_dual_robot(single_scene_xml: str, output_xml: str):
    """Generate dual-robot scene by duplicating the robot with r2_ prefix.

    Based on the existing box025 scene_dual_robot.xml pattern.
    """
    from lxml import etree

    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(single_scene_xml, parser)
    root = tree.getroot()

    worldbody = root.find("worldbody")

    # Find the robot body (first body with freejoint named "floating_base_joint")
    robot_body = None
    for body in worldbody.findall("body"):
        fj = body.find(".//freejoint[@name='floating_base_joint']")
        if fj is not None:
            robot_body = body
            break

    if robot_body is None:
        raise ValueError("Could not find robot body with floating_base_joint")

    # Deep copy robot body
    import copy
    robot2 = copy.deepcopy(robot_body)

    # Prefix all names with "r2_"
    def prefix_names(elem, prefix="r2_"):
        for attr in ["name", "joint", "body1", "body2", "geom1", "geom2", "site", "tendon"]:
            val = elem.get(attr)
            if val is not None:
                elem.set(attr, prefix + val)
        # Also handle class references that contain body names
        for child in elem:
            prefix_names(child, prefix)

    prefix_names(robot2)

    # Insert robot2 before the object body
    obj_body = worldbody.find(".//body[@name='object']")
    if obj_body is not None:
        obj_idx = list(worldbody).index(obj_body)
        worldbody.insert(obj_idx, robot2)
    else:
        worldbody.append(robot2)

    # Duplicate actuators
    actuator_section = root.find("actuator")
    if actuator_section is not None:
        actuators = list(actuator_section)
        # Only duplicate robot actuators (not object ones)
        for act in actuators:
            name = act.get("name", "")
            joint = act.get("joint", "")
            if "object" in name or "object" in joint:
                continue
            act2 = copy.deepcopy(act)
            act2.set("name", "r2_" + name)
            act2.set("joint", "r2_" + joint)
            actuator_section.append(act2)

    # Duplicate contact pairs for robot2
    contact_section = root.find("contact")
    if contact_section is not None:
        pairs = list(contact_section.findall("pair"))
        for pair in pairs:
            name = pair.get("name", "")
            g1 = pair.get("geom1", "")
            g2 = pair.get("geom2", "")
            # Skip object pairs (will add separately)
            if "object" in name:
                # Add r2 version of hand-object pairs
                if "hand" in name or "lh" in g1 or "rh" in g1:
                    pair2 = copy.deepcopy(pair)
                    pair2.set("name", "r2_" + name)
                    pair2.set("geom1", "r2_" + g1)
                    # geom2 stays as object_collision
                    contact_section.append(pair2)
                continue
            # Robot-floor pairs: duplicate with r2_ prefix
            pair2 = copy.deepcopy(pair)
            pair2.set("name", "r2_" + name)
            if g1 != "floor":
                pair2.set("geom1", "r2_" + g1)
            if g2 != "floor":
                pair2.set("geom2", "r2_" + g2)
            contact_section.append(pair2)

    tree.write(output_xml, pretty_print=True, xml_declaration=False)
    print(f"  Generated: {output_xml}")


def generate_dual_trajectory(case_info: dict) -> bool:
    """Generate dual trajectory for one case."""
    task = case_info["task"]
    p1_path = os.path.join(HOLOSOMA_DIR, case_info["p1_file"]) if case_info["p1_file"] else None
    p2_path = os.path.join(HOLOSOMA_DIR, case_info["p2_file"]) if case_info["p2_file"] else None

    if p1_path and not os.path.exists(p1_path):
        print(f"  SKIP: {p1_path} not found")
        return False
    if p2_path and not os.path.exists(p2_path):
        print(f"  SKIP: {p2_path} not found")
        return False

    # Load person data
    if p1_path and p2_path:
        d1 = np.load(p1_path)
        d2 = np.load(p2_path)
        qpos1 = d1["qpos"]  # (T, 43) = robot(36) + obj(7)
        qpos2 = d2["qpos"]
    elif p2_path:
        # desk005: only person2 exists. Use person2 as R1 (primary), mirror for R2
        d2 = np.load(p2_path)
        qpos2 = d2["qpos"]
        # Create a "standing" person1 as R2 (just copy pelvis + default joints)
        qpos1 = np.zeros_like(qpos2)
        qpos1[:, :3] = qpos2[:, :3]  # Same base position
        qpos1[:, 0] += 1.0  # Offset X by 1m
        qpos1[:, 3:7] = [1, 0, 0, 0]  # Identity quaternion
        qpos1[:, -7:] = qpos2[:, -7:]  # Same object
        print(f"  WARNING: desk005 has no person1, using offset standing pose as R2")
    else:
        print(f"  SKIP: no data")
        return False

    T = min(qpos1.shape[0], qpos2.shape[0])
    qpos1 = qpos1[:T]
    qpos2 = qpos2[:T]

    # Combine: [R1_base(7) + R1_joints(29) + R2_base(7) + R2_joints(29) + Obj(7)] = 79
    qpos_dual = np.zeros((T, 79), dtype=np.float64)
    qpos_dual[:, 0:36] = qpos1[:, 0:36]     # R1 robot
    qpos_dual[:, 36:72] = qpos2[:, 0:36]    # R2 robot
    qpos_dual[:, 72:79] = qpos1[:, 36:43]   # Object (from person1)

    # Compute qvel
    dt = 1.0 / 30.0
    qvel_dual = compute_qvel(qpos_dual, dt)

    # Ctrl = joint positions (no base) for both robots
    ctrl_dual = np.zeros((T, 58), dtype=np.float64)
    ctrl_dual[:, 0:29] = qpos_dual[:, 7:36]    # R1 joints
    ctrl_dual[:, 29:58] = qpos_dual[:, 43:72]  # R2 joints

    # Create output directories
    dual_dir = f"{SPIDER_DIR}/dual_humanoid_object/{task}"
    data_dir = f"{dual_dir}/0"
    os.makedirs(data_dir, exist_ok=True)

    # Generate scene_dual_robot.xml from single-person scene
    single_scene = f"{SPIDER_DIR}/humanoid_object/{task}/scene.xml"
    if not os.path.exists(single_scene):
        # Try the other person's scene
        alt_task = task.replace("person1", "person2") if "person1" in task else task.replace("person2", "person1")
        single_scene = f"{SPIDER_DIR}/humanoid_object/{alt_task}/scene.xml"
        if not os.path.exists(single_scene):
            print(f"  SKIP: no single scene.xml found")
            return False

    dual_scene = f"{dual_dir}/scene_dual_robot.xml"
    generate_scene_dual_robot(single_scene, dual_scene)

    # Compute contact positions using the dual scene
    contact_pos = compute_contact_pos(qpos_dual, dual_scene)

    # Contact indicator: 1 if hand is within 0.5m of object center
    obj_pos = qpos_dual[:, 72:75]
    contact = np.zeros((T, 4), dtype=np.float64)
    for h in range(4):
        dist = np.linalg.norm(contact_pos[:, h] - obj_pos, axis=1)
        contact[:, h] = (dist < 0.5).astype(float)

    # Save dual trajectory
    out_path = f"{data_dir}/trajectory_kinematic_dual.npz"
    np.savez(
        out_path,
        qpos=qpos_dual,
        qvel=qvel_dual,
        ctrl=ctrl_dual,
        contact=contact,
        contact_pos=contact_pos,
    )
    print(f"  Saved: {out_path} (T={T}, qpos={qpos_dual.shape})")

    return True


def main():
    for case_name, case_info in CASES.items():
        print(f"\n=== {case_name} ===")
        success = generate_dual_trajectory(case_info)
        if not success:
            print(f"  FAILED")


if __name__ == "__main__":
    main()
