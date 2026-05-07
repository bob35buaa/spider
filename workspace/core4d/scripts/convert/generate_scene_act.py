#!/usr/bin/env python3
"""Generate scene_act.xml for CORE4D cases.

Converts object body from freejoint to 3-slide + 3-hinge joints with position actuators.
Auto-selects the best euler convention per case to avoid gimbal lock.

Usage:
    uv run workspace/core4d/scripts/convert/generate_scene_act.py
"""

import json
import os
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

CASES = {
    "box025_person1": "box025_person1",
    "bucket010_person1": "bucket010_person1",
    "desk005_person2": "desk005_person2",
    "chair022_person1": "chair022_person1",
}

BASE = "example_datasets/processed/core4d/unitree_g1/humanoid_object"

# Euler conventions to try (extrinsic = MuJoCo hinge order)
EULER_CONVENTIONS = ["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"]

# Axis vectors for each letter
AXIS_MAP = {"X": "1 0 0", "Y": "0 1 0", "Z": "0 0 1"}


def find_best_euler_convention(task: str) -> str:
    """Find the euler convention that minimizes max |middle angle| over all frames."""
    ref_path = f"{BASE}/{task}/0/trajectory_kinematic_anchored.npz"
    if not os.path.exists(ref_path):
        ref_path = f"{BASE}/{task}/0/trajectory_kinematic.npz"
    ref = np.load(ref_path)
    quats_wxyz = ref["qpos"][:, 39:43]  # object quat (wxyz)

    best_conv = "XYZ"
    best_max_mid = 999.0

    for conv in EULER_CONVENTIONS:
        max_mid = 0.0
        for t in range(quats_wxyz.shape[0]):
            q = quats_wxyz[t]
            qx = [q[1], q[2], q[3], q[0]]  # wxyz → xyzw
            e = R.from_quat(qx).as_euler(conv)
            max_mid = max(max_mid, abs(e[1]))  # middle axis is gimbal lock axis
        if max_mid < best_max_mid:
            best_max_mid = max_mid
            best_conv = conv

    print(f"    Best euler convention: {best_conv} (max |mid| = {np.degrees(best_max_mid):.1f}°)")
    return best_conv


def generate_scene_act(task: str) -> tuple[str, str]:
    """Generate scene_act.xml from scene.xml for a given task.
    Returns (output_path, euler_convention).
    """
    scene_dir = f"{BASE}/{task}"
    scene_path = f"{scene_dir}/scene.xml"
    output_path = f"{scene_dir}/scene_act.xml"

    # Find best euler convention for this case
    euler_conv = find_best_euler_convention(task)

    tree = ET.parse(scene_path)
    root = tree.getroot()

    # Find object body
    object_body = None
    for body in root.iter("body"):
        if body.get("name") == "object":
            object_body = body
            break

    if object_body is None:
        raise ValueError(f"No 'object' body found in {scene_path}")

    # Remove freejoint from object body
    for fj in list(object_body.findall("freejoint")):
        object_body.remove(fj)

    # Add 6 joints: 3 slide (always XYZ) + 3 hinge (euler convention order)
    rot_axes = list(euler_conv)  # e.g. "ZYX" → ["Z", "Y", "X"]
    joint_defs = [
        ("object_pos_x", "slide", "1 0 0"),
        ("object_pos_y", "slide", "0 1 0"),
        ("object_pos_z", "slide", "0 0 1"),
        (f"object_rot_{rot_axes[0].lower()}", "hinge", AXIS_MAP[rot_axes[0]]),
        (f"object_rot_{rot_axes[1].lower()}", "hinge", AXIS_MAP[rot_axes[1]]),
        (f"object_rot_{rot_axes[2].lower()}", "hinge", AXIS_MAP[rot_axes[2]]),
    ]
    for i, (name, jtype, axis) in enumerate(joint_defs):
        object_body.insert(
            i,
            ET.Element("joint", {
                "name": name, "type": jtype, "axis": axis,
                "armature": "0.01", "damping": "100" if jtype == "slide" else "20",
            }),
        )

    # Add 6 position actuators
    actuator_elem = root.find("actuator")
    if actuator_elem is None:
        actuator_elem = ET.SubElement(root, "actuator")

    for name, _, _ in joint_defs:
        kp = "0"
        ET.SubElement(actuator_elem, "position", {
            "name": name, "joint": name, "kp": kp, "kv": "0",
        })

    # Write output
    tree.write(output_path, encoding="unicode")

    # Save euler convention metadata
    meta_path = f"{scene_dir}/scene_act_meta.json"
    with open(meta_path, "w") as f:
        json.dump({"euler_convention": euler_conv}, f)

    # Validate
    model = mujoco.MjModel.from_xml_path(output_path)
    print(f"  {task}: nq={model.nq}, nv={model.nv}, nu={model.nu}, euler={euler_conv} (wrote {output_path})")
    return output_path, euler_conv


def main():
    print("Generating scene_act.xml for CORE4D cases...")
    for case, task in CASES.items():
        try:
            generate_scene_act(task)
        except Exception as e:
            print(f"  {task}: ERROR — {e}")


if __name__ == "__main__":
    main()
