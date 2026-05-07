#!/usr/bin/env python3
"""Generate scene_act.xml for CORE4D cases.

Converts object body from freejoint to 3-slide + 3-hinge joints with position actuators.
This allows SPIDER's contact_guidance mechanism to drive the object along reference trajectory
with proper orientation control (each rotation axis is independently PD-controlled).

Usage:
    uv run workspace/core4d/scripts/convert/generate_scene_act.py
"""

import os
import xml.etree.ElementTree as ET

import mujoco

CASES = {
    "box025_person1": "box025_person1",
    "bucket010_person1": "bucket010_person1",
    "desk005_person2": "desk005_person2",
    "chair022_person1": "chair022_person1",
}

BASE = "example_datasets/processed/core4d/unitree_g1/humanoid_object"


def generate_scene_act(task: str) -> str:
    """Generate scene_act.xml from scene.xml for a given task."""
    scene_dir = f"{BASE}/{task}"
    scene_path = f"{scene_dir}/scene.xml"
    output_path = f"{scene_dir}/scene_act.xml"

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

    # Add 6 slide/hinge joints (same as HDMI/contact_guidance pattern)
    joint_defs = [
        ("object_pos_x", "slide", "1 0 0"),
        ("object_pos_y", "slide", "0 1 0"),
        ("object_pos_z", "slide", "0 0 1"),
        ("object_rot_x", "hinge", "1 0 0"),
        ("object_rot_y", "hinge", "0 1 0"),
        ("object_rot_z", "hinge", "0 0 1"),
    ]
    for i, (name, jtype, axis) in enumerate(joint_defs):
        object_body.insert(
            i,
            ET.Element("joint", {
                "name": name, "type": jtype, "axis": axis,
                "armature": "0.01", "frictionloss": "0.01",
            }),
        )

    # Add 6 position actuators (kp/kv=0, set at runtime by SPIDER contact_guidance)
    actuator_elem = root.find("actuator")
    if actuator_elem is None:
        actuator_elem = ET.SubElement(root, "actuator")

    for name, _, _ in joint_defs:
        ET.SubElement(actuator_elem, "position", {
            "name": name, "joint": name, "kp": "0", "kv": "0",
        })

    # Write output
    tree.write(output_path, encoding="unicode")

    # Validate
    model = mujoco.MjModel.from_xml_path(output_path)
    print(f"  {task}: nq={model.nq}, nv={model.nv}, nu={model.nu} (wrote {output_path})")
    return output_path


def main():
    print("Generating scene_act.xml for CORE4D cases...")
    for case, task in CASES.items():
        try:
            generate_scene_act(task)
        except Exception as e:
            print(f"  {task}: ERROR — {e}")


if __name__ == "__main__":
    main()
