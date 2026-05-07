#!/usr/bin/env python3
"""Generate scene_dual_robot_connect2.xml for all CORE4D dual cases.

Reads object geom size from scene XML and contact positions from dual trajectory
to compute connect constraint anchor points automatically.

Usage:
    python workspace/core4d/scripts/convert/generate_dual_connect_all.py
"""

import numpy as np
from lxml import etree
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import mujoco


SPIDER_DIR = "example_datasets/processed/core4d/unitree_g1"

CASES = ["box025_person1", "bucket010_person1", "chair022_person1", "desk005_person2"]

SOLREF = "-200 -30"   # Soft constraint (E017-d best)
SOLIMP = "0.9 0.95 0.01 0.5 2"


def get_object_half_extents(scene_xml: str) -> np.ndarray:
    """Read object collision geom size from scene XML."""
    model = mujoco.MjModel.from_xml_path(scene_xml)
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_collision")
    if geom_id == -1:
        # Try finding any geom with "object" and collision
        for g in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
            if name and "object" in name and model.geom_contype[g] > 0:
                geom_id = g
                break
    if geom_id == -1:
        print("  WARNING: no object collision geom found, using default")
        return np.array([0.3, 0.3, 0.3])
    return model.geom_size[geom_id].copy()


def compute_connect_anchors(dual_npz: str, obj_half: np.ndarray) -> dict:
    """Compute 2-connect anchor positions (one hand per robot, closest to object).

    Returns dict with keys "r1_right", "r2_right" (or whichever hand is closer).
    """
    data = np.load(dual_npz)
    qpos = data["qpos"]       # (T, 79)
    contact_pos = data["contact_pos"]  # (T, 4, 3) = R1L, R1R, R2L, R2R

    obj_pos = qpos[:, 72:75]
    obj_quat_wxyz = qpos[:, 75:79]

    T = qpos.shape[0]
    hand_labels = ["r1_left", "r1_right", "r2_left", "r2_right"]

    # Compute contact positions in object local frame
    contact_local = np.zeros((T, 4, 3))
    for t in range(T):
        qw, qx, qy, qz = obj_quat_wxyz[t]
        rot = R.from_quat([qx, qy, qz, qw])
        for h in range(4):
            delta = contact_pos[t, h] - obj_pos[t]
            contact_local[t, h] = rot.inv().apply(delta)

    # For each robot, pick the hand closest to object (mean distance)
    dists = np.linalg.norm(contact_pos - obj_pos[:, None, :], axis=-1).mean(axis=0)  # (4,)
    r1_hand = 0 if dists[0] < dists[1] else 1  # R1: left(0) or right(1)
    r2_hand = 2 if dists[2] < dists[3] else 3  # R2: left(2) or right(3)

    anchors = {}
    for idx in [r1_hand, r2_hand]:
        # Use median of frames where hand is reasonably close
        close_mask = np.linalg.norm(contact_pos[:, idx] - obj_pos, axis=1) < 1.0
        if close_mask.sum() > 5:
            median = np.median(contact_local[close_mask, idx], axis=0)
        else:
            median = contact_local[:, idx].mean(axis=0)

        # Clamp to object surface
        clamped = median.copy()
        clamped[0] = np.clip(clamped[0], -obj_half[0], obj_half[0])
        clamped[1] = np.clip(clamped[1], -obj_half[1], obj_half[1])
        clamped[2] = np.clip(clamped[2], -obj_half[2] * 0.8, obj_half[2] * 0.8)

        anchors[hand_labels[idx]] = clamped

    return anchors


def add_connect_constraints(scene_xml: str, output_xml: str, anchors: dict):
    """Add 2-connect constraints to dual robot scene."""
    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(scene_xml, parser)
    root = tree.getroot()

    obj_body = root.find(".//body[@name='object']")
    if obj_body is None:
        print(f"  ERROR: no 'object' body in {scene_xml}")
        return

    wrist_map = {
        "r1_left": "left_wrist_yaw_link",
        "r1_right": "right_wrist_yaw_link",
        "r2_left": "r2_left_wrist_yaw_link",
        "r2_right": "r2_right_wrist_yaw_link",
    }

    # Add anchor sites on object
    for label, pos in anchors.items():
        child = etree.SubElement(obj_body, "body", {
            "name": f"obj_contact_{label}",
            "pos": f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}",
        })
        etree.SubElement(child, "site", {
            "name": f"obj_connect_site_{label}",
            "size": "0.02",
            "rgba": "0 1 0 0.5",
        })

    # Add equality constraints
    eq = root.find("equality")
    if eq is None:
        eq = etree.SubElement(root, "equality")

    for label in anchors:
        wrist_body = wrist_map[label]
        etree.SubElement(eq, "connect", {
            "name": f"hand_obj_{label}",
            "body1": wrist_body,
            "body2": f"obj_contact_{label}",
            "anchor": "0.08 0 0",
            "solref": SOLREF,
            "solimp": SOLIMP,
        })

    tree.write(output_xml, pretty_print=True, xml_declaration=False)
    print(f"  Generated: {output_xml}")
    for label, pos in anchors.items():
        print(f"    {label}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")


def main():
    for case in CASES:
        print(f"\n=== {case} ===")

        dual_dir = f"{SPIDER_DIR}/dual_humanoid_object/{case}"
        dual_scene = f"{dual_dir}/scene_dual_robot.xml"
        dual_npz = f"{dual_dir}/0/trajectory_kinematic_dual.npz"
        connect_scene = f"{dual_dir}/scene_dual_robot_connect2.xml"

        if not Path(dual_scene).exists():
            print(f"  SKIP: {dual_scene} not found")
            continue
        if not Path(dual_npz).exists():
            print(f"  SKIP: {dual_npz} not found")
            continue

        # Get object size from single-person scene
        single_scene = f"{SPIDER_DIR}/humanoid_object/{case}/scene.xml"
        if not Path(single_scene).exists():
            alt = case.replace("person1", "person2") if "person1" in case else case.replace("person2", "person1")
            single_scene = f"{SPIDER_DIR}/humanoid_object/{alt}/scene.xml"
        obj_half = get_object_half_extents(single_scene)
        print(f"  Object half extents: {obj_half}")

        # Compute connect anchors
        anchors = compute_connect_anchors(dual_npz, obj_half)

        # Generate connect scene
        add_connect_constraints(dual_scene, connect_scene, anchors)


if __name__ == "__main__":
    main()
