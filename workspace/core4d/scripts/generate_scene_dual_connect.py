#!/usr/bin/env python3
"""Generate scene_dual_robot_connect.xml: dual-robot scene + connect equality constraints.

Extends scene_dual_robot.xml with MuJoCo `connect` constraints between each robot's
wrist links and the object, so that hands are "attached" to the box surface.

Also generates a 2-connect variant (one hand per robot) for reduced over-constraint.

Usage:
    python workspace/core4d/scripts/generate_scene_dual_connect.py \
        --task box025_person1 [--solref "-500 -50"] [--mode all]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from lxml import etree
from scipy.spatial.transform import Rotation as R


def compute_contact_local_dual(
    npz_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract 4 hand contact positions in object local frame from dual kinematic ref.

    Layout: qpos (T, 79) = R1_base(7) + R1_joints(29) + R2_base(7) + R2_joints(29) + Object(7)
    contact_pos (T, 4, 3) = R1_left, R1_right, R2_left, R2_right

    Returns:
        r1_left, r1_right, r2_left, r2_right: each (3,) in object local frame
    """
    data = np.load(npz_path)
    qpos = data["qpos"]  # (T, 79)
    contact_pos = data["contact_pos"]  # (T, 4, 3)

    obj_pos = qpos[:, 72:75]
    obj_quat_wxyz = qpos[:, 75:79]

    T = qpos.shape[0]
    contact_local = np.zeros((T, 4, 3))
    for t in range(T):
        qw, qx, qy, qz = obj_quat_wxyz[t]
        rot = R.from_quat([qx, qy, qz, qw])
        for h in range(4):
            delta = contact_pos[t, h] - obj_pos[t]
            contact_local[t, h] = rot.inv().apply(delta)

    # Use frames where hand is close to object
    dists = np.linalg.norm(contact_pos - obj_pos[:, None, :], axis=-1)  # (T, 4)
    box_half = np.array([0.305, 0.305, 0.446])

    results = []
    for h in range(4):
        close = dists[:, h] < 0.8
        if close.any():
            median = np.median(contact_local[close, h], axis=0)
        else:
            median = contact_local[:, h].mean(axis=0)
        # Clamp to box surface
        clamped = median.copy()
        clamped[0] = np.clip(clamped[0], -box_half[0], box_half[0])
        clamped[1] = np.clip(clamped[1], -box_half[1], box_half[1])
        clamped[2] = np.clip(clamped[2], -box_half[2] * 0.5, box_half[2] * 0.5)
        results.append(clamped)

    return results[0], results[1], results[2], results[3]


def generate_dual_connect_scene(
    base_xml: Path,
    output_xml: Path,
    contacts: dict[str, np.ndarray],
    solref: str = "-500 -50",
    solimp: str = "0.95 0.99 0.001 0.5 2",
) -> None:
    """Generate dual-robot scene XML with connect equality constraints.

    Args:
        base_xml: Path to scene_dual_robot.xml
        output_xml: Path to output XML
        contacts: dict mapping hand label to (3,) contact position in object local frame
            Keys: "r1_left", "r1_right", "r2_left", "r2_right"
        solref: MuJoCo solref string
        solimp: MuJoCo solimp string
    """
    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(str(base_xml), parser)
    root = tree.getroot()

    obj_body = root.find(".//body[@name='object']")
    if obj_body is None:
        raise ValueError("No body named 'object' in XML")

    # Mapping from contact label to wrist body name
    wrist_map = {
        "r1_left": "left_wrist_yaw_link",
        "r1_right": "right_wrist_yaw_link",
        "r2_left": "r2_left_wrist_yaw_link",
        "r2_right": "r2_right_wrist_yaw_link",
    }

    # Add child bodies on object at contact positions (massless anchors)
    for label, pos in contacts.items():
        child = etree.SubElement(
            obj_body,
            "body",
            {
                "name": f"obj_contact_{label}",
                "pos": f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}",
            },
        )
        etree.SubElement(
            child,
            "site",
            {
                "name": f"obj_connect_site_{label}",
                "size": "0.02",
                "rgba": "0 1 0 0.5",
            },
        )

    # Add equality section with connect constraints
    eq = root.find("equality")
    if eq is None:
        eq = etree.SubElement(root, "equality")

    for label in contacts:
        wrist_body = wrist_map[label]
        etree.SubElement(
            eq,
            "connect",
            {
                "name": f"hand_obj_{label}",
                "body1": wrist_body,
                "body2": f"obj_contact_{label}",
                "anchor": "0.08 0 0",  # palm offset in wrist frame
                "solref": solref,
                "solimp": solimp,
            },
        )

    tree.write(str(output_xml), xml_declaration=True, encoding="utf-8", pretty_print=True)
    print(f"Written: {output_xml}")
    print(f"  Constraints: {list(contacts.keys())}")
    print(f"  solref: {solref}, solimp: {solimp}")
    for label, pos in contacts.items():
        print(f"  {label}: obj_local = [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dual robot connect scene XMLs")
    parser.add_argument("--task", default="box025_person1")
    parser.add_argument("--solref", default="-500 -50", help="Constraint stiffness (softer than single-robot)")
    parser.add_argument("--solimp", default="0.95 0.99 0.001 0.5 2", help="Constraint impedance")
    parser.add_argument("--mode", default="all", choices=["4connect", "2connect", "all"],
                        help="4connect: all 4 hands; 2connect: 1 hand per robot; all: both")
    parser.add_argument("--data-root",
                        default="example_datasets/processed/core4d/unitree_g1/dual_humanoid_object")
    args = parser.parse_args()

    task_dir = Path(args.data_root) / args.task
    npz_path = task_dir / "0" / "trajectory_kinematic_dual.npz"
    base_xml = task_dir / "scene_dual_robot.xml"

    if not npz_path.exists():
        raise FileNotFoundError(f"Dual kinematic NPZ not found: {npz_path}")
    if not base_xml.exists():
        raise FileNotFoundError(f"Base scene XML not found: {base_xml}")

    # Compute contact points
    r1_left, r1_right, r2_left, r2_right = compute_contact_local_dual(str(npz_path))

    print("Contact points (object local frame):")
    for label, pos in [("R1_left", r1_left), ("R1_right", r1_right),
                       ("R2_left", r2_left), ("R2_right", r2_right)]:
        print(f"  {label}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")

    # Generate 4-connect scene
    if args.mode in ["4connect", "all"]:
        contacts_4 = {
            "r1_left": r1_left,
            "r1_right": r1_right,
            "r2_left": r2_left,
            "r2_right": r2_right,
        }
        output_4 = task_dir / "scene_dual_robot_connect.xml"
        generate_dual_connect_scene(base_xml, output_4, contacts_4, args.solref, args.solimp)

        # Verify
        import mujoco
        m = mujoco.MjModel.from_xml_path(str(output_4))
        print(f"\n4-connect verification: neq={m.neq}, nq={m.nq}, nv={m.nv}, nu={m.nu}")

    # Generate 2-connect scene (one dominant hand per robot to reduce over-constraint)
    if args.mode in ["2connect", "all"]:
        # R1's right hand (closer to -x face) + R2's right hand (closer to +x face)
        contacts_2 = {
            "r1_right": r1_right,
            "r2_right": r2_right,
        }
        output_2 = task_dir / "scene_dual_robot_connect2.xml"
        generate_dual_connect_scene(base_xml, output_2, contacts_2, args.solref, args.solimp)

        # Verify
        import mujoco
        m = mujoco.MjModel.from_xml_path(str(output_2))
        print(f"\n2-connect verification: neq={m.neq}, nq={m.nq}, nv={m.nv}, nu={m.nu}")


if __name__ == "__main__":
    main()
