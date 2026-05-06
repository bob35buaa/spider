#!/usr/bin/env python3
"""Generate scene_connect.xml: scene_forearm.xml + connect equality constraints.

Adds MuJoCo `connect` constraints between G1's wrist links and the object,
so that hands are "attached" to the object surface. This removes the grasp
variable and isolates the question: can G1's kinematics complete the carry motion?

Usage:
    python workspace/core4d/scripts/generate_scene_connect.py \
        --task box025_person1 [--solref "-500 -50"]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from lxml import etree
from scipy.spatial.transform import Rotation as R


def compute_contact_local(npz_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Extract hand contact positions in object local frame from kinematic ref.

    Returns:
        left_local:  (3,) median contact pos of left hand in object frame
        right_local: (3,) median contact pos of right hand in object frame
    """
    data = np.load(npz_path)
    qpos = data["qpos"]  # (T, 43)
    contact_pos = data["contact_pos"]  # (T, 2, 3) world

    obj_pos = qpos[:, 36:39]
    obj_quat_wxyz = qpos[:, 39:43]

    T = qpos.shape[0]
    contact_local = np.zeros((T, 2, 3))
    for t in range(T):
        qw, qx, qy, qz = obj_quat_wxyz[t]
        rot = R.from_quat([qx, qy, qz, qw])
        for h in range(2):
            delta = contact_pos[t, h] - obj_pos[t]
            contact_local[t, h] = rot.inv().apply(delta)

    # Use frames where hand is within 0.6m of object center
    dists = np.linalg.norm(contact_pos - obj_pos[:, None, :], axis=-1)
    close_l = dists[:, 0] < 0.6
    close_r = dists[:, 1] < 0.6

    left_local = np.median(contact_local[close_l, 0], axis=0) if close_l.any() else contact_local[:, 0].mean(axis=0)
    right_local = np.median(contact_local[close_r, 1], axis=0) if close_r.any() else contact_local[:, 1].mean(axis=0)

    return left_local, right_local


def generate_connect_scene(
    base_xml: Path,
    output_xml: Path,
    contact_left: np.ndarray,
    contact_right: np.ndarray,
    solref: str = "-500 -50",
) -> None:
    """Generate scene XML with connect equality constraints.

    MuJoCo `connect` constrains hand_palm → body2_origin. To connect the palm
    to a specific point on the object surface, we add **child bodies** on the
    object at the contact positions. These massless child bodies move rigidly
    with the object and serve as anchor targets for the connect constraint.

    Args:
        base_xml: Path to scene_forearm.xml
        output_xml: Path to write scene_connect.xml
        contact_left: (3,) left hand contact point in object local frame
        contact_right: (3,) right hand contact point in object local frame
        solref: MuJoCo solref string for constraint stiffness
    """
    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(str(base_xml), parser)
    root = tree.getroot()

    obj_body = root.find(".//body[@name='object']")
    if obj_body is None:
        raise ValueError("No body named 'object' in XML")

    # Add child bodies on the object at contact positions.
    # These are massless (inertial from parent), just position anchors.
    for side, pos in [("left", contact_left), ("right", contact_right)]:
        child = etree.SubElement(
            obj_body,
            "body",
            {
                "name": f"obj_contact_{side}",
                "pos": f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}",
            },
        )
        etree.SubElement(
            child,
            "site",
            {
                "name": f"obj_connect_site_{side}",
                "size": "0.02",
                "rgba": "1 0 0 0.5",
            },
        )

    # Add equality section with connect constraints.
    # connect body1=wrist_yaw_link body2=obj_contact_{side}
    # anchor = palm position in wrist_yaw_link frame (0.08, 0, 0)
    # This constrains: palm_point ≡ obj_contact_{side}_origin
    eq = root.find("equality")
    if eq is None:
        eq = etree.SubElement(root, "equality")

    for side in ["left", "right"]:
        etree.SubElement(
            eq,
            "connect",
            {
                "name": f"hand_obj_{side}",
                "body1": f"{side}_wrist_yaw_link",
                "body2": f"obj_contact_{side}",
                "anchor": "0.08 0 0",
                "solref": solref,
                "solimp": "0.95 0.99 0.001 0.5 2",
            },
        )

    tree.write(str(output_xml), xml_declaration=True, encoding="utf-8", pretty_print=True)
    print(f"Written: {output_xml}")
    print(f"  Left contact (obj local):  {contact_left}")
    print(f"  Right contact (obj local): {contact_right}")
    print(f"  solref: {solref}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate scene_connect.xml")
    parser.add_argument("--task", default="box025_person1")
    parser.add_argument("--solref", default="-500 -50", help="Constraint stiffness")
    parser.add_argument("--anchor-mode", default="auto", choices=["auto", "side_mid", "ref"],
                        help="auto: clamp ref contacts to box sides; side_mid: box ±x face center; ref: raw ref data")
    parser.add_argument("--data-root", default="example_datasets/processed/core4d/unitree_g1/humanoid_object")
    args = parser.parse_args()

    task_dir = Path(args.data_root) / args.task
    npz_path = task_dir / "0" / "trajectory_kinematic.npz"
    base_xml = task_dir / "scene_forearm.xml"
    output_xml = task_dir / "scene_connect.xml"

    if not npz_path.exists():
        raise FileNotFoundError(f"Kinematic NPZ not found: {npz_path}")
    if not base_xml.exists():
        raise FileNotFoundError(f"Base scene XML not found: {base_xml}")

    # Compute contact points
    left_local, right_local = compute_contact_local(str(npz_path))

    print(f"Raw contact points (object local frame):")
    print(f"  Left:  {left_local}")
    print(f"  Right: {right_local}")

    # Box half-extents (from collision geom in scene XML)
    box_half = np.array([0.305, 0.305, 0.446])

    if args.anchor_mode == "side_mid":
        # Place anchors at box ±x face centers (mid-height)
        left_local = np.array([box_half[0], 0.0, 0.0])
        right_local = np.array([-box_half[0], 0.0, 0.0])
        print(f"  Using side_mid mode: anchors at ±x face centers")
    elif args.anchor_mode == "auto":
        # Keep x/y from ref but clamp z to be within box and above center
        # This ensures contact points are on the box surface, reachable by G1
        left_local[2] = np.clip(left_local[2], -box_half[2] * 0.3, box_half[2] * 0.3)
        right_local[2] = np.clip(right_local[2], -box_half[2] * 0.3, box_half[2] * 0.3)
        # Clamp to box surface on x axis
        left_local[0] = np.clip(left_local[0], -box_half[0], box_half[0])
        right_local[0] = np.clip(right_local[0], -box_half[0], box_half[0])
        print(f"  Using auto mode: clamped z to ±{box_half[2]*0.3:.3f}")

    print(f"Final contact points:")
    print(f"  Left:  {left_local}")
    print(f"  Right: {right_local}")

    # Generate scene
    generate_connect_scene(
        base_xml=base_xml,
        output_xml=output_xml,
        contact_left=left_local,
        contact_right=right_local,
        solref=args.solref,
    )

    # Verify the generated XML loads
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(output_xml))
    print(f"\nVerification: neq={m.neq}, nq={m.nq}, nv={m.nv}, nu={m.nu}")


if __name__ == "__main__":
    main()
