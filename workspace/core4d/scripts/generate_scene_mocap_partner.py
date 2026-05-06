#!/usr/bin/env python3
"""Generate scene_mocap_partner.xml: scene_forearm.xml + partner hand mocap bodies.

Adds 2 mocap bodies (capsules) representing person2's hands. These bodies:
- Follow a prescribed trajectory (partner_hand_pos/quat from holosoma data)
- Have collision geometry that can interact with the box
- Enable cooperative lifting when person1 (G1) pushes from one side

Usage:
    python workspace/core4d/scripts/generate_scene_mocap_partner.py --task box025_person1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from lxml import etree


def generate_mocap_partner_scene(
    base_xml: Path,
    output_xml: Path,
    partner_init_pos: np.ndarray,  # (2, 3) initial positions
    partner_init_quat: np.ndarray,  # (2, 4) initial quats (wxyz)
) -> None:
    """Generate scene XML with mocap partner hand bodies."""
    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(str(base_xml), parser)
    root = tree.getroot()

    worldbody = root.find("worldbody")

    # Add 2 mocap bodies for partner hands
    for i, (side, pos, quat) in enumerate([
        ("left", partner_init_pos[0], partner_init_quat[0]),
        ("right", partner_init_pos[1], partner_init_quat[1]),
    ]):
        body = etree.SubElement(
            worldbody,
            "body",
            {
                "name": f"partner_{side}_hand",
                "mocap": "true",
                "pos": f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}",
                "quat": f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}",
            },
        )
        # Capsule geom for collision with box
        etree.SubElement(
            body,
            "geom",
            {
                "name": f"partner_{side}_hand_geom",
                "type": "capsule",
                "size": "0.04 0.08",
                "rgba": "0.2 0.2 0.8 0.5",
                "contype": "1",
                "conaffinity": "1",
                "friction": "2 0.005 0.001",
                "condim": "4",
                "group": "3",
            },
        )
        etree.SubElement(
            body,
            "site",
            {"name": f"partner_{side}_hand_site", "size": "0.02", "rgba": "0 0 1 0.5"},
        )

    # Add contact pairs: partner hands ↔ object
    contact = root.find("contact")
    for side in ["left", "right"]:
        etree.SubElement(
            contact,
            "pair",
            {
                "name": f"partner_{side}_hand_object",
                "geom1": f"partner_{side}_hand_geom",
                "geom2": "object_collision",
                "solref": "0.008 1",
                "friction": "2 1",
                "condim": "4",
            },
        )

    tree.write(str(output_xml), xml_declaration=True, encoding="utf-8", pretty_print=True)
    print(f"Written: {output_xml}")


def prepare_partner_trajectory(holosoma_npz: str, spider_npz: str, output_npz: str) -> None:
    """Resample partner hand data from 50fps to 30fps and save for SPIDER.

    Adds 'partner_pos' (T, 2, 3) and 'partner_quat' (T, 2, 4) to the output.
    """
    holo = np.load(holosoma_npz)
    spider = np.load(spider_npz)

    partner_pos_50 = holo["partner_hand_pos_w"]  # (205, 2, 3) @ 50fps
    partner_quat_50 = holo["partner_hand_quat_w"]  # (205, 2, 4) @ 50fps

    T_spider = spider["qpos"].shape[0]  # 124 frames @ 30fps
    T_holo = partner_pos_50.shape[0]  # 205 frames @ 50fps

    # Resample from 50fps to 30fps using linear interpolation
    t_spider = np.arange(T_spider) / 30.0  # time axis for spider frames
    t_holo = np.arange(T_holo) / 50.0  # time axis for holosoma frames

    partner_pos_30 = np.zeros((T_spider, 2, 3))
    partner_quat_30 = np.zeros((T_spider, 2, 4))

    for h in range(2):
        for d in range(3):
            partner_pos_30[:, h, d] = np.interp(t_spider, t_holo, partner_pos_50[:, h, d])
        for d in range(4):
            partner_quat_30[:, h, d] = np.interp(t_spider, t_holo, partner_quat_50[:, h, d])
        # Normalize quaternions
        norms = np.linalg.norm(partner_quat_30[:, h], axis=-1, keepdims=True)
        partner_quat_30[:, h] /= norms

    # Save augmented data
    data = dict(spider)
    data["partner_pos"] = partner_pos_30
    data["partner_quat"] = partner_quat_30
    np.savez(output_npz, **data)
    print(f"Saved partner data: partner_pos {partner_pos_30.shape}, partner_quat {partner_quat_30.shape}")
    print(f"  Output: {output_npz}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mocap partner scene")
    parser.add_argument("--task", default="box025_person1")
    parser.add_argument("--data-root", default="example_datasets/processed/core4d/unitree_g1/humanoid_object")
    parser.add_argument("--holosoma-root", default="/home/ubuntu/Workspace/holosoma/workspace/v2/results/converted_for_rl_trimmed")
    args = parser.parse_args()

    task_dir = Path(args.data_root) / args.task
    base_xml = task_dir / "scene_forearm.xml"
    output_xml = task_dir / "scene_mocap_partner.xml"

    # Load partner data
    holosoma_npz = Path(args.holosoma_root) / "20231011-048-person1-Box025_v2_trimmed_mj_w_obj_w_partner.npz"
    holo = np.load(str(holosoma_npz))
    partner_pos = holo["partner_hand_pos_w"]  # (205, 2, 3) @ 50fps
    partner_quat = holo["partner_hand_quat_w"]  # (205, 2, 4) xyzw or wxyz?

    # Use first frame as initial position for mocap bodies
    init_pos = partner_pos[0]  # (2, 3)
    # Convert quaternion to wxyz (MuJoCo format) if needed
    # Holosoma uses xyzw (scipy convention), MuJoCo uses wxyz
    init_quat_xyzw = partner_quat[0]  # (2, 4)
    init_quat_wxyz = np.zeros_like(init_quat_xyzw)
    init_quat_wxyz[:, 0] = init_quat_xyzw[:, 3]  # w
    init_quat_wxyz[:, 1:] = init_quat_xyzw[:, :3]  # xyz

    print(f"Partner init positions: {init_pos}")
    print(f"Partner init quats (wxyz): {init_quat_wxyz}")

    # Generate scene XML
    generate_mocap_partner_scene(base_xml, output_xml, init_pos, init_quat_wxyz)

    # Prepare resampled partner trajectory for SPIDER
    spider_npz = task_dir / "0" / "trajectory_kinematic.npz"
    output_traj = task_dir / "0" / "trajectory_kinematic_partner.npz"
    prepare_partner_trajectory(str(holosoma_npz), str(spider_npz), str(output_traj))

    # Verify the generated XML
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(output_xml))
    print(f"\nVerification: nbody={m.nbody}, nmocap={m.nmocap}, nq={m.nq}, nv={m.nv}")

    # Find mocap body ids
    for i in range(m.nbody):
        name = m.body(i).name
        if "partner" in name:
            print(f"  Mocap body: {name} (id={i}, mocapid={m.body(i).mocapid[0]})")


if __name__ == "__main__":
    main()
