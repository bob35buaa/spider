"""Generate HDMI scene XMLs for CORE4D cases from suitcase template.

Takes the original HDMI suitcase scene.xml (with correct robot physics) and
replaces only the object body with CORE4D object geometry.

Usage:
    uv run workspace/core4d/scripts/convert/generate_hdmi_scene.py --case box023_person1
    uv run workspace/core4d/scripts/convert/generate_hdmi_scene.py --case box025_person1
    uv run workspace/core4d/scripts/convert/generate_hdmi_scene.py  # all cases
"""

import argparse
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

SPIDER_ROOT = Path(__file__).resolve().parents[4]
SUITCASE_SCENE = (
    SPIDER_ROOT
    / "example_datasets/processed/hdmi/unitree_g1/humanoid_object/move_suitcase/scene/mjlab scene.xml"
)
CORE4D_BASE = SPIDER_ROOT / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
HDMI_BASE = SPIDER_ROOT / "example_datasets/processed/hdmi/unitree_g1/humanoid_object"

# Map CORE4D case → HDMI task name
CASES = {
    "box023_person1": "move_box023",
    "box025_person1": "move_box025",
}


def get_object_info(case: str) -> dict:
    """Extract object mesh path and collision AABB from CORE4D scene.xml."""
    scene_path = CORE4D_BASE / case / "scene.xml"
    model = mujoco.MjModel.from_xml_path(str(scene_path))

    # Find object collision geom
    obj_half_ext = None
    for gi in range(model.ngeom):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gi)
        if gname and "object_collision" in gname:
            obj_half_ext = model.geom_size[gi].copy()
            break

    if obj_half_ext is None:
        raise ValueError(f"No object_collision geom found in {scene_path}")

    # Find object mesh path
    obj_name = case.split("_person")[0]  # e.g. "box023"
    mesh_path = CORE4D_BASE / "../../assets/objects" / obj_name / f"{obj_name}_m.obj"
    if not mesh_path.exists():
        # Try alternative path
        mesh_path = SPIDER_ROOT / "example_datasets/processed/core4d/assets/objects" / obj_name / f"{obj_name}_m.obj"

    return {
        "half_ext": obj_half_ext,
        "mesh_path": mesh_path,
        "obj_name": obj_name,
    }


def generate_scene(case: str):
    """Generate HDMI scene from suitcase template for a CORE4D case."""
    task_name = CASES[case]
    output_dir = HDMI_BASE / task_name / "scene"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "mjlab scene.xml"

    # Get object info
    info = get_object_info(case)
    half_ext = info["half_ext"]
    mesh_path = info["mesh_path"]
    obj_name = info["obj_name"]

    print(f"  Case: {case}")
    print(f"  Object: {obj_name}, half_ext={half_ext}")
    print(f"  Mesh: {mesh_path}")

    # Parse suitcase template
    tree = ET.parse(str(SUITCASE_SCENE))
    root = tree.getroot()

    # 1. Add mesh asset for the CORE4D object
    asset_elem = root.find("asset")
    if asset_elem is None:
        asset_elem = ET.SubElement(root, "asset")

    # Compute relative mesh path from output scene dir to the object mesh
    rel_mesh = os.path.relpath(str(mesh_path.resolve()), str(output_dir.resolve()))

    # Check if mesh already exists in assets
    existing_meshes = {m.get("name") for m in asset_elem.findall("mesh")}
    if obj_name not in existing_meshes:
        ET.SubElement(asset_elem, "mesh", {
            "name": obj_name,
            "file": rel_mesh,
            "scale": "1 1 1",
        })

    # Add material for visual mesh
    existing_materials = {m.get("name") for m in asset_elem.findall("material")}
    if "box_material" not in existing_materials:
        ET.SubElement(asset_elem, "material", {
            "name": "box_material",
            "rgba": "0.8 0.6 0.3 1",
        })

    # 2. Replace suitcase body content
    suitcase_body = None
    for body in root.find("worldbody").iter("body"):
        if "suitcase" in body.get("name", ""):
            suitcase_body = body
            break

    if suitcase_body is None:
        raise ValueError("suitcase body not found in template")

    # Keep body pos as-is (suitcase default: "0.4 0.05 0")
    # This provides the PD tracking force via slide offset

    # Remove existing geoms and inertial from suitcase body
    for child in list(suitcase_body):
        tag = child.tag
        if tag in ("geom", "inertial"):
            suitcase_body.remove(child)

    # Add new inertial (mass=2.0, compute from box AABB)
    # I = m/12 * (b^2+c^2, a^2+c^2, a^2+b^2) for box with half-extents a,b,c
    m = 2.0
    a, b, c = half_ext * 2  # full dimensions
    ixx = m / 12 * (b**2 + c**2)
    iyy = m / 12 * (a**2 + c**2)
    izz = m / 12 * (a**2 + b**2)

    # Insert inertial after joint
    joint_idx = None
    for i, child in enumerate(suitcase_body):
        if child.tag == "joint":
            joint_idx = i
            break

    insert_idx = (joint_idx + 1) if joint_idx is not None else 0
    inertial_elem = ET.Element("inertial", {
        "pos": "0 0 0",
        "mass": f"{m}",
        "diaginertia": f"{ixx:.5f} {iyy:.5f} {izz:.5f}",
    })
    suitcase_body.insert(insert_idx, inertial_elem)

    # Add visual mesh geom (no collision)
    visual_geom = ET.Element("geom", {
        "name": f"suitcase/{obj_name}_visual",
        "type": "mesh",
        "mesh": obj_name,
        "material": "box_material",
        "group": "2",
        "contype": "0",
        "conaffinity": "0",
    })
    suitcase_body.append(visual_geom)

    # Add collision box geom
    size_str = f"{half_ext[0]:.4f} {half_ext[1]:.4f} {half_ext[2]:.4f}"
    collision_geom = ET.Element("geom", {
        "name": "suitcase/suitcase_collision",
        "type": "box",
        "size": size_str,
        "rgba": "0.6 0.4 0.2 0.3",
        "group": "3",
        "contype": "1",
        "conaffinity": "1",
        "friction": "1 0.005 0.0001",
        "condim": "3",
    })
    suitcase_body.append(collision_geom)

    # 3. Update sensors to reference correct collision geom name
    # (sensors reference suitcase/suitcase body which is unchanged)

    # 4. Update keyframes — remove suitcase init state (will be set by HDMI code)
    keyframe_elem = root.find("keyframe")
    if keyframe_elem is not None:
        for key in list(keyframe_elem.findall("key")):
            if "suitcase" in key.get("name", ""):
                keyframe_elem.remove(key)

    # 5. Write output
    tree.write(str(output_path), encoding="unicode", xml_declaration=False)
    print(f"  Output: {output_path}")

    # 5b. Symlink robot mesh directory from suitcase scene
    robot_mesh_src = SUITCASE_SCENE.parent / "robot"
    robot_mesh_dst = output_dir / "robot"
    if robot_mesh_dst.exists():
        if robot_mesh_dst.is_symlink():
            robot_mesh_dst.unlink()
        else:
            shutil.rmtree(str(robot_mesh_dst))
    robot_mesh_dst.symlink_to(robot_mesh_src.resolve())
    print(f"  Symlinked: robot/ → {robot_mesh_src}")

    # 6. Verify MuJoCo can load it
    try:
        m = mujoco.MjModel.from_xml_path(str(output_path))
        print(f"  Verified: nq={m.nq}, nv={m.nv}, nu={m.nu}, nbody={m.nbody}")
    except Exception as e:
        print(f"  [ERROR] MuJoCo load failed: {e}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, default=None, help="Specific case (e.g. box023_person1)")
    args = parser.parse_args()

    cases = [args.case] if args.case else list(CASES.keys())

    print(f"Generating HDMI scenes from suitcase template")
    print(f"Template: {SUITCASE_SCENE}")
    print(f"Cases: {cases}")
    print()

    for case in cases:
        if case not in CASES:
            print(f"  [SKIP] Unknown case: {case}")
            continue
        print(f"--- {case} ---")
        generate_scene(case)
        print()


if __name__ == "__main__":
    main()
