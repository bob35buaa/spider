#!/usr/bin/env python3
"""Generate scene.xml files for new CORE4D cases (bucket010, chair022, desk005).

Uses bucket005 scene.xml as template, modifying:
- Object mesh reference
- Object initial pos/quat (from holosoma source data)
- Object collision geometry (bounding box approximation)
- Object mass/inertia
"""

import os
import re

import numpy as np
import trimesh

ROOT = "/home/ubuntu/Workspace/spider"
PROCESSED = f"{ROOT}/example_datasets/processed/core4d/unitree_g1/humanoid_object"
OBJ_MODELS = f"{ROOT}/workspace/core4d/object_models/object_models"
HOLOSOMA_DATA = "/home/ubuntu/Workspace/holosoma/workspace/v2/results/retarget_replace_batch_trimmed"

# Template: use bucket005 scene.xml
TEMPLATE_PATH = f"{PROCESSED}/bucket005_person1/scene.xml"

CASES = {
    "bucket010_person1": {
        "source_npz": f"{HOLOSOMA_DATA}/20231003_2-059-person1-bucket010_with_obj_original.npz",
        "mesh_file": "bucket010_m.obj",
        "mesh_dir": "bucket010",
        "mesh_name": "bucket010",
        "material_name": "bucket010_material",
        "material_rgba": "0.5 0.4 0.3 1",
        "mass": 3.0,  # larger bucket
    },
    "chair022_person1": {
        "source_npz": f"{HOLOSOMA_DATA}/20231020-084-person1-chair022_with_obj_original.npz",
        "mesh_file": "chair022_m.obj",
        "mesh_dir": "chair022",
        "mesh_name": "chair022",
        "material_name": "chair022_material",
        "material_rgba": "0.6 0.4 0.2 1",
        "mass": 4.0,  # chair is heavier
    },
    "desk005_person2": {
        "source_npz": f"{HOLOSOMA_DATA}/20231023-030-person2-desk005_with_obj_original.npz",
        "mesh_file": "desk005_m.obj",
        "mesh_dir": "desk005",
        "mesh_name": "desk005",
        "material_name": "desk005_material",
        "material_rgba": "0.6 0.5 0.3 1",
        "mass": 5.0,  # desk is heavy
    },
}


def get_obj_bbox(mesh_dir: str, mesh_file: str) -> np.ndarray:
    """Get half-extents of object bounding box."""
    path = f"{ROOT}/example_datasets/processed/core4d/assets/objects/{mesh_dir}/{mesh_file}"
    mesh = trimesh.load(path)
    extents = mesh.bounding_box.extents
    return extents / 2  # half-extents for MuJoCo


def get_initial_obj_state(source_npz: str) -> tuple:
    """Get initial object pos and quat from source data."""
    data = np.load(source_npz, allow_pickle=True)
    qpos = data["qpos"]
    # Object DOFs are last 7: pos(3) + quat(4)
    obj_pos = qpos[0, 36:39]
    obj_quat = qpos[0, 39:43]
    return obj_pos, obj_quat


def compute_inertia(mass: float, half_extents: np.ndarray) -> np.ndarray:
    """Compute box inertia tensor diagonal."""
    a, b, c = half_extents * 2  # full extents
    Ix = mass / 12 * (b**2 + c**2)
    Iy = mass / 12 * (a**2 + c**2)
    Iz = mass / 12 * (a**2 + b**2)
    return np.array([Ix, Iy, Iz])


def generate_scene_xml(task: str, config: dict) -> str:
    """Generate scene.xml for a new case based on bucket005 template."""
    with open(TEMPLATE_PATH) as f:
        template = f.read()

    # Get object properties
    half_ext = get_obj_bbox(config["mesh_dir"], config["mesh_file"])
    obj_pos, obj_quat = get_initial_obj_state(config["source_npz"])
    inertia = compute_inertia(config["mass"], half_ext)

    # Replace mesh asset declaration
    template = re.sub(
        r'<mesh name="bucket005".*?/>',
        f'<mesh name="{config["mesh_name"]}" '
        f'file="../../../../../example_datasets/processed/core4d/assets/objects/{config["mesh_dir"]}/{config["mesh_file"]}" '
        f'scale="1 1 1" />',
        template,
    )

    # Replace material
    template = re.sub(
        r'<material name="bucket_material".*?/>',
        f'<material name="{config["material_name"]}" rgba="{config["material_rgba"]}" />',
        template,
    )

    # Replace object body
    pos_str = f"{obj_pos[0]:.4f} {obj_pos[1]:.4f} {obj_pos[2]:.4f}"
    quat_str = f"{obj_quat[0]:.6f} {obj_quat[1]:.6f} {obj_quat[2]:.6f} {obj_quat[3]:.6f}"
    inertia_str = f"{inertia[0]:.4f} {inertia[1]:.4f} {inertia[2]:.4f}"
    size_str = f"{half_ext[0]:.4f} {half_ext[1]:.4f} {half_ext[2]:.4f}"

    # Replace object body section
    obj_body_pattern = r'<body name="object".*?</body>\s*</worldbody>'
    obj_body_new = (
        f'<body name="object" pos="{pos_str}" quat="{quat_str}">\n'
        f'      <freejoint name="object_joint" />\n'
        f'      <inertial pos="0 0 0" mass="{config["mass"]}" diaginertia="{inertia_str}" />\n'
        f'      <geom name="object_visual" type="mesh" mesh="{config["mesh_name"]}" '
        f'material="{config["material_name"]}" group="2" contype="0" conaffinity="0" />\n'
        f'      <geom name="object_collision" type="box" size="{size_str}" '
        f'rgba="0.4 0.5 0.6 0.3" group="3" contype="1" conaffinity="1" '
        f'friction="1 0.005 0.0001" condim="3" />\n'
        f'      <site name="trace_object" size="0.02" rgba="0 0 1 1" />\n'
        f'    </body>\n'
        f'  </worldbody>'
    )
    template = re.sub(obj_body_pattern, obj_body_new, template, flags=re.DOTALL)

    return template


def generate_task_info(task: str) -> dict:
    """Generate task_info.json with contact site IDs (always hand sites: 11, 15)."""
    return {"contact_site_ids": [11, 15]}


def main():
    import json

    for task, config in CASES.items():
        print(f"\n{'='*60}")
        print(f"Generating: {task}")
        print(f"{'='*60}")

        # Generate scene.xml
        scene_xml = generate_scene_xml(task, config)
        out_dir = f"{PROCESSED}/{task}"
        os.makedirs(out_dir, exist_ok=True)

        scene_path = f"{out_dir}/scene.xml"
        with open(scene_path, "w") as f:
            f.write(scene_xml)
        print(f"  Wrote: {scene_path}")

        # Generate task_info.json
        task_info = generate_task_info(task)
        task_info_path = f"{out_dir}/task_info.json"
        with open(task_info_path, "w") as f:
            json.dump(task_info, f, indent=2)
        print(f"  Wrote: {task_info_path}")

        # Verify with MuJoCo
        import mujoco
        try:
            model = mujoco.MjModel.from_xml_path(scene_path)
            print(f"  MuJoCo load OK: nq={model.nq}, nv={model.nv}, nu={model.nu}")
            assert model.nq == 43, f"Expected nq=43, got {model.nq}"
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        # Get object properties for summary
        half_ext = get_obj_bbox(config["mesh_dir"], config["mesh_file"])
        obj_pos, obj_quat = get_initial_obj_state(config["source_npz"])
        data = np.load(config["source_npz"])
        obj_z = data["qpos"][:, 38]
        print(f"  Object bbox half-extents: {half_ext}")
        print(f"  Object init pos: {obj_pos}")
        print(f"  Object z range: {obj_z.min():.3f} → {obj_z.max():.3f} (lift={obj_z.max()-obj_z[0]:.3f}m)")


if __name__ == "__main__":
    main()
