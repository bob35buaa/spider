#!/usr/bin/env python3
"""Set collision box = mesh_AABB × margin for specified CORE4D cases.

Unlike fix_collision_boxes.py (which has 8% tolerance), this tool
forces the exact margin without any tolerance check.

Updates both scene.xml and scene_act.xml.

Usage:
    uv run workspace/core4d/scripts/convert/set_collision_margin.py \
        --cases box025_person1 bucket010_person1 desk005_person2 \
        --margin 0.95

    # Dry run to see what would change:
    uv run workspace/core4d/scripts/convert/set_collision_margin.py \
        --cases box025_person1 --margin 0.95 --dry-run
"""

import argparse
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

BASE = "example_datasets/processed/core4d/unitree_g1/humanoid_object"


def get_mesh_aabb(model: mujoco.MjModel) -> np.ndarray | None:
    """Get visual mesh AABB half-extents from compiled MuJoCo model."""
    for i in range(model.ngeom):
        if model.geom(i).name == "object_visual":
            return model.geom(i).size.copy()
    return None


def get_collision_size(model: mujoco.MjModel) -> np.ndarray | None:
    """Get current collision box half-extents."""
    for i in range(model.ngeom):
        if model.geom(i).name == "object_collision":
            return model.geom(i).size.copy()
    return None


def update_collision_in_xml(xml_path: str, new_size: np.ndarray) -> bool:
    """Update object_collision geom size in an XML file. Returns True if found."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    found = False
    for geom in root.iter("geom"):
        if geom.get("name") == "object_collision":
            size_str = f"{new_size[0]:.4f} {new_size[1]:.4f} {new_size[2]:.4f}"
            geom.set("size", size_str)
            found = True
            break
    if found:
        tree.write(xml_path, encoding="unicode")
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description="Set collision box margin for CORE4D cases")
    parser.add_argument("--cases", nargs="+", required=True, help="Case names (e.g. box025_person1)")
    parser.add_argument("--margin", type=float, required=True, help="Margin multiplier (e.g. 0.95, 1.00)")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without modifying files")
    args = parser.parse_args()

    print(f"Setting collision boxes: margin={args.margin}, dry_run={args.dry_run}")
    print(f"{'Case':<24} {'Mesh AABB (half)':<30} {'Old Collision':<30} {'New Collision':<30} {'Old Ratio':<16} {'Vol Change'}")
    print("-" * 155)

    for case in args.cases:
        scene_path = f"{BASE}/{case}/scene.xml"

        try:
            model = mujoco.MjModel.from_xml_path(scene_path)
        except Exception as e:
            print(f"{case:<24} ERROR loading scene.xml: {e}")
            continue

        mesh_aabb = get_mesh_aabb(model)
        old_col = get_collision_size(model)

        if mesh_aabb is None or old_col is None:
            print(f"{case:<24} SKIP: missing object_visual or object_collision geom")
            continue

        new_size = mesh_aabb * args.margin
        old_ratio = old_col / mesh_aabb
        vol_old = 8 * np.prod(old_col)
        vol_new = 8 * np.prod(new_size)
        vol_change = (vol_new / vol_old - 1) * 100

        m_str = " x ".join(f"{v:.4f}" for v in mesh_aabb)
        o_str = " x ".join(f"{v:.4f}" for v in old_col)
        n_str = " x ".join(f"{v:.4f}" for v in new_size)
        r_str = " x ".join(f"{v:.2f}" for v in old_ratio)

        prefix = "[DRY] " if args.dry_run else ""
        print(f"{case:<24} {m_str:<30} {o_str:<30} {n_str:<30} {r_str:<16} {prefix}{vol_change:+.1f}%")

        if args.dry_run:
            continue

        # Update scene.xml
        update_collision_in_xml(scene_path, new_size)
        # Validate
        mujoco.MjModel.from_xml_path(scene_path)

        # Update scene_act.xml if exists
        scene_act_path = f"{BASE}/{case}/scene_act.xml"
        try:
            if update_collision_in_xml(scene_act_path, new_size):
                mujoco.MjModel.from_xml_path(scene_act_path)
                print(f"  -> scene_act.xml also updated")
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"  -> WARNING: scene_act.xml update failed: {e}")

    print(f"\nDone. {'(dry run, no files changed)' if args.dry_run else 'Files updated.'}")


if __name__ == "__main__":
    main()
