#!/usr/bin/env python3
"""Fix collision box sizes to match visual mesh AABB for all CORE4D cases.

Bug: scene.xml files were generated from templates (box025/bucket005/desk005),
so collision box sizes were copied verbatim instead of being set per-object.
This causes severe physics mismatch — e.g. box023's collision box is 1.8x its
visual mesh, making the robot hit invisible walls.

Fix: set collision box half-size = mesh AABB half-size × margin (default 1.05).

Usage:
    uv run workspace/core4d/scripts/convert/fix_collision_boxes.py [--margin 1.05] [--dry-run]
"""

import argparse
import os
import re
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

BASE = "example_datasets/processed/core4d/unitree_g1/humanoid_object"
MARGIN = 1.05  # 5% margin around mesh AABB


def get_mesh_aabb(model: mujoco.MjModel) -> np.ndarray | None:
    """Get visual mesh AABB half-extents from MuJoCo model."""
    for i in range(model.ngeom):
        if model.geom(i).name == "object_visual":
            return model.geom(i).size.copy()
    return None


def get_collision_size(model: mujoco.MjModel) -> tuple[np.ndarray | None, int]:
    """Get collision box half-extents and geom type."""
    for i in range(model.ngeom):
        if model.geom(i).name == "object_collision":
            return model.geom(i).size.copy(), int(model.geom(i).type[0])
    return None, -1


def fix_scene_xml(scene_path: str, margin: float, dry_run: bool) -> dict:
    """Fix collision box in scene.xml. Returns diagnostic info."""
    model = mujoco.MjModel.from_xml_path(scene_path)
    mesh_aabb = get_mesh_aabb(model)
    col_size, col_type = get_collision_size(model)

    if mesh_aabb is None or col_size is None:
        return {"status": "skip", "reason": "missing geom"}

    new_size = mesh_aabb * margin
    ratios = col_size / mesh_aabb
    max_deviation = float(np.max(np.abs(ratios - 1.0)))

    info = {
        "mesh_aabb": mesh_aabb.tolist(),
        "old_collision": col_size.tolist(),
        "new_collision": new_size.tolist(),
        "ratios": ratios.tolist(),
        "max_deviation": max_deviation,
    }

    if max_deviation < 0.08:  # within 8% — already close enough
        info["status"] = "ok"
        return info

    info["status"] = "fixed"

    if dry_run:
        return info

    # Parse XML and update collision size
    tree = ET.parse(scene_path)
    root = tree.getroot()

    for geom in root.iter("geom"):
        if geom.get("name") == "object_collision":
            size_str = f"{new_size[0]:.4f} {new_size[1]:.4f} {new_size[2]:.4f}"
            geom.set("size", size_str)
            break

    tree.write(scene_path, encoding="unicode")

    # Validate the fixed file loads
    mujoco.MjModel.from_xml_path(scene_path)

    return info


def main():
    parser = argparse.ArgumentParser(description="Fix collision boxes for CORE4D cases")
    parser.add_argument("--margin", type=float, default=MARGIN, help="Margin multiplier (default 1.05)")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without modifying files")
    parser.add_argument("--cases", nargs="*", help="Specific cases to fix (default: all)")
    args = parser.parse_args()

    cases = args.cases or sorted(os.listdir(BASE))

    print(f"Collision Box Fix (margin={args.margin}, dry_run={args.dry_run})")
    print(f"{'Case':<30} {'Mesh AABB (half)':<28} {'Old Collision':<28} {'New Collision':<28} {'Status'}")
    print("-" * 140)

    fixed_count = 0
    for case in cases:
        scene_path = os.path.join(BASE, case, "scene.xml")
        if not os.path.exists(scene_path):
            continue

        try:
            info = fix_scene_xml(scene_path, args.margin, args.dry_run)
        except Exception as e:
            import traceback
            print(f"{case:<30} ERROR: {e}")
            traceback.print_exc()
            continue

        if info["status"] == "skip":
            print(f"{case:<30} SKIPPED: {info.get('reason', 'unknown')}")
            continue

        mesh_str = " × ".join(f"{v:.3f}" for v in info["mesh_aabb"])
        old_str = " × ".join(f"{v:.3f}" for v in info["old_collision"])
        new_str = " × ".join(f"{v:.3f}" for v in info["new_collision"])

        if info["status"] == "ok":
            print(f"{case:<30} {mesh_str:<28} {old_str:<28} {'(no change needed)':<28} ✅")
        else:
            prefix = "[DRY] " if args.dry_run else ""
            print(f"{case:<30} {mesh_str:<28} {old_str:<28} {new_str:<28} {prefix}🔧 FIXED")
            fixed_count += 1

    action = "would fix" if args.dry_run else "fixed"
    print(f"\nDone: {action} {fixed_count} cases out of {len(cases)}")

    # Also fix scene_act.xml if it exists (same collision box)
    if not args.dry_run and fixed_count > 0:
        print("\nAlso updating scene_act.xml files...")
        for case in cases:
            scene_act_path = os.path.join(BASE, case, "scene_act.xml")
            if not os.path.exists(scene_act_path):
                continue
            scene_path = os.path.join(BASE, case, "scene.xml")
            # Read the fixed collision size from scene.xml
            model = mujoco.MjModel.from_xml_path(scene_path)
            _, _ = get_collision_size(model)
            mesh_aabb = get_mesh_aabb(model)
            if mesh_aabb is None:
                continue
            new_size = mesh_aabb * args.margin

            tree = ET.parse(scene_act_path)
            root = tree.getroot()
            for geom in root.iter("geom"):
                if geom.get("name") == "object_collision":
                    size_str = f"{new_size[0]:.4f} {new_size[1]:.4f} {new_size[2]:.4f}"
                    geom.set("size", size_str)
                    break
            tree.write(scene_act_path, encoding="unicode")
            # Validate
            mujoco.MjModel.from_xml_path(scene_act_path)
            print(f"  {case}/scene_act.xml updated ✅")


if __name__ == "__main__":
    main()
