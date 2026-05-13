#!/usr/bin/env python3
"""Patch a CORE4D scene.xml from sphere hand_collision to 3-box per side.

Background: spider/assets/robots/unitree_g1/robot.xml has hand_collision as a
single 5cm sphere. The HDMI suitcase template uses 3 boxes per hand
(wrist cuff + palm slab + angled finger pad). E051 identified the sphere as
a root cause of physics issues but the fix was never propagated to G1.

Since each scene.xml is a self-contained MJCF (the hand_collision default
is INLINED, not pulled from robot.xml at load time), this script patches
each scene.xml directly.

What it does:
  1. Removes <default class="hand_collision"> sphere
  2. Replaces <geom name="lh"  class="hand_collision" /> with 3 inline boxes (lh, lh2, lh3)
  3. Replaces <geom name="rh"  class="hand_collision" /> with 3 inline boxes (rh, rh2, rh3)
  4. Expands hand collision pairs from 4 (lh+rh × floor+object) to 12 (3 boxes × 4 pairs)

Usage:
    .venv/bin/python workspace/core4d/scripts/convert/patch_hand_3box.py \
        --cases box023_person1 bucket005_s2_person1 ...

    # Dry run:
    .venv/bin/python workspace/core4d/scripts/convert/patch_hand_3box.py \
        --cases box023_person1 --dry-run
"""

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

BASE = "example_datasets/processed/core4d/unitree_g1/humanoid_object"

# 3-box geometry per side (in wrist_yaw_link frame, MuJoCo half-extents)
# Source: example_datasets/processed/hdmi/.../move_suitcase/scene/mjlab scene.xml
LEFT_BOXES = [
    # (name, size, pos, quat or None)
    ("lh",  "0.05 0.025 0.025", "0.02 0 0",     None),
    ("lh2", "0.05 0.01 0.05",   "0.09 0 0",     None),
    ("lh3", "0.025 0.01 0.05",  "0.15 -0.01 0", "0.980067 0 0 -0.198669"),
]
RIGHT_BOXES = [
    ("rh",  "0.05 0.025 0.025", "0.02 0 0",     None),
    ("rh2", "0.05 0.01 0.05",   "0.09 0 0",     None),
    ("rh3", "0.025 0.01 0.05",  "0.15 0.01 0",  "0.980067 0 0 0.198669"),
]

LEFT_BOX_NAMES = [b[0] for b in LEFT_BOXES]   # ["lh", "lh2", "lh3"]
RIGHT_BOX_NAMES = [b[0] for b in RIGHT_BOXES]  # ["rh", "rh2", "rh3"]


def fmt_geom(name: str, size: str, pos: str, quat: str | None) -> str:
    """Format a single inline <geom> element as text."""
    quat_attr = f' quat="{quat}"' if quat else ""
    return (
        f'<geom name="{name}" type="box" size="{size}" pos="{pos}"{quat_attr} '
        f'condim="1" />'
    )


def patch_scene_xml(xml_path: Path, dry_run: bool = False) -> dict:
    """Patch a single scene.xml. Returns a stats dict."""
    text = xml_path.read_text()
    stats = {
        "default_removed": False,
        "lh_geom_replaced": False,
        "rh_geom_replaced": False,
        "old_pair_count": 0,
        "new_pair_count": 0,
    }

    # 1. Remove <default class="hand_collision"> ... </default>
    #    (single-line in current files)
    pattern_default = re.compile(
        r'\s*<default class="hand_collision">\s*<geom[^/]+/>\s*</default>\s*\n?',
        re.MULTILINE,
    )
    new_text, n = pattern_default.subn("\n", text)
    if n > 0:
        stats["default_removed"] = True
        text = new_text

    # 2. Replace <geom name="lh" class="hand_collision" /> with 3 boxes
    pattern_lh = re.compile(r'<geom name="lh" class="hand_collision"\s*/>')
    if pattern_lh.search(text):
        replacement = "\n                          ".join(
            fmt_geom(*b) for b in LEFT_BOXES
        )
        text = pattern_lh.sub(replacement, text, count=1)
        stats["lh_geom_replaced"] = True

    # 3. Replace <geom name="rh" class="hand_collision" /> with 3 boxes
    pattern_rh = re.compile(r'<geom name="rh" class="hand_collision"\s*/>')
    if pattern_rh.search(text):
        replacement = "\n                          ".join(
            fmt_geom(*b) for b in RIGHT_BOXES
        )
        text = pattern_rh.sub(replacement, text, count=1)
        stats["rh_geom_replaced"] = True

    # 4. Expand hand <pair>s. Match the existing 4 pairs and replace with 12.
    #    Existing patterns (verified in box023 scene.xml lines 387-388, 402-403):
    #      <pair name="left_hand_floor"   geom1="lh" geom2="floor" .../>
    #      <pair name="right_hand_floor"  geom1="rh" geom2="floor" .../>
    #      <pair name="left_hand_object"  geom1="lh" geom2="object_collision" .../>
    #      <pair name="right_hand_object" geom1="rh" geom2="object_collision" .../>

    # Use ET parse for the <pair> section to be robust
    tree = ET.parse(xml_path)
    root = tree.getroot()
    contact_elem = root.find("contact")
    if contact_elem is None:
        raise RuntimeError(f"{xml_path}: no <contact> section")

    # Collect existing hand pairs to extract solref/friction/condim attributes
    floor_attrs = None
    object_attrs = None
    pairs_to_remove = []
    for pair in contact_elem.findall("pair"):
        name = pair.get("name", "")
        if name in ("left_hand_floor", "right_hand_floor"):
            if floor_attrs is None:
                floor_attrs = {k: v for k, v in pair.attrib.items()
                               if k not in ("name", "geom1", "geom2")}
            pairs_to_remove.append(pair)
        elif name in ("left_hand_object", "right_hand_object"):
            if object_attrs is None:
                object_attrs = {k: v for k, v in pair.attrib.items()
                                if k not in ("name", "geom1", "geom2")}
            pairs_to_remove.append(pair)
    stats["old_pair_count"] = len(pairs_to_remove)

    if floor_attrs is None or object_attrs is None:
        # Pairs may have been already patched
        stats["new_pair_count"] = len(
            [p for p in contact_elem.findall("pair")
             if p.get("name", "").endswith(("_floor", "_object"))
             and p.get("geom1", "") in LEFT_BOX_NAMES + RIGHT_BOX_NAMES]
        )
        if dry_run:
            return stats
        if stats["lh_geom_replaced"] or stats["rh_geom_replaced"] or stats["default_removed"]:
            xml_path.write_text(text)
            # Re-validate by reload (now that geoms changed but pairs may be stale)
            mujoco.MjModel.from_xml_path(str(xml_path))
        return stats

    # We have the existing attribute templates; build 12 new pairs
    new_pairs = []
    for box_name in LEFT_BOX_NAMES + RIGHT_BOX_NAMES:
        side = "left" if box_name.startswith("l") else "right"
        # vs floor
        attrs = {"name": f"{box_name}_floor", "geom1": box_name, "geom2": "floor", **floor_attrs}
        new_pairs.append(("floor", attrs))
        # vs object_collision
        attrs = {"name": f"{box_name}_object", "geom1": box_name, "geom2": "object_collision", **object_attrs}
        new_pairs.append(("object", attrs))
    stats["new_pair_count"] = len(new_pairs)

    if dry_run:
        return stats

    # Write the geom changes (default + lh + rh) first
    if stats["lh_geom_replaced"] or stats["rh_geom_replaced"] or stats["default_removed"]:
        xml_path.write_text(text)

    # Now reparse, edit pairs, and write again
    tree = ET.parse(xml_path)
    root = tree.getroot()
    contact_elem = root.find("contact")
    # Remove old hand pairs
    for pair in list(contact_elem.findall("pair")):
        n = pair.get("name", "")
        if n in ("left_hand_floor", "right_hand_floor",
                 "left_hand_object", "right_hand_object"):
            contact_elem.remove(pair)
    # Append new hand pairs at end of contact section
    for _kind, attrs in new_pairs:
        ET.SubElement(contact_elem, "pair", attrs)

    tree.write(str(xml_path), encoding="unicode")
    # Validate
    mujoco.MjModel.from_xml_path(str(xml_path))

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch CORE4D scene.xml from sphere to 3-box hand")
    parser.add_argument("--cases", nargs="+", required=True, help="Case names")
    parser.add_argument("--include-scene-act", action="store_true", default=True,
                        help="Also patch scene_act.xml (default: yes)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change")
    args = parser.parse_args()

    print(f"Patching hand collision: sphere -> 3 box per side, dry_run={args.dry_run}")
    print(f"{'Case':<28} {'File':<20} {'def_rm':<8} {'lh_rep':<8} {'rh_rep':<8} {'old_pairs':<10} {'new_pairs':<10} {'check':<10}")
    print("-" * 110)

    for case in args.cases:
        for fname in (["scene.xml", "scene_act.xml"] if args.include_scene_act else ["scene.xml"]):
            xml_path = Path(BASE) / case / fname
            if not xml_path.exists():
                print(f"{case:<28} {fname:<20} MISSING")
                continue
            try:
                stats = patch_scene_xml(xml_path, dry_run=args.dry_run)
                # Verify load (already done in patch unless dry run)
                check = "ok"
                if not args.dry_run:
                    try:
                        mujoco.MjModel.from_xml_path(str(xml_path))
                    except Exception as e:
                        check = f"FAIL: {e}"
                print(f"{case:<28} {fname:<20} "
                      f"{str(stats['default_removed']):<8} "
                      f"{str(stats['lh_geom_replaced']):<8} "
                      f"{str(stats['rh_geom_replaced']):<8} "
                      f"{stats['old_pair_count']:<10} "
                      f"{stats['new_pair_count']:<10} "
                      f"{check:<10}")
            except Exception as e:
                print(f"{case:<28} {fname:<20} ERROR: {e}")

    print(f"\nDone. {'(dry run)' if args.dry_run else ''}")


if __name__ == "__main__":
    main()
