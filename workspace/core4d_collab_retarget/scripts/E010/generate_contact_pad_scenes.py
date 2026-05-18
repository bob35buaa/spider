#!/usr/bin/env python3
"""Generate contact-pad scene XMLs for E010."""

from __future__ import annotations

import argparse
import csv
import xml.etree.ElementTree as ET
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
BASE = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
VARIANTS = REPO / "workspace/core4d_collab_retarget/scripts/E010/variants.tsv"

FIELDNAMES = [
    "variant",
    "source_task",
    "mask_slug",
    "person_idx",
    "queue",
    "role",
    "wave",
    "point_local_x",
    "point_local_y",
    "point_local_z",
    "pad_size",
    "support_proxy_max_xy_speed",
    "support_proxy_ref_dt",
    "hold_contact_rew_scale",
    "hold_contact_sigma",
    "hold_contact_start_eval_time",
    "hold_contact_end_eval_time",
    "hold_contact_require_ref_contact",
]


def read_variants(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(
            csv.DictReader(
                (line for line in f if line.strip() and not line.startswith("#")),
                delimiter="\t",
                fieldnames=FIELDNAMES,
            )
        )


def scene_name_for_pad_size(pad_size: str) -> str:
    return f"scene_contact_pad{int(round(float(pad_size) * 100)):02d}"


def _indent(elem: ET.Element, level: int = 0) -> None:
    spacer = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = spacer + "  "
        for child in elem:
            _indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = spacer
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = spacer


def generate_scene(task: str, pad_size: float) -> Path:
    task_dir = BASE / task
    src = task_dir / "scene.xml"
    if not src.is_file():
        raise FileNotFoundError(src)

    tree = ET.parse(src)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    contact = root.find("contact")
    if worldbody is None or contact is None:
        raise ValueError(f"{src} missing worldbody/contact")

    for old in list(worldbody.findall("body")):
        if old.get("name") == "support_proxy_pad":
            worldbody.remove(old)
    for old in list(contact.findall("pair")):
        if old.get("name") == "support_proxy_pad_object":
            contact.remove(old)

    pad = ET.Element(
        "body",
        {
            "name": "support_proxy_pad",
            "mocap": "true",
            "pos": "0 0 1",
        },
    )
    ET.SubElement(
        pad,
        "geom",
        {
            "name": "support_proxy_pad_geom",
            "type": "sphere",
            "size": f"{pad_size:.4g}",
            "rgba": "0.1 0.35 1 0.35",
            "group": "3",
            "contype": "2",
            "conaffinity": "1",
            "friction": "2 1 0.001",
            "condim": "4",
            "priority": "2",
        },
    )
    ET.SubElement(
        pad,
        "site",
        {
            "name": "trace_support_proxy_pad",
            "size": "0.025",
            "rgba": "0.1 0.35 1 1",
        },
    )
    worldbody.append(pad)

    ET.SubElement(
        contact,
        "pair",
        {
            "name": "support_proxy_pad_object",
            "geom1": "support_proxy_pad_geom",
            "geom2": "object_collision",
            "solref": "0.008 1",
            "friction": "2 1",
            "condim": "4",
        },
    )

    _indent(root)
    out = task_dir / f"{scene_name_for_pad_size(str(pad_size))}.xml"
    tree.write(out, encoding="unicode", xml_declaration=False)
    text = out.read_text(encoding="utf-8")
    if not text.endswith("\n"):
        out.write_text(text + "\n", encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", type=Path, default=VARIANTS)
    args = parser.parse_args()

    seen: set[tuple[str, float]] = set()
    for row in read_variants(args.variants):
        key = (row["source_task"], float(row["pad_size"]))
        if key in seen:
            continue
        seen.add(key)
        out = generate_scene(*key)
        print(f"Wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
