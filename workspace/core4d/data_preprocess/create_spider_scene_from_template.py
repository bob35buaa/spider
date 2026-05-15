#!/usr/bin/env python3
"""Create a SPIDER CORE4D scene by copying an existing scene template.

The script is intentionally small: it does not invent object geometry. It
copies a known-good SPIDER scene.xml, updates the object freejoint body's
initial pose from a Holosoma trimmed qpos, and optionally generates scene_act.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[3]
BASE = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"


def fmt(values: np.ndarray, digits: int) -> str:
    return " ".join(f"{float(v):.{digits}f}" for v in values)


def find_object_body(root: ET.Element, object_body_name: str) -> ET.Element:
    for body in root.iter("body"):
        if body.get("name") == object_body_name:
            return body
    raise ValueError(f"No object body named '{object_body_name}' found")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-scene", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--qpos", type=Path, required=True)
    parser.add_argument("--data-id", default="0")
    parser.add_argument("--object-body-name", default="object")
    parser.add_argument("--date", default=None)
    parser.add_argument("--seq", default=None)
    parser.add_argument("--person", default=None)
    parser.add_argument("--object-name", default=None)
    parser.add_argument("--object-model-rel", default=None)
    parser.add_argument("--generate-scene-act", action="store_true")
    args = parser.parse_args()

    if not args.source_scene.is_file():
        raise FileNotFoundError(args.source_scene)
    if not args.qpos.is_file():
        raise FileNotFoundError(args.qpos)

    out_dir = BASE / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_path = out_dir / "scene.xml"

    src_resolved = args.source_scene.resolve()
    dst_resolved = scene_path.resolve()
    if src_resolved != dst_resolved:
        shutil.copy2(args.source_scene, scene_path)
    elif not scene_path.is_file():
        raise FileNotFoundError(scene_path)

    qpos = np.load(args.qpos, allow_pickle=True)["qpos"]
    if qpos.ndim == 3:
        qpos = qpos[:, 0, :]
    if qpos.shape[1] < 43:
        raise ValueError(f"Expected qpos dim >=43, got {qpos.shape}")
    obj_pos = qpos[0, 36:39]
    obj_quat = qpos[0, 39:43]

    tree = ET.parse(scene_path)
    root = tree.getroot()
    object_body = find_object_body(root, args.object_body_name)
    object_body.set("pos", fmt(obj_pos, 4))
    object_body.set("quat", fmt(obj_quat, 6))
    tree.write(scene_path, encoding="unicode")

    meta = {
        "source_scene": str(args.source_scene),
        "source_qpos": str(args.qpos),
        "task": args.task,
        "data_id": str(args.data_id),
        "date": args.date,
        "seq": args.seq,
        "person": args.person,
        "object_name": args.object_name,
        "object_model_rel": args.object_model_rel,
        "object_body_name": args.object_body_name,
        "object_initial_pos": obj_pos.tolist(),
        "object_initial_quat": obj_quat.tolist(),
    }
    meta_path = out_dir / "task_info.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    print(f"Wrote {scene_path}")
    print(f"Wrote {meta_path}")
    print(f"MuJoCo scene OK: nq={model.nq}, nv={model.nv}, nu={model.nu}")

    if args.generate_scene_act:
        sys.path.insert(0, str(REPO / "workspace/core4d/scripts/convert"))
        from generate_scene_act import generate_scene_act

        scene_act_path, euler = generate_scene_act(args.task)
        print(f"Wrote {scene_act_path} euler={euler}")


if __name__ == "__main__":
    main()
