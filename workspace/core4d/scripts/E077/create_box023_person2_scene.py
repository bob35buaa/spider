#!/usr/bin/env python3
"""Create box023_person2 scene files from box023_person1 template."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[4]
BASE = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
DEFAULT_SOURCE_SCENE = BASE / "box023_person1/scene.xml"
DEFAULT_TASK = "box023_person2"
DEFAULT_QPOS = (
    REPO
    / "workspace/core4d/results/E077/holosoma_box023_person2/trimmed/"
    / "20231008-045-person2-Box023_with_obj_original.npz"
)


def fmt(values: np.ndarray, digits: int) -> str:
    return " ".join(f"{float(v):.{digits}f}" for v in values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-scene", type=Path, default=DEFAULT_SOURCE_SCENE)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--qpos", type=Path, default=DEFAULT_QPOS)
    parser.add_argument("--generate-scene-act", action="store_true")
    args = parser.parse_args()

    out_dir = BASE / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_path = out_dir / "scene.xml"
    shutil.copy2(args.source_scene, scene_path)

    qpos = np.load(args.qpos, allow_pickle=True)["qpos"]
    obj_pos = qpos[0, 36:39]
    obj_quat = qpos[0, 39:43]

    tree = ET.parse(scene_path)
    root = tree.getroot()
    object_body = None
    for body in root.iter("body"):
        if body.get("name") == "object":
            object_body = body
            break
    if object_body is None:
        raise ValueError(f"No object body found in {scene_path}")
    object_body.set("pos", fmt(obj_pos, 4))
    object_body.set("quat", fmt(obj_quat, 6))
    tree.write(scene_path, encoding="unicode")

    meta_path = out_dir / "task_info.json"
    with open(meta_path, "w") as f:
        json.dump(
            {
                "source_scene": str(args.source_scene),
                "source_qpos": str(args.qpos),
                "object_initial_pos": obj_pos.tolist(),
                "object_initial_quat": obj_quat.tolist(),
            },
            f,
            indent=2,
        )

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
