#!/usr/bin/env python3
"""Verify a generated single-agent CORE4D SPIDER case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[3]
BASE = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"


def load_npz(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return dict(np.load(path, allow_pickle=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--source-scene", type=Path, required=True)
    parser.add_argument("--trimmed", type=Path, required=True)
    parser.add_argument("--data-id", default="0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    case_dir = BASE / args.task
    scene = case_dir / "scene.xml"
    scene_act = case_dir / "scene_act.xml"
    traj = case_dir / str(args.data_id) / "trajectory_kinematic.npz"

    trimmed = load_npz(args.trimmed)
    spider = load_npz(traj)
    model_scene = mujoco.MjModel.from_xml_path(str(scene))
    model_act = mujoco.MjModel.from_xml_path(str(scene_act))

    qpos = spider["qpos"]
    data = mujoco.MjData(model_scene)
    data.qpos[:] = qpos[0]
    mujoco.mj_forward(model_scene, data)

    trim_matches = bool(
        trimmed["qpos"].shape[0] == qpos.shape[0]
        and np.allclose(trimmed["qpos"], qpos)
    )

    summary = {
        "task": args.task,
        "case_dir": str(case_dir),
        "source_scene": str(args.source_scene),
        "trimmed": str(args.trimmed),
        "trajectory": str(traj),
        "trimmed_qpos_shape": list(trimmed["qpos"].shape),
        "spider_qpos_shape": list(spider["qpos"].shape),
        "spider_qvel_shape": list(spider["qvel"].shape),
        "spider_ctrl_shape": list(spider["ctrl"].shape),
        "spider_contact_shape": list(spider["contact"].shape),
        "trimmed_qpos_matches_spider_qpos": trim_matches,
        "scene": {
            "path": str(scene),
            "nq": int(model_scene.nq),
            "nv": int(model_scene.nv),
            "nu": int(model_scene.nu),
        },
        "scene_act": {
            "path": str(scene_act),
            "nq": int(model_act.nq),
            "nv": int(model_act.nv),
            "nu": int(model_act.nu),
        },
        "first_object_qpos": qpos[0, 36:43].tolist(),
        "first_robot_root_qpos": qpos[0, :7].tolist(),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if not trim_matches:
        raise RuntimeError("trimmed qpos does not match generated SPIDER qpos")
    if model_scene.nq != 43 or model_act.nq != 42:
        raise RuntimeError("Unexpected MuJoCo nq for scene or scene_act")


if __name__ == "__main__":
    main()
