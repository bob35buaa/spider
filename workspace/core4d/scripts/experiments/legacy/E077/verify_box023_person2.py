#!/usr/bin/env python3
"""Verify the E077 box023_person2 data construction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[4]
BASE = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
RESULTS = REPO / "workspace/core4d/results/E077"
DEFAULT_OUT = RESULTS / "box023_person2_verify_summary.json"


def load_npz(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return dict(np.load(path, allow_pickle=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--trim-start", type=int, default=42)
    args = parser.parse_args()

    p1_traj_path = BASE / "box023_person1/0/trajectory_kinematic.npz"
    p2_dir = BASE / "box023_person2"
    p2_traj_path = p2_dir / "0/trajectory_kinematic.npz"
    p2_scene = p2_dir / "scene.xml"
    p2_scene_act = p2_dir / "scene_act.xml"
    p2_retarget_path = (
        RESULTS
        / "holosoma_box023_person2/retargeted/20231008-045-person2-Box023_with_obj_original.npz"
    )
    p2_trim_path = (
        RESULTS
        / "holosoma_box023_person2/trimmed/20231008-045-person2-Box023_with_obj_original.npz"
    )
    mask_path = RESULTS / "contact_masks/box023/raw_contact_mask_3cm.npz"
    p1_converted_path = (
        Path("/home/ubuntu/Workspace/holosoma/workspace/pipeline/results/e046_batch/converted")
        / "20231008-045-person1-Box023_with_obj.npz"
    )
    p2_converted_path = (
        RESULTS / "holosoma_box023_person2/converted/20231008-045-person2-Box023_with_obj.npz"
    )

    p1 = load_npz(p1_traj_path)
    p2 = load_npz(p2_traj_path)
    p2_retarget = load_npz(p2_retarget_path)
    p2_trim = load_npz(p2_trim_path)
    mask = load_npz(mask_path)
    p1_converted = load_npz(p1_converted_path)
    p2_converted = load_npz(p2_converted_path)

    start = args.trim_start
    end = start + p2_trim["qpos"].shape[0]
    trim_matches_retarget = bool(
        np.allclose(p2_trim["qpos"], p2_retarget["qpos"][start:end])
    )
    converted_object_match = bool(
        np.allclose(p1_converted["object_poses"], p2_converted["object_poses"])
    )
    object_qpos_diff = p2["qpos"][:, 36:43] - p1["qpos"][:, 36:43]
    p2_contact_all_one = bool(np.allclose(p2["contact"], 1.0))

    model_scene = mujoco.MjModel.from_xml_path(str(p2_scene))
    model_act = mujoco.MjModel.from_xml_path(str(p2_scene_act))
    data_scene = mujoco.MjData(model_scene)
    data_scene.qpos[:] = p2["qpos"][0]
    mujoco.mj_forward(model_scene, data_scene)

    summary = {
        "box023_person2_dir": str(p2_dir),
        "qpos_shape": list(p2["qpos"].shape),
        "qvel_shape": list(p2["qvel"].shape),
        "ctrl_shape": list(p2["ctrl"].shape),
        "contact_shape": list(p2["contact"].shape),
        "retarget_qpos_shape": list(p2_retarget["qpos"].shape),
        "trim_qpos_shape": list(p2_trim["qpos"].shape),
        "trim_matches_retarget_42_178": trim_matches_retarget,
        "converted_object_poses_match": converted_object_match,
        "person1_person2_object_qpos_match_after_retarget_scale": bool(
            np.allclose(p1["qpos"][:, 36:43], p2["qpos"][:, 36:43], atol=1e-8)
        ),
        "person1_person2_object_qpos_diff": {
            "max_abs": float(np.max(np.abs(object_qpos_diff))),
            "pos_mean": object_qpos_diff[:, :3].mean(axis=0).tolist(),
            "pos_min": object_qpos_diff[:, :3].min(axis=0).tolist(),
            "pos_max": object_qpos_diff[:, :3].max(axis=0).tolist(),
            "quat_max_abs": float(np.max(np.abs(object_qpos_diff[:, 3:]))),
        },
        "spider_contact_all_one": p2_contact_all_one,
        "scene": {
            "path": str(p2_scene),
            "nq": int(model_scene.nq),
            "nv": int(model_scene.nv),
            "nu": int(model_scene.nu),
        },
        "scene_act": {
            "path": str(p2_scene_act),
            "nq": int(model_act.nq),
            "nv": int(model_act.nv),
            "nu": int(model_act.nu),
        },
        "first_object_qpos": p2["qpos"][0, 36:43].tolist(),
        "first_robot_root_qpos": p2["qpos"][0, :7].tolist(),
        "contact_mask_3cm_shapes": {
            "raw": list(mask["raw_contact_mask_3cm"].shape),
            "spider": list(mask["spider_contact_mask_3cm"].shape),
            "eval": list(mask["eval_contact_mask_3cm"].shape),
        },
        "note": (
            "SPIDER contact is still converter output. Converted CORE4D object "
            "poses match exactly across person1/person2, but Holosoma retarget "
            "preprocess scales object motion by each person's smpl_scale, so "
            "retargeted qpos object trajectories are not directly mergeable "
            "without a common-scale alignment."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    if not trim_matches_retarget:
        raise RuntimeError("trimmed qpos does not match retargeted[42:178]")
    if not converted_object_match:
        raise RuntimeError("converted CORE4D object poses do not match")
    if model_scene.nq != 43 or model_act.nq != 42:
        raise RuntimeError("Unexpected MuJoCo nq for scene or scene_act")


if __name__ == "__main__":
    main()
