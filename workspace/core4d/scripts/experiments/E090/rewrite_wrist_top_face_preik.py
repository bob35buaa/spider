#!/usr/bin/env python3
"""Rewrite converted OmniRetarget wrist targets onto the box world-up face.

Input is the converted CORE4D NPZ consumed by Holosoma robot_retarget.py:
`global_joint_positions` (T, 22, 3), `object_poses` (T, 7) in
[qw, qx, qy, qz, x, y, z]. The script only rewrites joints 20/21
(L_Wrist/R_Wrist) and preserves all other arrays and the original file name.
"""
from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

TOP_OFFSET_M = 0.05
EDGE_MARGIN_M = 0.02
WRIST_INDICES = (20, 21)


def quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = [float(x) for x in q]
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def parse_collision_half_extents(scene_xml: Path) -> np.ndarray:
    root = ET.parse(scene_xml).getroot()
    for body in root.iter("body"):
        if body.attrib.get("name") != "object":
            continue
        for geom in body.iter("geom"):
            if geom.attrib.get("type") == "box" and "collision" in geom.attrib.get("name", "").lower():
                return np.array([float(x) for x in geom.attrib["size"].split()], dtype=np.float64)
    raise RuntimeError(f"No object collision box found in {scene_xml}")


def project_to_world_up_face(point_w: np.ndarray, obj_pose: np.ndarray, half: np.ndarray) -> np.ndarray:
    quat = obj_pose[:4]
    obj_pos = obj_pose[4:7].astype(np.float64)
    rot = quat_wxyz_to_mat(quat)
    point_l = rot.T @ (point_w.astype(np.float64) - obj_pos)

    local_world_up = rot.T @ np.array([0.0, 0.0, 1.0])
    axis = int(np.argmax(np.abs(local_world_up)))
    sign = float(np.sign(local_world_up[axis])) or 1.0

    target_l = point_l.copy()
    for a in range(3):
        if a == axis:
            continue
        target_l[a] = float(np.clip(target_l[a], -half[a] + EDGE_MARGIN_M, half[a] - EDGE_MARGIN_M))
    target_l[axis] = sign * (half[axis] + TOP_OFFSET_M)
    return (rot @ target_l + obj_pos).astype(np.float32)


def rewrite_npz(input_npz: Path, output_npz: Path, scene_xml: Path) -> dict:
    half = parse_collision_half_extents(scene_xml)
    src = np.load(input_npz)
    arrays = {k: src[k] for k in src.files}
    joints = arrays["global_joint_positions"].copy()
    object_poses = arrays["object_poses"]
    if joints.ndim != 3 or joints.shape[1] <= max(WRIST_INDICES):
        raise ValueError(f"Unexpected global_joint_positions shape: {joints.shape}")
    if object_poses.shape[0] != joints.shape[0]:
        raise ValueError(f"T mismatch: joints={joints.shape[0]} object_poses={object_poses.shape[0]}")

    before = joints[:, WRIST_INDICES, :].copy()
    for t in range(joints.shape[0]):
        for idx in WRIST_INDICES:
            joints[t, idx] = project_to_world_up_face(joints[t, idx], object_poses[t], half)

    arrays["global_joint_positions"] = joints
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_npz, **arrays)

    delta = np.linalg.norm(joints[:, WRIST_INDICES, :] - before, axis=-1)
    return {
        "input_npz": str(input_npz),
        "output_npz": str(output_npz),
        "scene_xml": str(scene_xml),
        "T": int(joints.shape[0]),
        "half_extents": half.tolist(),
        "mean_delta_m": float(delta.mean()),
        "max_delta_m": float(delta.max()),
        "top_offset_m": TOP_OFFSET_M,
        "edge_margin_m": EDGE_MARGIN_M,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--scene-xml", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    args = parser.parse_args()

    input_npz = args.input_dir / f"{args.task_name}.npz"
    output_npz = args.output_dir / f"{args.task_name}.npz"
    if not input_npz.is_file():
        raise FileNotFoundError(input_npz)
    summary = rewrite_npz(input_npz, output_npz, args.scene_xml)

    # Preserve any sidecar files the converter may create in future.
    for sidecar in args.input_dir.iterdir():
        if sidecar.name == input_npz.name or not sidecar.is_file():
            continue
        dst = args.output_dir / sidecar.name
        if not dst.exists():
            shutil.copy2(sidecar, dst)

    summary_json = args.summary_json or args.output_dir / "topface_rewrite_summary.json"
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
