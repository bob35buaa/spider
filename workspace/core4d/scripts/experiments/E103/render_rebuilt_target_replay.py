#!/usr/bin/env python3
"""Render MuJoCo kinematic replay videos for E103 rebuilt target cases."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np


SCENE_ROOT = Path("example_datasets/processed/core4d/unitree_g1/humanoid_object")


def read_registry(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return {row["task"]: row for row in csv.DictReader(f, delimiter="\t")}


def make_camera(model: mujoco.MjModel, data: mujoco.MjData) -> mujoco.MjvCamera:
    mujoco.mj_forward(model, data)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = 55
    cam.elevation = -18
    cam.distance = 3.0
    obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    look = np.array([0.0, 0.0, 0.55], dtype=float)
    if obj_id >= 0 and pelvis_id >= 0:
        look = 0.5 * data.xpos[obj_id] + 0.5 * data.xpos[pelvis_id]
        look[2] = max(float(look[2]), 0.55)
    cam.lookat[:] = look
    return cam


def render_frame(renderer: mujoco.Renderer, model: mujoco.MjModel, data: mujoco.MjData, qpos: np.ndarray) -> np.ndarray:
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    cam = make_camera(model, data)
    renderer.update_scene(data, camera=cam)
    return renderer.render()


def render_case(task: str, registry: dict[str, dict[str, str]], out_dir: Path) -> dict[str, str]:
    case_dir = SCENE_ROOT / task
    scene = case_dir / "scene.xml"
    traj = case_dir / "0" / "trajectory_kinematic.npz"
    verify = Path("workspace/core4d/results/E103/verify") / f"{task}_verify_summary.json"
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    qpos = np.load(traj, allow_pickle=True)["qpos"]

    out_dir.mkdir(parents=True, exist_ok=True)
    key_indices = sorted(set([0, max(0, len(qpos) // 3), max(0, 2 * len(qpos) // 3), len(qpos) - 1]))
    key_paths: list[Path] = []
    frames = []
    with mujoco.Renderer(model, height=720, width=960) as renderer:
        for i, q in enumerate(qpos):
            frame = render_frame(renderer, model, data, q)
            frames.append(frame)
            if i in key_indices:
                key_path = out_dir / f"{task}_f{i:04d}.png"
                imageio.imwrite(key_path, frame)
                key_paths.append(key_path)

    video = out_dir / f"{task}_kinematic_replay.mp4"
    imageio.mimsave(video, frames, fps=24)

    sheet = out_dir / f"{task}_replay_sheet.png"
    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.1])
    for idx, path in enumerate(key_paths[:4]):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        ax.imshow(imageio.imread(path))
        ax.set_title(path.stem.rsplit("_", 1)[-1])
        ax.axis("off")
    ax = fig.add_subplot(gs[:, 2])
    ax.axis("off")
    reg = registry.get(task, {})
    verify_data = json.loads(verify.read_text(encoding="utf-8")) if verify.is_file() else {}
    lines = [
        f"task: {task}",
        f"frames: {len(qpos)}",
        f"video: {video.name}",
        "",
        "Scene audit",
        f"action: {reg.get('action', '')}",
        f"inertial: {reg.get('inertial_status', '')}",
        f"geometry: {reg.get('geometry_status', '')}",
        f"robot unique pairs: {reg.get('robot_inertial_unique_pairs', '')}",
        f"pelvis mass: {reg.get('pelvis_mass', '')}",
        f"object mass: {reg.get('object_mass', '')}",
        "",
        "Verify",
        f"trimmed match: {verify_data.get('trimmed_qpos_matches_spider_qpos', '')}",
        f"scene nq/nv/nu: {verify_data.get('scene', {}).get('nq', '')}/"
        f"{verify_data.get('scene', {}).get('nv', '')}/{verify_data.get('scene', {}).get('nu', '')}",
        f"scene_act nq/nv/nu: {verify_data.get('scene_act', {}).get('nq', '')}/"
        f"{verify_data.get('scene_act', {}).get('nv', '')}/{verify_data.get('scene_act', {}).get('nu', '')}",
    ]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=11, family="monospace")
    fig.suptitle(f"E103 rebuilt target kinematic replay: {task}", fontsize=15)
    fig.savefig(sheet, dpi=150)
    plt.close(fig)

    return {
        "task": task,
        "frames": str(len(qpos)),
        "registry_action": reg.get("action", ""),
        "inertial_status": reg.get("inertial_status", ""),
        "geometry_status": reg.get("geometry_status", ""),
        "trimmed_match": str(verify_data.get("trimmed_qpos_matches_spider_qpos", "")),
        "sheet": str(sheet),
        "video": str(video),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    args = parser.parse_args()

    registry = read_registry(args.registry)
    rows = [render_case(task, registry, args.out_dir) for task in args.tasks]

    lines = [
        "# E103 Rebuilt Target Kinematic Replay Review",
        "",
        "MuJoCo kinematic replay of selected targets regenerated from clean source templates.",
        "These videos are data validation artifacts, not CEM/dynamics success labels.",
        "",
        "| task | frames | action | inertial | geometry | verify qpos | sheet | video |",
        "|---|---:|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['task']}` | {row['frames']} | `{row['registry_action']}` | `{row['inertial_status']}` | "
            f"`{row['geometry_status']}` | `{row['trimmed_match']}` | [sheet]({Path(row['sheet']).name}) | [video]({Path(row['video']).name}) |"
        )
    (args.out_dir / "REVIEW.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"review -> {args.out_dir / 'REVIEW.md'}")


if __name__ == "__main__":
    main()
