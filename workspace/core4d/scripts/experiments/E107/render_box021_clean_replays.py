#!/usr/bin/env python3
"""Render MuJoCo kinematic replay sheets/videos for E107 Box021 clean targets."""

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


REPO = Path(__file__).resolve().parents[4]
TASK_ROOT = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
SUMMARY_TSV = REPO / "workspace/core4d/results/E107/box021_clean_gate_summary.tsv"
OUT_DIR = REPO / "workspace/core4d/results/E107/visuals/box021_clean_replay"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def make_camera(model: mujoco.MjModel, data: mujoco.MjData) -> mujoco.MjvCamera:
    mujoco.mj_forward(model, data)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = 55
    cam.elevation = -18
    cam.distance = 3.0
    obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    look = np.array([0.0, 0.0, 0.65], dtype=float)
    if obj_id >= 0 and pelvis_id >= 0:
        look = 0.55 * data.xpos[obj_id] + 0.45 * data.xpos[pelvis_id]
        look[2] = max(float(look[2]), 0.55)
    cam.lookat[:] = look
    return cam


def render_frame(renderer: mujoco.Renderer, model: mujoco.MjModel, data: mujoco.MjData, qpos: np.ndarray) -> np.ndarray:
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=make_camera(model, data))
    return renderer.render()


def render_case(row: dict[str, str], out_dir: Path, fps: int, video_stride: int) -> dict[str, str]:
    task = row["derived_task"]
    task_dir = TASK_ROOT / task
    scene = task_dir / "scene.xml"
    traj = task_dir / "0/trajectory_kinematic.npz"
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    qpos = np.load(traj, allow_pickle=True)["qpos"]

    key_indices = sorted(set([0, max(0, len(qpos) // 3), max(0, 2 * len(qpos) // 3), len(qpos) - 1]))
    key_paths: list[Path] = []
    frames = []
    with mujoco.Renderer(model, height=720, width=960) as renderer:
        for i, q in enumerate(qpos):
            frame = render_frame(renderer, model, data, q)
            if i in key_indices:
                key_path = out_dir / f"{task}_f{i:04d}.png"
                imageio.imwrite(key_path, frame)
                key_paths.append(key_path)
            if i % video_stride == 0 or i == len(qpos) - 1:
                frames.append(frame)

    video = out_dir / f"{task}_kinematic_replay.mp4"
    imageio.mimsave(video, frames, fps=fps)

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
    lines = [
        f"target: {row['target_task']}",
        f"derived: {row['derived_task']}",
        f"frames: {len(qpos)}",
        f"gate: {row['failure_mode']}",
        "",
        "Data checks",
        f"source clean: {row['source_scene_clean']}",
        f"rebuild: {row['reconstruction_ok']}",
        f"qpos match: {row['qpos_matches_legacy_spider']}",
        f"scene: {row['scene_dims']}",
        f"scene_act: {row['scene_act_dims']}",
        f"robot polluted: {row['scene_robot_polluted_mass_29_632']}/{row['scene_act_robot_polluted_mass_29_632']}",
        "",
        "Raw-contact gate",
        f"3cm: {row['raw_decision_3cm']} both={row['target_both_active_frac_3cm']}",
        f"5cm: {row['raw_decision_5cm']} both={row['target_both_active_frac_5cm']}",
    ]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=10.5, family="monospace")
    fig.suptitle(f"E107 Box021 clean target replay: {task}", fontsize=15)
    fig.savefig(sheet, dpi=150)
    plt.close(fig)
    return {
        "target_task": row["target_task"],
        "derived_task": task,
        "frames": str(len(qpos)),
        "gate": row["failure_mode"],
        "sheet": str(sheet),
        "video": str(video),
    }


def write_review(rows: list[dict[str, str]], out_dir: Path) -> None:
    lines = [
        "# E107 Box021 Clean Replay Review",
        "",
        "MuJoCo kinematic replay of Box021 clean targets rebuilt from E103 source templates.",
        "These videos validate data reconstruction only; they are not CEM success labels.",
        "",
        "| target | derived | frames | gate | sheet | video |",
        "|---|---|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['target_task']}` | `{row['derived_task']}` | {row['frames']} | `{row['gate']}` | "
            f"[sheet]({Path(row['sheet']).name}) | [video]({Path(row['video']).name}) |"
        )
    (out_dir / "REVIEW.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "render_manifest.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=SUMMARY_TSV)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--video-stride", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [row for row in read_rows(args.summary) if row.get("reconstruction_ok") == "True"]
    if args.limit > 0:
        rows = rows[: args.limit]
    rendered = [render_case(row, args.out_dir, args.fps, args.video_stride) for row in rows]
    write_review(rendered, args.out_dir)
    print(f"rendered={len(rendered)} review={args.out_dir / 'REVIEW.md'}")


if __name__ == "__main__":
    main()
