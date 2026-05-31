#!/usr/bin/env python3
"""Render E103 MuJoCo audit visuals.

The visuals are evidence, not a physics validation: inertial pollution is
numeric, so each rendered frame is paired with audit metrics that identify the
corrupted robot mass/inertia and object geometry policy.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np


DEFAULT_ROOT = Path("example_datasets/processed/core4d/unitree_g1/humanoid_object")
DEFAULT_CASES = [
    "box023_person1",
    "box004_person1",
    "box021_person1",
    "box026_person2",
    "d003_box021_20231018_029_p2",
    "e091_box026_20231018_039_p2",
]


def read_tsv(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        return {row["task"]: row for row in csv.DictReader(f, delimiter="\t")}


def render_camera(renderer: mujoco.Renderer, data: mujoco.MjData, camera: mujoco.MjvCamera) -> np.ndarray:
    renderer.update_scene(data, camera=camera)
    return renderer.render()


def make_camera(model: mujoco.MjModel, data: mujoco.MjData, azimuth: float, elevation: float = -18.0) -> mujoco.MjvCamera:
    mujoco.mj_forward(model, data)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = azimuth
    cam.elevation = elevation
    cam.distance = 3.0
    cam.lookat[:] = np.array([0.0, 0.0, 0.55])
    obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    if obj_id >= 0:
        cam.lookat[:] = data.xpos[obj_id]
        cam.lookat[2] = max(float(cam.lookat[2]), 0.45)
    return cam


def render_views(scene: Path, out_prefix: Path) -> tuple[list[Path], Path]:
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    view_paths: list[Path] = []
    views = [
        ("front", 90.0),
        ("left", 180.0),
        ("back", 270.0),
        ("iso", 45.0),
    ]
    with mujoco.Renderer(model, height=720, width=960) as renderer:
        for name, azimuth in views:
            cam = make_camera(model, data, azimuth=azimuth)
            frame = render_camera(renderer, data, cam)
            path = out_prefix.parent / f"{out_prefix.name}_{name}.png"
            imageio.imwrite(path, frame)
            view_paths.append(path)

        frames = []
        for azimuth in np.linspace(0, 360, 72, endpoint=False):
            cam = make_camera(model, data, azimuth=float(azimuth))
            frames.append(render_camera(renderer, data, cam))
    video_path = out_prefix.parent / f"{out_prefix.name}_turntable.mp4"
    imageio.mimsave(video_path, frames, fps=24)
    return view_paths, video_path


def write_metric_sheet(
    task: str,
    inertial: dict[str, str],
    geometry: dict[str, str],
    view_paths: list[Path],
    video_path: Path,
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.1])

    for idx, path in enumerate(view_paths[:4]):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        ax.imshow(imageio.imread(path))
        ax.set_title(path.stem.rsplit("_", 1)[-1])
        ax.axis("off")

    ax = fig.add_subplot(gs[:, 2])
    ax.axis("off")
    lines = [
        f"task: {task}",
        "",
        "Inertial audit",
        f"status: {inertial.get('status', '')}",
        f"robot unique pairs: {inertial.get('robot_inertial_unique_pairs', '')}",
        f"pelvis mass: {inertial.get('pelvis_mass', '')}",
        f"hip mass: {inertial.get('left_hip_pitch_mass', '')}",
        f"object mass: {inertial.get('object_mass', '')}",
        "",
        "Geometry audit",
        f"status: {geometry.get('status', '')}",
        f"mesh extents: {geometry.get('mesh_extents_m', '')}",
        f"expected half: {geometry.get('expected_collision_half_extents_m', '')}",
        f"collision half: {geometry.get('object_collision_half_extents_m', '')}",
        f"max rel err: {geometry.get('collision_max_rel_error', '')}",
        "",
        f"video: {video_path.name}",
    ]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=11, family="monospace")
    fig.suptitle(f"E103 scene audit: {task}", fontsize=16)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_review(out_dir: Path, rows: list[dict[str, str]]) -> None:
    lines = [
        "# E103 MuJoCo Scene Audit Visuals",
        "",
        "These renderings pair MuJoCo 3D views with inertial/geometry audit metrics.",
        "The visual mesh can look normal while inertial data is corrupted; the metric panel is part of the evidence.",
        "",
        "| task | registry_action | inertial | geometry | sheet | video |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['task']}` | `{row['action']}` | `{row['inertial_status']}` | `{row['geometry_status']}` | "
            f"[sheet]({Path(row['sheet']).name}) | [video]({Path(row['video']).name}) |"
        )
    (out_dir / "REVIEW.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--inertial", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cases", nargs="*", default=DEFAULT_CASES)
    args = parser.parse_args()

    inertial = read_tsv(args.inertial)
    geometry = read_tsv(args.geometry)
    registry = read_tsv(args.registry)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    review_rows: list[dict[str, str]] = []
    for task in args.cases:
        scene = args.root / task / "scene.xml"
        if not scene.is_file():
            print(f"skip missing scene {task}: {scene}")
            continue
        prefix = args.out_dir / task
        view_paths, video_path = render_views(scene, prefix)
        sheet_path = args.out_dir / f"{task}_audit_sheet.png"
        write_metric_sheet(task, inertial.get(task, {}), geometry.get(task, {}), view_paths, video_path, sheet_path)
        reg = registry.get(task, {})
        review_rows.append(
            {
                "task": task,
                "action": reg.get("action", ""),
                "inertial_status": inertial.get(task, {}).get("status", ""),
                "geometry_status": geometry.get(task, {}).get("status", ""),
                "sheet": str(sheet_path),
                "video": str(video_path),
            }
        )
        print(f"rendered {task}: {sheet_path} {video_path}")

    write_review(args.out_dir, review_rows)
    print(f"wrote {args.out_dir / 'REVIEW.md'}")


if __name__ == "__main__":
    main()
