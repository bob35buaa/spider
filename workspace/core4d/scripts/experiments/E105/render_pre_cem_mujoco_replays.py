#!/usr/bin/env python3
"""Render E105 pre-CEM MuJoCo kinematic replays and target sheets."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))

from e105_common import REPO, RESULTS_ROOT, TASK_ROOT, rel, read_variants  # noqa: E402


OUT_ROOT = RESULTS_ROOT / "pre_cem_visuals"
GATE_TSV = RESULTS_ROOT / "pre_cem_visual_gate.tsv"


def make_camera(model: mujoco.MjModel, data: mujoco.MjData) -> mujoco.MjvCamera:
    mujoco.mj_forward(model, data)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = 55
    cam.elevation = -18
    cam.distance = 3.0
    obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    look = np.array([0.0, 0.0, 0.6], dtype=float)
    if obj_id >= 0 and pelvis_id >= 0:
        look = 0.5 * data.xpos[obj_id] + 0.5 * data.xpos[pelvis_id]
        look[2] = max(float(look[2]), 0.55)
    cam.lookat[:] = look
    return cam


def render_kinematic(row: dict[str, str], out_dir: Path, overwrite: bool) -> tuple[Path, Path, int]:
    task_dir = TASK_ROOT / row["derived_task"]
    scene = task_dir / "scene.xml"
    traj = task_dir / "0/trajectory_kinematic.npz"
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    qpos = np.load(traj, allow_pickle=True)["qpos"]
    out_dir.mkdir(parents=True, exist_ok=True)
    video = out_dir / f"{row['variant']}_kinematic_replay.mp4"
    sheet = out_dir / f"{row['variant']}_kinematic_sheet.png"
    if video.is_file() and sheet.is_file() and not overwrite:
        return video, sheet, int(qpos.shape[0])

    key_ids = sorted(set(np.linspace(0, qpos.shape[0] - 1, num=min(8, qpos.shape[0]), dtype=int).tolist()))
    key_images: list[tuple[int, np.ndarray]] = []
    frames = []
    with mujoco.Renderer(model, height=720, width=960) as renderer:
        for i, q in enumerate(qpos):
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=make_camera(model, data))
            frame = renderer.render()
            frames.append(frame)
            if i in key_ids:
                key_images.append((i, frame))
    imageio.mimsave(video, frames, fps=24)

    cols = 4
    rows_n = int(np.ceil(len(key_images) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(16, 4 * rows_n), squeeze=False, constrained_layout=True)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (idx, image) in zip(axes.ravel(), key_images):
        ax.imshow(image)
        ax.set_title(f"f{idx:04d}")
    fig.suptitle(f"{row['variant']} | {row['derived_task']} | clean kinematic replay")
    fig.savefig(sheet, dpi=150)
    plt.close(fig)
    return video, sheet, int(qpos.shape[0])


def target_sheet(row: dict[str, str], out_dir: Path, overwrite: bool) -> Path:
    out = out_dir / f"{row['variant']}_target_sheet.png"
    if out.is_file() and not overwrite:
        return out
    scene = TASK_ROOT / row["derived_task"] / "scene.xml"
    model = mujoco.MjModel.from_xml_path(str(scene))
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_collision")
    half = model.geom_size[gid, :3]

    target_path = REPO / row["target_npz"] if row["target_npz"] else None
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    panels = [("x", "y", 0, 1), ("x", "z", 0, 2), ("y", "z", 1, 2)]
    colors = ["#cc3311", "#0077bb"]
    title_extra = "ref_fk: no external target NPZ"
    if target_path and target_path.is_file():
        data = np.load(target_path, allow_pickle=True)
        key = "spider_contact_target_object_local"
        target = data[key].astype(float)
        title_extra = f"{row['target_route']} target | shape={target.shape}"
        for ax, (xlab, ylab, ix, iy) in zip(axes, panels):
            ax.add_patch(
                plt.Rectangle(
                    (-half[ix], -half[iy]),
                    2 * half[ix],
                    2 * half[iy],
                    fill=False,
                    edgecolor="black",
                    linewidth=1.2,
                )
            )
            for hand in (0, 1):
                ax.scatter(target[:, hand, ix], target[:, hand, iy], s=11, alpha=0.45, c=colors[hand], label=("left" if hand == 0 else "right"))
            ax.set_xlabel(f"local {xlab} (m)")
            ax.set_ylabel(f"local {ylab} (m)")
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
    else:
        for ax, (xlab, ylab, ix, iy) in zip(axes, panels):
            ax.add_patch(
                plt.Rectangle(
                    (-half[ix], -half[iy]),
                    2 * half[ix],
                    2 * half[iy],
                    fill=False,
                    edgecolor="black",
                    linewidth=1.2,
                )
            )
            ax.set_xlabel(f"local {xlab} (m)")
            ax.set_ylabel(f"local {ylab} (m)")
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.25)
            ax.text(0.5, 0.5, "ref_fk route\nno external target", ha="center", va="center", transform=ax.transAxes)
    fig.suptitle(f"{row['variant']} | {title_extra}")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def render_one(row: dict[str, str], overwrite: bool) -> dict[str, Any]:
    out_dir = OUT_ROOT / row["variant"]
    video, sheet, frames = render_kinematic(row, out_dir, overwrite)
    tgt = target_sheet(row, out_dir, overwrite)
    meta = {
        "variant": row["variant"],
        "route": row["route"],
        "source_task": row["source_task"],
        "derived_task": row["derived_task"],
        "target_route": row["target_route"],
        "target_npz": row["target_npz"],
        "frames": frames,
        "kinematic_video": rel(video),
        "kinematic_sheet": rel(sheet),
        "target_sheet": rel(tgt),
        "review_status": "PENDING_SUBAGENT",
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "README.md").write_text(
        "\n".join(
            [
                f"# Pre-CEM Visual Package: {row['variant']}",
                "",
                f"- route: `{row['route']}`",
                f"- source task: `{row['source_task']}`",
                f"- derived task: `{row['derived_task']}`",
                f"- target route: `{row['target_route']}`",
                f"- kinematic replay: [{video.name}]({video.name})",
                f"- kinematic sheet: [{sheet.name}]({sheet.name})",
                f"- target sheet: [{tgt.name}]({tgt.name})",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return meta


def write_gate(rows_out: list[dict[str, Any]]) -> None:
    GATE_TSV.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "variant",
        "route",
        "source_task",
        "derived_task",
        "target_route",
        "target_npz",
        "frames",
        "kinematic_video",
        "kinematic_sheet",
        "target_sheet",
        "review_status",
    ]
    with GATE_TSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows_out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("variants", nargs="*")
    args = parser.parse_args()

    variant_rows = read_variants()
    wanted = {row["variant"] for row in variant_rows} if args.all or not args.variants else set(args.variants)
    rendered = [render_one(row, args.overwrite) for row in variant_rows if row["variant"] in wanted]
    if not rendered:
        raise SystemExit("No variants rendered.")
    write_gate(rendered)
    print(f"wrote {rel(GATE_TSV)} rows={len(rendered)}")


if __name__ == "__main__":
    main()
