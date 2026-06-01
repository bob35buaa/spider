#!/usr/bin/env python3
"""Render E106 pre-CEM MuJoCo kinematic replays for clean derived tasks."""

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

from e106_common import RESULTS_ROOT, TASK_ROOT, rel, read_variants  # noqa: E402


OUT_ROOT = RESULTS_ROOT / "pre_cem_visuals"
REVIEW_ROOT = RESULTS_ROOT / "pre_cem_visual_review"
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


def render_variant(row: dict[str, str], overwrite: bool) -> dict[str, Any]:
    task_dir = TASK_ROOT / row["derived_task"]
    scene = task_dir / "scene.xml"
    traj = task_dir / "0/trajectory_kinematic.npz"
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    qpos = np.load(traj, allow_pickle=True)["qpos"]

    out_dir = OUT_ROOT / row["variant"]
    review_dir = REVIEW_ROOT / row["variant"]
    out_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)
    video = out_dir / f"{row['variant']}_kinematic_replay.mp4"
    sheet = out_dir / f"{row['variant']}_kinematic_sheet.png"

    if not (video.is_file() and sheet.is_file()) or overwrite:
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
        fig.suptitle(f"{row['variant']} | {row['derived_task']} | E106 clean kinematic replay")
        fig.savefig(sheet, dpi=150)
        plt.close(fig)

    meta = {
        "variant": row["variant"],
        "source_task": row["source_task"],
        "derived_task": row["derived_task"],
        "split": row["split"],
        "frames": int(qpos.shape[0]),
        "kinematic_video": rel(video),
        "kinematic_sheet": rel(sheet),
        "review_dir": rel(review_dir),
        "review_status": "PENDING_SUBAGENT",
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    review = review_dir / "REVIEW.md"
    if not review.is_file():
        review.write_text(
            "\n".join(
                [
                    f"# E106 Pre-CEM Review: {row['variant']}",
                    "",
                    "Status: PENDING_SUBAGENT",
                    "",
                    f"- source task: `{row['source_task']}`",
                    f"- clean task: `{row['derived_task']}`",
                    f"- split: `{row['split']}`",
                    f"- kinematic replay: `{rel(video)}`",
                    f"- kinematic sheet: `{rel(sheet)}`",
                    "",
                    "Medium subagent should replace `Status: PENDING_SUBAGENT` with `Status: PASS`, `Status: PASS_WITH_NOTES`, or `Status: FAIL_PRE_CEM_VISUAL` after visual inspection.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    return meta


def write_gate(rows: list[dict[str, Any]]) -> None:
    fields = [
        "variant",
        "source_task",
        "derived_task",
        "split",
        "frames",
        "kinematic_video",
        "kinematic_sheet",
        "review_dir",
        "review_status",
    ]
    GATE_TSV.parent.mkdir(parents=True, exist_ok=True)
    with GATE_TSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("variants", nargs="*")
    args = parser.parse_args()

    rows = read_variants()
    wanted = {row["variant"] for row in rows} if args.all or not args.variants else set(args.variants)
    rendered = [render_variant(row, args.overwrite) for row in rows if row["variant"] in wanted]
    if not rendered:
        raise SystemExit("No E106 variants rendered.")
    write_gate(rendered)
    print(f"wrote {rel(GATE_TSV)} rows={len(rendered)}")


if __name__ == "__main__":
    main()
