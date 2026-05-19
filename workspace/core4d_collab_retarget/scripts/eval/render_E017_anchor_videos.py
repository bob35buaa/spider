#!/usr/bin/env python3
"""Render anchor-position videos for E017 audit cases.

The regular E016/E017 comparison videos answer whether a rollout works. This
script answers a narrower question: where do E014 GT, E016 centroid, and E017
auto anchors sit on the moving reference object?
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import imageio
import mujoco
import numpy as np
import torch

from spider.interp import interp


REPO = Path(__file__).resolve().parents[4]
BASE = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
RESULTS = REPO / "workspace/core4d_collab_retarget/results/E017"
METHOD_AUDIT = RESULTS / "anchor_method_audit.csv"
OUT_DIR = RESULTS / "anchor_visual"


@dataclass(frozen=True)
class AnchorMarker:
    label: str
    point_local: np.ndarray
    rgba: tuple[float, float, float, float]
    bgr: tuple[int, int, int]


def read_method_rows() -> dict[str, list[dict[str, str]]]:
    with METHOD_AUDIT.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        out.setdefault(row["source_variant"], []).append(row)
    return out


def marker_from_row(row: dict[str, str], label: str, rgba: tuple[float, float, float, float]) -> AnchorMarker:
    point = np.array(
        [float(row["anchor_x"]), float(row["anchor_y"]), float(row["anchor_z"])],
        dtype=np.float64,
    )
    bgr = tuple(int(255 * c) for c in (rgba[2], rgba[1], rgba[0]))
    return AnchorMarker(label, point, rgba, bgr)


def build_markers(source_variant: str, rows_by_variant: dict[str, list[dict[str, str]]]) -> tuple[str, list[AnchorMarker]]:
    rows = rows_by_variant[source_variant]
    rows_by_method = {row["method"]: row for row in rows}
    source_task = rows[0]["source_task"]
    markers: list[AnchorMarker] = []

    e016 = rows_by_method.get("E016_centroid")
    e017 = rows_by_method.get("E017_auto")
    if e016 is not None:
        markers.append(marker_from_row(e016, "E016 centroid", (1.0, 0.08, 0.06, 1.0)))
    if e017 is not None:
        markers.append(marker_from_row(e017, "E017 auto", (0.05, 0.32, 1.0, 1.0)))
    if e016 is not None and e016.get("gt_available") == "True":
        gt = np.array(
            [
                float(e016["gt_anchor_x"]),
                float(e016["gt_anchor_y"]),
                float(e016["gt_anchor_z"]),
            ],
            dtype=np.float64,
        )
        markers.insert(0, AnchorMarker("E014 GT", gt, (0.0, 0.85, 0.20, 1.0), (51, 217, 0)))
    return source_task, markers


def object_qadr(model: mujoco.MjModel) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    if body_id < 0:
        raise ValueError("model has no body named object")
    joint_id = model.body_jntadr[body_id]
    if joint_id < 0:
        raise ValueError("object body has no joint")
    return int(model.jnt_qposadr[joint_id])


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    qvec = q[1:]
    uv = np.cross(qvec, v)
    uuv = np.cross(qvec, uv)
    return v + 2.0 * (q[0] * uv + uuv)


def upsample_ref(qpos: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return qpos
    qpos_t = torch.from_numpy(qpos).to(torch.float32).unsqueeze(0)
    return interp(qpos_t, factor).squeeze(0).numpy()


def add_marker_geoms(
    scene: mujoco.MjvScene,
    positions: list[np.ndarray],
    markers: list[AnchorMarker],
    *,
    radius: float,
) -> None:
    for pos, marker in zip(positions, markers):
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([radius, 0.0, 0.0], dtype=np.float64),
            np.asarray(pos, dtype=np.float64),
            np.eye(3, dtype=np.float64).reshape(-1),
            np.asarray(marker.rgba, dtype=np.float32),
        )
        scene.ngeom += 1


def update_scene(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    camera: int | str | mujoco.MjvCamera,
    markers: list[AnchorMarker],
    positions: list[np.ndarray],
    *,
    radius: float,
) -> np.ndarray:
    renderer.update_scene(data, camera)
    add_marker_geoms(renderer.scene, positions, markers, radius=radius)
    return renderer.render()


def draw_legend(
    image: np.ndarray,
    *,
    title: str,
    panel: str,
    markers: list[AnchorMarker],
    frame_idx: int,
    frame_count: int,
) -> np.ndarray:
    out = image.copy()
    cv2.putText(out, f"{title} | {panel}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 2)
    cv2.putText(
        out,
        f"frame {frame_idx + 1}/{frame_count}",
        (10, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (80, 80, 80),
        1,
    )
    y = 70
    for marker in markers:
        cv2.rectangle(out, (12, y - 10), (28, y + 6), marker.bgr, thickness=-1)
        x, yloc, z = marker.point_local
        text = f"{marker.label}: [{x:+.3f}, {yloc:+.3f}, {z:+.3f}]"
        cv2.putText(out, text, (36, y + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.46, marker.bgr, 1)
        y += 22
    return out


def top_camera(obj_pos: np.ndarray, distance: float) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = obj_pos
    cam.distance = distance
    cam.azimuth = 90.0
    cam.elevation = -90.0
    return cam


def render_case(
    source_variant: str,
    source_task: str,
    markers: list[AnchorMarker],
    *,
    width: int,
    height: int,
    fps: int,
    upsample: int,
    force: bool,
) -> tuple[Path, Path]:
    task_dir = BASE / source_task
    scene = task_dir / "scene.xml"
    kin = task_dir / "0/trajectory_kinematic.npz"
    for path in (scene, kin):
        if not path.is_file():
            raise FileNotFoundError(path)

    out = OUT_DIR / f"{source_variant}_anchor_positions.mp4"
    sheet_dir = OUT_DIR / f"{source_variant}_frames"
    sheet = sheet_dir / "sheet.jpg"
    if out.is_file() and out.stat().st_size > 0 and not force:
        return out, sheet

    model = mujoco.MjModel.from_xml_path(str(scene))
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    renderer = mujoco.Renderer(model, height=height, width=width)
    data = mujoco.MjData(model)

    qpos = np.load(kin)["qpos"].reshape(-1, model.nq)
    qpos = upsample_ref(qpos, upsample)
    qadr = object_qadr(model)
    obj_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    obj_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_collision")
    half = model.geom_size[obj_geom, :3].astype(np.float64)
    marker_radius = float(max(0.025, min(0.06, np.min(half[:2]) * 0.22)))
    top_distance = float(max(1.2, 4.2 * np.linalg.norm(half[:2])))

    frames: list[np.ndarray] = []
    frame_count = len(qpos)
    for i, q in enumerate(qpos):
        data.qpos[: model.nq] = q[: model.nq]
        mujoco.mj_forward(model, data)
        obj_pos = data.xpos[obj_body].copy()
        obj_quat = q[qadr + 3 : qadr + 7].astype(np.float64)
        positions = [obj_pos + quat_apply(obj_quat, marker.point_local) for marker in markers]

        front = update_scene(renderer, data, "track", markers, positions, radius=marker_radius)
        front = draw_legend(
            front,
            title=source_variant,
            panel="front",
            markers=markers,
            frame_idx=i,
            frame_count=frame_count,
        )

        top = update_scene(renderer, data, top_camera(obj_pos, top_distance), markers, positions, radius=marker_radius)
        top = draw_legend(
            top,
            title=source_variant,
            panel="top",
            markers=markers,
            frame_idx=i,
            frame_count=frame_count,
        )
        frames.append(np.concatenate([front, top], axis=1))

    renderer.close()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out), frames, fps=fps, codec="libx264", quality=8)

    sheet_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(out),
            "-vf",
            "select='eq(n,0)+eq(n,50)+eq(n,100)+eq(n,150)+eq(n,200)+eq(n,250)',scale=360:-1,tile=3x2",
            "-frames:v",
            "1",
            str(sheet),
        ],
        cwd=REPO,
        check=True,
    )
    print(f"Video saved: {out} ({len(frames)} frames, {fps} fps)")
    return out, sheet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="source variant such as E016_box023_p2. Defaults to GT cases.",
    )
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--upsample", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    rows_by_variant = read_method_rows()
    cases = args.case or ["E016_box023_p2", "E016_box025_p2"]
    index: list[str] = ["# E017 Anchor Position Videos", ""]
    for source_variant in cases:
        source_task, markers = build_markers(source_variant, rows_by_variant)
        video, sheet = render_case(
            source_variant,
            source_task,
            markers,
            width=args.width,
            height=args.height,
            fps=args.fps,
            upsample=args.upsample,
            force=args.force,
        )
        index.append(f"- `{source_variant}`: `{video.relative_to(REPO)}`")
        index.append(f"  - sheet: `{sheet.relative_to(REPO)}`")
    (OUT_DIR / "anchor_visual_eval.md").write_text("\n".join(index) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
