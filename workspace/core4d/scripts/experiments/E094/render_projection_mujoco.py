#!/usr/bin/env python3
"""Render E094 projection targets in MuJoCo."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[4]
E093_DIR = REPO / "workspace/core4d/scripts/E093"
if str(E093_DIR) not in sys.path:
    sys.path.insert(0, str(E093_DIR))

from audit_contact_geometry import HANDBOX_URDF, HANDS, name2id, parse_handbox_urdf  # noqa: E402


E093_ROOT = REPO / "workspace/core4d/results/E093/contact_geometry"
OUT_ROOT = REPO / "workspace/core4d/results/E094/handbox_target_projection"
DEFAULT_MANIFEST = E093_ROOT / "case_manifest.tsv"
DEFAULT_POINTS = OUT_ROOT / "per_frame_projection.csv"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def add_geom(scene: mujoco.MjvScene, geom_type: mujoco.mjtGeom, size: np.ndarray, pos: np.ndarray, mat: np.ndarray, rgba: np.ndarray) -> None:
    if scene.ngeom >= len(scene.geoms):
        return
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], geom_type, size.astype(float), pos.astype(float), mat.reshape(-1).astype(float), rgba.astype(float))
    scene.ngeom += 1


def local_vec(row: dict[str, str], prefix: str) -> np.ndarray:
    return np.array([float(row[f"{prefix}_x"]), float(row[f"{prefix}_y"]), float(row[f"{prefix}_z"])], dtype=np.float64)


def finite_vec(vec: np.ndarray) -> bool:
    return bool(np.isfinite(vec).all())


def local_to_world(obj_pos: np.ndarray, obj_mat: np.ndarray, local: np.ndarray) -> np.ndarray:
    return obj_pos + obj_mat @ local


def object_box_world_corners(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    obj_bid = name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    obj_gid = name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_collision")
    half = model.geom_size[obj_gid].astype(np.float64)
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float64)
    obj_pos = data.xpos[obj_bid].copy()
    obj_mat = data.xmat[obj_bid].reshape(3, 3).copy()
    return (signs * half) @ obj_mat.T + obj_pos


def frame_marker_world_points(data: mujoco.MjData, model: mujoco.MjModel, rows: list[dict[str, str]]) -> np.ndarray:
    if not rows:
        return np.empty((0, 3), dtype=np.float64)
    obj_bid = name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    obj_pos = data.xpos[obj_bid].copy()
    obj_mat = data.xmat[obj_bid].reshape(3, 3).copy()
    pts = []
    for row in rows:
        for prefix in ("raw_local", "old_wrist5_local", "support_patch_local", "reward_target_local"):
            point = local_vec(row, prefix)
            if finite_vec(point):
                pts.append(local_to_world(obj_pos, obj_mat, point))
    return np.stack(pts, axis=0) if pts else np.empty((0, 3), dtype=np.float64)


def make_auto_camera(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: np.ndarray,
    points_by_frame: dict[int, list[dict[str, str]]],
    azimuth: float,
    elevation: float,
    distance_scale: float,
    min_distance: float,
) -> tuple[mujoco.MjvCamera, dict[str, float]]:
    frames = np.unique(np.linspace(0, qpos.shape[0] - 1, num=min(48, qpos.shape[0]), dtype=int))
    pts = []
    for frame in frames:
        data.qpos[:] = qpos[int(frame)]
        mujoco.mj_forward(model, data)
        pts.append(data.xpos[1:].copy())
        pts.append(object_box_world_corners(model, data))
        marker_pts = frame_marker_world_points(data, model, points_by_frame.get(int(frame), []))
        if marker_pts.size:
            pts.append(marker_pts)
    all_pts = np.concatenate(pts, axis=0)
    all_pts = all_pts[np.isfinite(all_pts).all(axis=1)]
    lo = all_pts.min(axis=0)
    hi = all_pts.max(axis=0)
    center = (lo + hi) * 0.5
    span = hi - lo
    center[2] = float(np.clip(center[2], 0.45, 0.75))
    radius = max(0.5 * float(np.linalg.norm(span[:2])), 0.5 * float(span[2]) * 1.8, 0.9)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.type = int(mujoco.mjtCamera.mjCAMERA_FREE)
    cam.lookat[:] = center
    cam.distance = max(min_distance, radius * distance_scale)
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam, {
        "camera_lookat_x": float(center[0]),
        "camera_lookat_y": float(center[1]),
        "camera_lookat_z": float(center[2]),
        "camera_distance": float(cam.distance),
    }


def add_projection_markers(
    renderer: mujoco.Renderer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    frame_rows: list[dict[str, str]],
    handbox: dict[str, dict[str, np.ndarray]],
    wrist_ids: dict[str, int],
) -> None:
    scene = renderer.scene
    obj_bid = name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    obj_pos = data.xpos[obj_bid].copy()
    obj_mat = data.xmat[obj_bid].reshape(3, 3).copy()
    identity = np.eye(3, dtype=np.float64)
    colors = {
        "raw": {"left": np.array([1.0, 0.55, 0.05, 1.0]), "right": np.array([0.0, 0.65, 0.45, 1.0])},
        "old": {"left": np.array([0.85, 0.05, 0.05, 1.0]), "right": np.array([0.05, 0.25, 0.95, 1.0])},
        "patch": {"left": np.array([1.0, 0.9, 0.0, 1.0]), "right": np.array([0.0, 0.85, 1.0, 1.0])},
        "reward": {"left": np.array([0.7, 0.1, 0.9, 1.0]), "right": np.array([0.5, 0.0, 0.9, 1.0])},
        "handbox": np.array([0.1, 0.75, 0.2, 0.23]),
    }
    for row in frame_rows:
        hand = row["hand"]
        raw = local_vec(row, "raw_local")
        old = local_vec(row, "old_wrist5_local")
        patch = local_vec(row, "support_patch_local")
        reward = local_vec(row, "reward_target_local")
        if finite_vec(raw) and row["raw_active"].lower() == "true":
            add_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.026, 0.0, 0.0]), local_to_world(obj_pos, obj_mat, raw), identity, colors["raw"][hand])
        if finite_vec(old):
            add_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.018, 0.0, 0.0]), local_to_world(obj_pos, obj_mat, old), identity, colors["old"][hand])
        if finite_vec(patch):
            add_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.023, 0.0, 0.0]), local_to_world(obj_pos, obj_mat, patch), identity, colors["patch"][hand])
        if finite_vec(reward):
            add_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.019, 0.0, 0.0]), local_to_world(obj_pos, obj_mat, reward), identity, colors["reward"][hand])

        wrist_pos = data.xpos[wrist_ids[hand]].copy()
        wrist_mat = data.xmat[wrist_ids[hand]].reshape(3, 3).copy()
        hb = handbox[hand]
        add_geom(scene, mujoco.mjtGeom.mjGEOM_BOX, hb["half"], wrist_pos + wrist_mat @ hb["offset"], wrist_mat @ hb["rot"], colors["handbox"])


def render_case(
    row: dict[str, str],
    points_by_frame: dict[int, list[dict[str, str]]],
    handbox: dict[str, dict[str, np.ndarray]],
    out_root: Path,
    width: int,
    height: int,
    video_frames: int,
) -> dict[str, Any]:
    case_id = row["case_id"]
    task = row["task"]
    model = mujoco.MjModel.from_xml_path(row["scene_xml"])
    data = mujoco.MjData(model)
    qpos = np.load(row["trajectory_npz"], allow_pickle=True)["qpos"].astype(np.float64)
    wrist_ids = {
        "left": name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link"),
        "right": name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link"),
    }
    key_dir = out_root / "visuals/mujoco/keyframes"
    video_dir = out_root / "visuals/mujoco/videos"
    key_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    key_path = key_dir / f"{case_id}_{task}_projection_keyframes.png"
    video_path = video_dir / f"{case_id}_{task}_projection.mp4"
    renderer = mujoco.Renderer(model, width=width, height=height)
    camera, camera_meta = make_auto_camera(model, data, qpos, points_by_frame, 135.0, -20.0, 3.2, 3.4)

    def render_frame(frame: int) -> np.ndarray:
        data.qpos[:] = qpos[frame]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        add_projection_markers(renderer, model, data, points_by_frame.get(frame, []), handbox, wrist_ids)
        return renderer.render()

    key_frames = np.linspace(0, qpos.shape[0] - 1, num=min(6, qpos.shape[0]), dtype=int)
    images = [render_frame(int(f)) for f in key_frames]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for ax, img, frame in zip(axes.flat, images, key_frames):
        ax.imshow(img)
        ax.set_title(f"f{int(frame)}")
        ax.axis("off")
    for ax in axes.flat[len(images) :]:
        ax.axis("off")
    fig.suptitle(f"{case_id} | {task} | raw orange/green, old red/blue, patch yellow/cyan, reward purple, handbox green")
    fig.savefig(key_path, dpi=150)
    plt.close(fig)

    idxs = np.linspace(0, qpos.shape[0] - 1, num=min(video_frames, qpos.shape[0]), dtype=int)
    writer = imageio.get_writer(video_path, fps=8, codec="libx264", quality=7)
    for frame in idxs:
        writer.append_data(render_frame(int(frame)))
    writer.close()
    renderer.close()
    return {
        "case_id": case_id,
        "task": task,
        "keyframes_png": str(key_path),
        "video_mp4": str(video_path),
        "video_frames": int(len(idxs)),
        "width": width,
        "height": height,
        "keyframes_exists": key_path.is_file(),
        "video_exists": video_path.is_file(),
        "keyframes_bytes": key_path.stat().st_size if key_path.is_file() else 0,
        "video_bytes": video_path.stat().st_size if video_path.is_file() else 0,
        **camera_meta,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--points", type=Path, default=DEFAULT_POINTS)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--video-frames", type=int, default=48)
    args = parser.parse_args()

    point_rows = read_tsv(args.points)
    case_ids = {r["case_id"] for r in point_rows}
    manifest = [r for r in read_tsv(args.manifest) if r["case_id"] in case_ids and str(r.get("ready", "")).lower() == "true"]
    if args.tasks:
        wanted = set(args.tasks)
        manifest = [r for r in manifest if r["case_id"] in wanted or r["task"] in wanted]
    points_by_case: dict[str, dict[int, list[dict[str, str]]]] = {}
    for row in point_rows:
        points_by_case.setdefault(row["case_id"], {}).setdefault(int(row["frame"]), []).append(row)
    handbox = parse_handbox_urdf(HANDBOX_URDF)
    records = []
    for row in manifest:
        print(f"[E094-render] {row['case_id']} {row['task']}")
        records.append(render_case(row, points_by_case.get(row["case_id"], {}), handbox, args.out_root, args.width, args.height, args.video_frames))
    fields = [
        "case_id",
        "task",
        "keyframes_png",
        "video_mp4",
        "video_frames",
        "width",
        "height",
        "camera_lookat_x",
        "camera_lookat_y",
        "camera_lookat_z",
        "camera_distance",
        "keyframes_exists",
        "video_exists",
        "keyframes_bytes",
        "video_bytes",
    ]
    write_tsv(args.out_root / "visuals/mujoco/projection_render_manifest.tsv", records, fields)
    print(f"[E094-render] rendered {len(records)} cases")


if __name__ == "__main__":
    main()
