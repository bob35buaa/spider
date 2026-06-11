#!/usr/bin/env python3
"""Render MuJoCo keyframes/videos with E093 contact-geometry markers."""

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


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_contact_geometry import (  # noqa: E402
    HANDBOX_URDF,
    HANDS,
    HAND_PREFIX,
    THREEBOX,
    name2id,
    parse_handbox_urdf,
)


REPO = Path(__file__).resolve().parents[4]
OUT_ROOT = REPO / "workspace/core4d/results/E093/contact_geometry"
DEFAULT_MANIFEST = OUT_ROOT / "case_manifest.tsv"
DEFAULT_POINTS = OUT_ROOT / "per_frame_points.csv"
AUTO_CAMERA = "auto"


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


def local_to_world(obj_pos: np.ndarray, obj_mat: np.ndarray, local: np.ndarray) -> np.ndarray:
    return obj_pos + obj_mat @ local


def local_vec(row: dict[str, str], prefix: str) -> np.ndarray:
    return np.array([float(row[f"{prefix}_x"]), float(row[f"{prefix}_y"]), float(row[f"{prefix}_z"])], dtype=np.float64)


def finite_vec(vec: np.ndarray) -> bool:
    return bool(np.isfinite(vec).all())


def add_contact_markers(
    renderer: mujoco.Renderer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    frame_rows: list[dict[str, str]],
    handbox: dict[str, dict[str, np.ndarray]],
    wrist_ids: dict[str, int],
    geom_ids: dict[str, int],
) -> None:
    scene = renderer.scene
    obj_bid = name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    obj_pos = data.xpos[obj_bid].copy()
    obj_mat = data.xmat[obj_bid].reshape(3, 3).copy()
    identity = np.eye(3, dtype=np.float64)
    colors = {
        "left_raw": np.array([1.0, 0.55, 0.05, 1.0]),
        "right_raw": np.array([0.0, 0.65, 0.45, 1.0]),
        "left_wrist": np.array([0.85, 0.05, 0.05, 1.0]),
        "right_wrist": np.array([0.05, 0.25, 0.95, 1.0]),
        "left_sphere": np.array([0.85, 0.05, 0.05, 0.28]),
        "right_sphere": np.array([0.05, 0.25, 0.95, 0.28]),
        "handbox": np.array([0.1, 0.75, 0.2, 0.25]),
        "threebox": np.array([0.65, 0.2, 0.75, 0.22]),
        "contact_pos": np.array([0.1, 0.1, 0.1, 1.0]),
    }

    for row in frame_rows:
        hand = row["hand"]
        raw = local_vec(row, "raw_local")
        wrist5 = local_vec(row, "wrist5_local")
        contact = local_vec(row, "contact_pos_local")
        sphere = local_vec(row, "sphere_center_local")
        if finite_vec(raw) and row["raw_active"].lower() == "true":
            add_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.028, 0.0, 0.0]), local_to_world(obj_pos, obj_mat, raw), identity, colors[f"{hand}_raw"])
        if finite_vec(wrist5):
            add_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.022, 0.0, 0.0]), local_to_world(obj_pos, obj_mat, wrist5), identity, colors[f"{hand}_wrist"])
        if finite_vec(contact):
            add_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.014, 0.0, 0.0]), local_to_world(obj_pos, obj_mat, contact), identity, colors["contact_pos"])
        if finite_vec(sphere):
            radius = float(model.geom_size[geom_ids[hand], 0])
            add_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0.0, 0.0]), local_to_world(obj_pos, obj_mat, sphere), identity, colors[f"{hand}_sphere"])

        wrist_pos = data.xpos[wrist_ids[hand]].copy()
        wrist_mat = data.xmat[wrist_ids[hand]].reshape(3, 3).copy()
        hb = handbox[hand]
        add_geom(
            scene,
            mujoco.mjtGeom.mjGEOM_BOX,
            hb["half"],
            wrist_pos + wrist_mat @ hb["offset"],
            wrist_mat @ hb["rot"],
            colors["handbox"],
        )
        for _name, center_off, half, rot in THREEBOX[hand]:
            add_geom(
                scene,
                mujoco.mjtGeom.mjGEOM_BOX,
                half,
                wrist_pos + wrist_mat @ center_off,
                wrist_mat @ rot,
                colors["threebox"],
            )


def object_box_world_corners(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    obj_bid = name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    obj_gid = name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_collision")
    half = model.geom_size[obj_gid].astype(np.float64)
    signs = np.array(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ],
        dtype=np.float64,
    )
    local = signs * half
    obj_pos = data.xpos[obj_bid].copy()
    obj_mat = data.xmat[obj_bid].reshape(3, 3).copy()
    return local @ obj_mat.T + obj_pos


def frame_marker_world_points(data: mujoco.MjData, model: mujoco.MjModel, frame_rows: list[dict[str, str]]) -> np.ndarray:
    if not frame_rows:
        return np.empty((0, 3), dtype=np.float64)
    obj_bid = name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    obj_pos = data.xpos[obj_bid].copy()
    obj_mat = data.xmat[obj_bid].reshape(3, 3).copy()
    points = []
    for row in frame_rows:
        for prefix in ("raw_local", "wrist5_local", "contact_pos_local", "sphere_center_local"):
            point = local_vec(row, prefix)
            if finite_vec(point):
                points.append(local_to_world(obj_pos, obj_mat, point))
    if not points:
        return np.empty((0, 3), dtype=np.float64)
    return np.stack(points, axis=0)


def make_auto_camera(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: np.ndarray,
    points_by_frame: dict[int, list[dict[str, str]]],
    *,
    azimuth: float,
    elevation: float,
    distance_scale: float,
    min_distance: float,
    lookat_z: float | None,
) -> tuple[mujoco.MjvCamera, dict[str, float]]:
    sample_count = min(48, int(qpos.shape[0]))
    sample_frames = np.unique(np.linspace(0, qpos.shape[0] - 1, num=sample_count, dtype=int))
    bounds_points = []
    for frame in sample_frames:
        data.qpos[:] = qpos[int(frame)]
        mujoco.mj_forward(model, data)
        bounds_points.append(data.xpos[1:].copy())
        bounds_points.append(object_box_world_corners(model, data))
        marker_points = frame_marker_world_points(data, model, points_by_frame.get(int(frame), []))
        if marker_points.size:
            bounds_points.append(marker_points)

    points = np.concatenate(bounds_points, axis=0)
    points = points[np.isfinite(points).all(axis=1)]
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    center = (lo + hi) * 0.5
    span = hi - lo

    if lookat_z is None:
        center[2] = float(np.clip(center[2], 0.45, 0.75))
    else:
        center[2] = float(lookat_z)

    half_xy = 0.5 * float(np.linalg.norm(span[:2]))
    half_z = 0.5 * float(span[2])
    radius = max(half_xy, half_z * 1.8, 0.9)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.type = int(mujoco.mjtCamera.mjCAMERA_FREE)
    cam.lookat[:] = center
    cam.distance = max(float(min_distance), radius * float(distance_scale))
    cam.azimuth = float(azimuth)
    cam.elevation = float(elevation)

    meta = {
        "camera_lookat_x": float(center[0]),
        "camera_lookat_y": float(center[1]),
        "camera_lookat_z": float(center[2]),
        "camera_distance": float(cam.distance),
        "camera_azimuth": float(cam.azimuth),
        "camera_elevation": float(cam.elevation),
        "camera_span_x": float(span[0]),
        "camera_span_y": float(span[1]),
        "camera_span_z": float(span[2]),
    }
    return cam, meta


def render_case(
    row: dict[str, str],
    points_by_frame: dict[int, list[dict[str, str]]],
    handbox: dict[str, dict[str, np.ndarray]],
    out_root: Path,
    camera: str,
    video_frames: int,
    width: int,
    height: int,
    camera_azimuth: float,
    camera_elevation: float,
    camera_distance_scale: float,
    camera_min_distance: float,
    camera_lookat_z: float | None,
) -> dict[str, Any]:
    task = row["task"]
    case_id = row["case_id"]
    model = mujoco.MjModel.from_xml_path(row["scene_xml"])
    data = mujoco.MjData(model)
    qpos = np.load(row["trajectory_npz"], allow_pickle=True)["qpos"].astype(np.float64)
    wrist_ids = {
        "left": name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link"),
        "right": name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link"),
    }
    geom_ids = {"left": name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "lh"), "right": name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "rh")}

    key_dir = out_root / "visuals/mujoco/keyframes"
    video_dir = out_root / "visuals/mujoco/videos"
    key_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    key_path = key_dir / f"{case_id}_{task}_mujoco_keyframes.png"
    video_path = video_dir / f"{case_id}_{task}_geometry.mp4"

    renderer = mujoco.Renderer(model, width=width, height=height)
    camera_arg: str | mujoco.MjvCamera
    camera_meta: dict[str, float]
    if camera.lower() == AUTO_CAMERA:
        camera_arg, camera_meta = make_auto_camera(
            model,
            data,
            qpos,
            points_by_frame,
            azimuth=camera_azimuth,
            elevation=camera_elevation,
            distance_scale=camera_distance_scale,
            min_distance=camera_min_distance,
            lookat_z=camera_lookat_z,
        )
    else:
        camera_arg = camera
        camera_meta = {
            "camera_lookat_x": float("nan"),
            "camera_lookat_y": float("nan"),
            "camera_lookat_z": float("nan"),
            "camera_distance": float("nan"),
            "camera_azimuth": float("nan"),
            "camera_elevation": float("nan"),
            "camera_span_x": float("nan"),
            "camera_span_y": float("nan"),
            "camera_span_z": float("nan"),
        }

    def render_frame(frame: int) -> np.ndarray:
        data.qpos[:] = qpos[frame]
        mujoco.mj_forward(model, data)
        try:
            renderer.update_scene(data, camera=camera_arg)
        except Exception:
            renderer.update_scene(data)
        add_contact_markers(renderer, model, data, points_by_frame.get(frame, []), handbox, wrist_ids, geom_ids)
        return renderer.render()

    key_frames = np.linspace(0, qpos.shape[0] - 1, num=min(6, qpos.shape[0]), dtype=int)
    key_images = [render_frame(int(f)) for f in key_frames]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for ax, img, frame in zip(axes.flat, key_images, key_frames):
        ax.imshow(img)
        ax.set_title(f"f{int(frame)}")
        ax.axis("off")
    for ax in axes.flat[len(key_images) :]:
        ax.axis("off")
    fig.suptitle(f"{case_id} | {task} | raw orange/green, wrist+5cm red/blue, sphere translucent, handbox green, 3-box purple")
    fig.savefig(key_path, dpi=150)
    plt.close(fig)

    video_idxs = np.linspace(0, qpos.shape[0] - 1, num=min(video_frames, qpos.shape[0]), dtype=int)
    writer = imageio.get_writer(video_path, fps=8, codec="libx264", quality=7)
    for frame in video_idxs:
        writer.append_data(render_frame(int(frame)))
    writer.close()
    renderer.close()
    return {
        "case_id": case_id,
        "task": task,
        "camera": camera,
        "keyframes_png": str(key_path),
        "video_mp4": str(video_path),
        "video_frames": int(len(video_idxs)),
        "width": int(width),
        "height": int(height),
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
    parser.add_argument("--camera", default=AUTO_CAMERA, help="MuJoCo camera name or 'auto' for fixed full-body free camera.")
    parser.add_argument("--video-frames", type=int, default=64)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-20.0)
    parser.add_argument("--camera-distance-scale", type=float, default=3.2)
    parser.add_argument("--camera-min-distance", type=float, default=3.4)
    parser.add_argument("--camera-lookat-z", type=float, default=None)
    parser.add_argument("--tasks", nargs="*", default=None)
    args = parser.parse_args()

    manifest = [r for r in read_tsv(args.manifest) if str(r.get("ready", "")).lower() == "true"]
    if args.tasks:
        wanted = set(args.tasks)
        manifest = [r for r in manifest if r["task"] in wanted or r["case_id"] in wanted]
    point_rows = read_tsv(args.points)
    points_by_case: dict[str, dict[int, list[dict[str, str]]]] = {}
    for row in point_rows:
        points_by_case.setdefault(row["case_id"], {}).setdefault(int(row["frame"]), []).append(row)
    handbox = parse_handbox_urdf(HANDBOX_URDF)

    records: list[dict[str, Any]] = []
    for row in manifest:
        print(f"[E093-render] {row['case_id']} {row['task']}")
        records.append(
            render_case(
                row,
                points_by_case.get(row["case_id"], {}),
                handbox,
                args.out_root,
                args.camera,
                args.video_frames,
                args.width,
                args.height,
                args.camera_azimuth,
                args.camera_elevation,
                args.camera_distance_scale,
                args.camera_min_distance,
                args.camera_lookat_z,
            )
        )
    fields = [
        "case_id",
        "task",
        "camera",
        "keyframes_png",
        "video_mp4",
        "video_frames",
        "width",
        "height",
        "camera_lookat_x",
        "camera_lookat_y",
        "camera_lookat_z",
        "camera_distance",
        "camera_azimuth",
        "camera_elevation",
        "camera_span_x",
        "camera_span_y",
        "camera_span_z",
        "keyframes_exists",
        "video_exists",
        "keyframes_bytes",
        "video_bytes",
    ]
    write_tsv(args.out_root / "visuals/mujoco/mujoco_render_manifest.tsv", records, fields)
    print(f"[E093-render] rendered {len(records)} cases")


if __name__ == "__main__":
    main()
