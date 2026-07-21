#!/usr/bin/env python3
"""Render E167 B2 postprocess NPZ files as MuJoCo replay MP4s."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R


DEFAULT_VARIANTS = Path("workspace/core4d/scripts/experiments/E167/variants.tsv")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def selected_postprocess_rows(rows: Iterable[dict[str, str]], specs: set[tuple[str, str]]) -> list[dict[str, str]]:
    out = []
    for row in rows:
        if row.get("arm_kind") != "postprocess":
            continue
        if specs and (row["short_case_id"], row["arm"]) not in specs and (row["short_case_id"], "") not in specs:
            continue
        out.append(row)
    return out


def parse_case_arms(text: str) -> set[tuple[str, str]]:
    specs: set[tuple[str, str]] = set()
    for item in text.split():
        if not item:
            continue
        if ":" in item:
            case, arm = item.split(":", 1)
        else:
            case, arm = item, ""
        specs.add((case, arm))
    return specs


def qpos_from_npz(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    qpos = np.asarray(data["qpos"], dtype=np.float64)
    if qpos.ndim == 3:
        # SPIDER CEM trajectory convention: world 0 is the nominal replay track.
        qpos = qpos[:, 0, :]
    if qpos.ndim != 2:
        raise ValueError(f"Unsupported qpos shape in {path}: {qpos.shape}")
    return qpos


def raw_ref_qpos(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    qpos = np.asarray(data["qpos"], dtype=np.float64)
    if qpos.ndim != 2:
        raise ValueError(f"Unsupported reference qpos shape in {path}: {qpos.shape}")
    return qpos


def convert_ref_to_scene_act(raw_qpos: np.ndarray, model: mujoco.MjModel, scene_xml: Path) -> np.ndarray:
    if raw_qpos.shape[1] == model.nq:
        return raw_qpos
    if raw_qpos.shape[1] < model.nq or model.nq < 6:
        raise ValueError(f"Cannot convert ref qpos shape {raw_qpos.shape} to model.nq={model.nq}")

    nq_model = int(model.nq)
    nq_robot = nq_model - 6
    obj_pos_world = raw_qpos[:, nq_robot : nq_robot + 3]
    obj_quat_wxyz = raw_qpos[:, nq_robot + 3 : nq_robot + 7]

    obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    if obj_body_id < 0:
        raise ValueError(f"Scene has no body named object: {scene_xml}")

    body_pos = model.body_pos[obj_body_id]
    body_quat_wxyz = model.body_quat[obj_body_id]
    body_quat_xyzw = [
        body_quat_wxyz[1],
        body_quat_wxyz[2],
        body_quat_wxyz[3],
        body_quat_wxyz[0],
    ]
    r_body = R.from_quat(body_quat_xyzw)
    obj_slide_pos = r_body.inv().apply(obj_pos_world - body_pos[np.newaxis, :])

    meta_path = scene_xml.parent / "scene_act_meta.json"
    euler_conv = "XYZ"
    if meta_path.is_file():
        euler_conv = json.loads(meta_path.read_text()).get("euler_convention", "XYZ")

    obj_quat_xyzw = np.column_stack(
        [
            obj_quat_wxyz[:, 1],
            obj_quat_wxyz[:, 2],
            obj_quat_wxyz[:, 3],
            obj_quat_wxyz[:, 0],
        ]
    )
    r_world = R.from_quat(obj_quat_xyzw)
    obj_euler = (r_body.inv() * r_world).as_euler(euler_conv)

    converted = np.zeros((raw_qpos.shape[0], nq_model), dtype=np.float64)
    converted[:, :nq_robot] = raw_qpos[:, :nq_robot]
    converted[:, nq_robot : nq_robot + 3] = obj_slide_pos
    converted[:, nq_robot + 3 : nq_robot + 6] = obj_euler
    return converted


def build_camera(model: mujoco.MjModel, qpos_tracks: list[np.ndarray]) -> mujoco.MjvCamera:
    data = mujoco.MjData(model)
    pts = []
    stride = max(1, max(track.shape[0] for track in qpos_tracks) // 80)
    for track in qpos_tracks:
        for qpos in track[::stride]:
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            pts.append(data.xpos[1:].copy())
    all_pts = np.concatenate(pts, axis=0)
    all_pts = all_pts[np.isfinite(all_pts).all(axis=1)]
    if all_pts.size == 0:
        center = np.array([0.0, 0.0, 0.8], dtype=np.float64)
        radius = 1.0
    else:
        lo = all_pts.min(axis=0)
        hi = all_pts.max(axis=0)
        center = (lo + hi) * 0.5
        span = np.maximum(hi - lo, 1e-6)
        radius = max(0.5 * float(np.linalg.norm(span[:2])), 0.5 * float(span[2]) * 1.7, 1.0)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.type = int(mujoco.mjtCamera.mjCAMERA_FREE)
    cam.lookat[:] = center
    cam.distance = max(3.8, radius * 3.0)
    cam.azimuth = 135.0
    cam.elevation = -22.0
    return cam


def render_panel(
    renderer: mujoco.Renderer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: np.ndarray,
    camera: mujoco.MjvCamera,
    label: str,
) -> np.ndarray:
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    options = mujoco.MjvOption()
    mujoco.mjv_defaultOption(options)
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    renderer.update_scene(data, camera=camera, scene_option=options)
    image = renderer.render()
    cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
    return image


def render_video(
    scene_xml: Path,
    ref_qpos: np.ndarray,
    sim_qpos: np.ndarray,
    out_path: Path,
    fps: int,
    repeat: int,
    max_frames: int | None,
) -> None:
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    if sim_qpos.shape[1] != model.nq:
        raise ValueError(f"Sim qpos nq mismatch for {out_path}: {sim_qpos.shape[1]} vs {model.nq}")
    ref_qpos = convert_ref_to_scene_act(ref_qpos, model, scene_xml)
    if ref_qpos.shape[1] != model.nq:
        raise ValueError(f"Ref qpos nq mismatch for {out_path}: {ref_qpos.shape[1]} vs {model.nq}")

    n = min(sim_qpos.shape[0], ref_qpos.shape[0])
    if max_frames is not None:
        n = min(n, max_frames)
    sim_qpos = sim_qpos[:n]
    ref_qpos = ref_qpos[:n]

    model.vis.global_.offwidth = 720
    model.vis.global_.offheight = 480
    renderer = mujoco.Renderer(model, height=480, width=720)
    ref_data = mujoco.MjData(model)
    sim_data = mujoco.MjData(model)
    camera = build_camera(model, [ref_qpos, sim_qpos])

    frames = []
    for i in range(n):
        ref_img = render_panel(renderer, model, ref_data, ref_qpos[i], camera, "ref")
        sim_img = render_panel(renderer, model, sim_data, sim_qpos[i], camera, "sim")
        frame = np.concatenate([ref_img, sim_img], axis=1)
        for _ in range(repeat):
            frames.append(frame.copy())
    renderer.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=fps)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", type=Path, default=DEFAULT_VARIANTS)
    parser.add_argument("--case-arms", default="", help="Whitespace-separated case or case:arm selectors")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    specs = parse_case_arms(args.case_arms)
    rows = selected_postprocess_rows(read_rows(args.variants), specs)
    if not rows:
        raise SystemExit("No E167 postprocess rows selected")

    for row in rows:
        npz_path = Path(row["postprocess_output_npz"])
        scene_xml = Path(row["rubber_scene_act"])
        ref_npz = Path(row["trajectory"])
        out_path = npz_path.with_name(f"{npz_path.stem}_full.mp4")
        if out_path.is_file() and not args.overwrite:
            print(f"[skip] {out_path}")
            continue
        if not npz_path.is_file():
            raise FileNotFoundError(npz_path)
        if not scene_xml.is_file():
            raise FileNotFoundError(scene_xml)
        if not ref_npz.is_file():
            raise FileNotFoundError(ref_npz)
        print(f"[render] {row['variant']} -> {out_path}", flush=True)
        render_video(
            scene_xml=scene_xml,
            ref_qpos=raw_ref_qpos(ref_npz),
            sim_qpos=qpos_from_npz(npz_path),
            out_path=out_path,
            fps=args.fps,
            repeat=args.repeat,
            max_frames=args.max_frames,
        )
    print(f"E167 postprocess video render complete: rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
