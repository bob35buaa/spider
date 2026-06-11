#!/usr/bin/env python3
"""Render E144 template videos showing real mesh and collision proxy side by side."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO = Path(__file__).resolve().parents[4]
DEFAULT_RUN_ROOT = REPO / "workspace/core4d/results/E144/E144_full_nonbox_raw_contact"
FIELDS = [
    "source_scene_task",
    "object_key",
    "object_category",
    "person",
    "release_decision",
    "proxy_policy",
    "scene_xml",
    "render_status",
    "render_notes",
    "video_path",
    "sheet_path",
]


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        lines = [line for line in f if line.strip() and not line.startswith("#")]
        if not lines:
            return []
        return list(csv.DictReader(lines, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    ):
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def camera_for(model: mujoco.MjModel, data: mujoco.MjData, azimuth: float) -> mujoco.MjvCamera:
    mujoco.mj_forward(model, data)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = azimuth
    cam.elevation = -20
    cam.distance = 3.0
    look = np.array([0.0, 0.0, 0.55], dtype=float)
    obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if obj_id >= 0 and pelvis_id >= 0:
        look = 0.5 * data.xpos[obj_id] + 0.5 * data.xpos[pelvis_id]
        look[2] = max(float(look[2]), 0.55)
    elif obj_id >= 0:
        look = data.xpos[obj_id].copy()
        look[2] = max(float(look[2]), 0.55)
    cam.lookat[:] = look
    return cam


def geom_names(model: mujoco.MjModel) -> list[str]:
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, idx) or "" for idx in range(model.ngeom)]


def configure_rgba(model: mujoco.MjModel, names: list[str], mode: str, base_rgba: np.ndarray) -> None:
    model.geom_rgba[:] = base_rgba
    for idx, name in enumerate(names):
        group = int(model.geom_group[idx])
        is_object_visual = name == "object_visual"
        is_object_collision = name.startswith("object_collision")
        if mode == "mesh":
            if is_object_visual:
                model.geom_rgba[idx] = np.array([0.15, 0.45, 0.85, 1.0])
            elif is_object_collision:
                model.geom_rgba[idx, 3] = 0.0
            elif group == 2:
                model.geom_rgba[idx, 3] = min(float(model.geom_rgba[idx, 3]), 0.22)
            elif group == 3:
                model.geom_rgba[idx, 3] = 0.0
        elif mode == "collision":
            if is_object_visual:
                model.geom_rgba[idx, 3] = 0.03
            elif is_object_collision:
                model.geom_rgba[idx] = np.array([1.0, 0.18, 0.02, 0.72])
            elif group == 2:
                model.geom_rgba[idx, 3] = min(float(model.geom_rgba[idx, 3]), 0.12)
            elif group == 3:
                model.geom_rgba[idx, 3] = 0.0
        elif mode == "overlay":
            if is_object_visual:
                model.geom_rgba[idx] = np.array([0.15, 0.45, 0.85, 0.48])
            elif is_object_collision:
                model.geom_rgba[idx] = np.array([1.0, 0.18, 0.02, 0.55])
            elif group == 2:
                model.geom_rgba[idx, 3] = min(float(model.geom_rgba[idx, 3]), 0.18)
            elif group == 3:
                model.geom_rgba[idx, 3] = 0.0
        else:
            raise ValueError(f"unknown mode: {mode}")


def label_image(image: np.ndarray, title: str, font: ImageFont.ImageFont) -> Image.Image:
    im = Image.fromarray(image)
    label_h = 34
    out = Image.new("RGB", (im.width, im.height + label_h), "white")
    out.paste(im, (0, label_h))
    draw = ImageDraw.Draw(out)
    draw.rectangle([0, 0, im.width, label_h], fill=(245, 245, 245))
    draw.text((8, 6), title, fill=(0, 0, 0), font=font)
    return out


def render_frame(
    renderer: mujoco.Renderer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    names: list[str],
    base_rgba: np.ndarray,
    azimuth: float,
    font: ImageFont.ImageFont,
) -> np.ndarray:
    panels = []
    for mode, title in (
        ("mesh", "real mesh"),
        ("collision", "collision proxy"),
        ("overlay", "mesh + collision"),
    ):
        mujoco.mj_resetData(model, data)
        configure_rgba(model, names, mode, base_rgba)
        mujoco.mj_forward(model, data)
        if hasattr(renderer, "_scene_option"):
            renderer._scene_option.geomgroup[:] = 1
        renderer.update_scene(data, camera=camera_for(model, data, azimuth))
        panels.append(label_image(renderer.render(), title, font))
    gap = 8
    width = sum(panel.width for panel in panels) + gap * (len(panels) - 1)
    height = max(panel.height for panel in panels)
    out = Image.new("RGB", (width, height), "white")
    x = 0
    for panel in panels:
        out.paste(panel, (x, 0))
        x += panel.width + gap
    return np.asarray(out)


def render_task(row: dict[str, str], out_root: Path, width: int, height: int, frames: int, fps: int, overwrite: bool) -> dict[str, str]:
    task = row["source_scene_task"]
    scene = Path(row["scene_xml"])
    case_dir = out_root / "templates" / task
    video_path = case_dir / f"{task}_mesh_collision_review.mp4"
    sheet_path = case_dir / f"{task}_mesh_collision_review_sheet.png"
    base = {
        "source_scene_task": task,
        "object_key": row.get("object_key", ""),
        "object_category": row.get("object_category", ""),
        "person": row.get("person", ""),
        "release_decision": row.get("release_decision", row.get("review_decision", "")),
        "proxy_policy": row.get("draft_policy", row.get("approved_collision_policy", "")),
        "scene_xml": str(scene),
    }
    if video_path.is_file() and sheet_path.is_file() and not overwrite:
        return {**base, "render_status": "pass", "render_notes": "existing output reused", "video_path": str(video_path), "sheet_path": str(sheet_path)}
    if not scene.is_file():
        return {**base, "render_status": "not_rendered", "render_notes": "missing scene_xml", "video_path": "", "sheet_path": ""}
    try:
        case_dir.mkdir(parents=True, exist_ok=True)
        model = mujoco.MjModel.from_xml_path(str(scene))
        data = mujoco.MjData(model)
        names = geom_names(model)
        base_rgba = model.geom_rgba.copy()
        font = load_font(18)
        rendered = []
        key_images = []
        frame_count = max(int(frames), 1)
        azimuths = np.linspace(35.0, 395.0, frame_count, endpoint=False)
        key_slots = {0, frame_count // 4, frame_count // 2, (3 * frame_count) // 4}
        with mujoco.Renderer(model, height=height, width=width) as renderer:
            for idx, azimuth in enumerate(azimuths):
                image = render_frame(renderer, model, data, names, base_rgba, float(azimuth), font)
                rendered.append(image)
                if idx in key_slots:
                    key_images.append((azimuth, image))
        imageio.mimsave(video_path, rendered, fps=fps)
        tile_w, tile_h = rendered[0].shape[1], rendered[0].shape[0]
        cols = 2
        rows = int(np.ceil(len(key_images) / cols))
        label_h = 38
        title_h = 46
        sheet = Image.new("RGB", (cols * tile_w, title_h + rows * (tile_h + label_h)), "white")
        draw = ImageDraw.Draw(sheet)
        title_font = load_font(24)
        small_font = load_font(18)
        draw.text((10, 10), f"{task} | {base['object_category']} | {base['release_decision']} | {base['proxy_policy']}", fill=(0, 0, 0), font=title_font)
        for idx, (azimuth, image) in enumerate(key_images):
            rr, cc = divmod(idx, cols)
            x = cc * tile_w
            y = title_h + rr * (tile_h + label_h)
            sheet.paste(Image.fromarray(image), (x, y))
            draw.text((x + 8, y + tile_h + 8), f"az={azimuth:.0f}", fill=(0, 0, 0), font=small_font)
        sheet.save(sheet_path)
        return {**base, "render_status": "pass", "render_notes": "rendered mesh/collision comparison", "video_path": str(video_path), "sheet_path": str(sheet_path)}
    except Exception as exc:  # noqa: BLE001
        return {**base, "render_status": "render_error", "render_notes": f"{type(exc).__name__}: {exc}", "video_path": "", "sheet_path": ""}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--width", type=int, default=360)
    parser.add_argument("--height", type=int, default=300)
    parser.add_argument("--frames", type=int, default=18)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_root = args.run_root.expanduser().resolve()
    out_dir = args.out_dir or run_root / "s2_templates_mesh_collision_review"
    out_dir = out_dir.expanduser().resolve()
    draft_rows = read_tsv(run_root / "s2_templates/draft_proxy/draft_proxy_review_queue.tsv")
    released_rows = [
        {
            "source_scene_task": row["source_scene_task"],
            "object_key": row.get("object_key", ""),
            "object_category": row.get("object_category", ""),
            "person": row.get("person", ""),
            "release_decision": row.get("review_decision", ""),
            "draft_policy": row.get("approved_collision_policy", ""),
            "scene_xml": row.get("proxy_scene_xml", ""),
        }
        for row in read_tsv(run_root / "s2_templates/nonbox_template_review.tsv")
        if row.get("review_decision") == "approve_clean"
    ]
    rows = sorted([*draft_rows, *released_rows], key=lambda row: (row.get("object_category", ""), row.get("object_key", ""), row.get("person", "")))
    rendered = [render_task(row, out_dir, args.width, args.height, args.frames, args.fps, args.overwrite) for row in rows]
    write_tsv(out_dir / "mesh_collision_review_manifest.tsv", rendered, FIELDS)
    write_json(out_dir / "mesh_collision_review_manifest.json", rendered)
    summary = {
        "rows": len(rendered),
        "render_status_counts": dict(Counter(row["render_status"] for row in rendered)),
        "release_decision_counts": dict(Counter(row["release_decision"] for row in rendered)),
        "object_category_counts": dict(Counter(row["object_category"] for row in rendered)),
        "out_dir": str(out_dir),
    }
    write_json(out_dir / "mesh_collision_review_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
