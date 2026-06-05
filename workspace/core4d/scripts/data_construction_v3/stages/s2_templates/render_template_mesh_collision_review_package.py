#!/usr/bin/env python3
"""Render source-template mesh/collision overlay review package."""

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


FIELDS = [
    "source_scene_task",
    "object_key",
    "object_category",
    "person",
    "template_status",
    "recommended_action",
    "proxy_policy",
    "scene_xml",
    "render_status",
    "render_notes",
    "video_path",
    "sheet_path",
]


def read_tsv(path: Path) -> list[dict[str, str]]:
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


def geom_names(model: mujoco.MjModel) -> list[str]:
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, idx) or "" for idx in range(model.ngeom)]


def camera_for(model: mujoco.MjModel, data: mujoco.MjData, azimuth: float, object_only: bool) -> mujoco.MjvCamera:
    mujoco.mj_forward(model, data)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = azimuth
    cam.elevation = -18 if object_only else -20
    obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if obj_id >= 0:
        look = data.xpos[obj_id].copy()
    else:
        look = np.array([0.0, 0.0, 0.55], dtype=float)
    if not object_only and obj_id >= 0 and pelvis_id >= 0:
        look = 0.5 * data.xpos[obj_id] + 0.5 * data.xpos[pelvis_id]
        look[2] = max(float(look[2]), 0.55)
    cam.lookat[:] = look
    cam.distance = 1.35 if object_only else 3.0
    return cam


def configure_rgba(model: mujoco.MjModel, names: list[str], mode: str, base_rgba: np.ndarray, object_only: bool) -> None:
    model.geom_rgba[:] = base_rgba
    if object_only:
        model.geom_rgba[:, 3] = 0.0
    for idx, name in enumerate(names):
        group = int(model.geom_group[idx])
        is_object_visual = name == "object_visual"
        is_object_collision = name.startswith("object_collision")
        if mode == "mesh":
            if is_object_visual:
                model.geom_rgba[idx] = np.array([0.12, 0.35, 0.95, 1.0])
            elif is_object_collision:
                model.geom_rgba[idx, 3] = 0.0
            elif not object_only and group == 2:
                model.geom_rgba[idx, 3] = min(float(model.geom_rgba[idx, 3]), 0.22)
            elif group == 3:
                model.geom_rgba[idx, 3] = 0.0
        elif mode == "collision":
            if is_object_visual:
                model.geom_rgba[idx, 3] = 0.06 if object_only else 0.03
            elif is_object_collision:
                model.geom_rgba[idx] = np.array([1.0, 0.12, 0.02, 0.68 if object_only else 0.72])
            elif not object_only and group == 2:
                model.geom_rgba[idx, 3] = min(float(model.geom_rgba[idx, 3]), 0.12)
            elif group == 3:
                model.geom_rgba[idx, 3] = 0.0
        elif mode == "overlay":
            if is_object_visual:
                model.geom_rgba[idx] = np.array([0.12, 0.35, 0.95, 0.72 if object_only else 0.48])
            elif is_object_collision:
                model.geom_rgba[idx] = np.array([1.0, 0.12, 0.02, 0.44 if object_only else 0.55])
            elif not object_only and group == 2:
                model.geom_rgba[idx, 3] = min(float(model.geom_rgba[idx, 3]), 0.18)
            elif group == 3:
                model.geom_rgba[idx, 3] = 0.0
        else:
            raise ValueError(f"unknown mode: {mode}")


def label_image(image: np.ndarray, title: str, font: ImageFont.ImageFont) -> Image.Image:
    im = Image.fromarray(image)
    label_h = 30
    out = Image.new("RGB", (im.width, im.height + label_h), "white")
    out.paste(im, (0, label_h))
    draw = ImageDraw.Draw(out)
    draw.rectangle([0, 0, im.width, label_h], fill=(245, 245, 245))
    draw.text((8, 5), title, fill=(0, 0, 0), font=font)
    return out


def render_frame(
    renderer: mujoco.Renderer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    names: list[str],
    base_rgba: np.ndarray,
    azimuth: float,
    font: ImageFont.ImageFont,
    object_only: bool,
) -> np.ndarray:
    panels = []
    for mode, title in (("mesh", "real mesh"), ("collision", "collision proxy"), ("overlay", "mesh + collision")):
        mujoco.mj_resetData(model, data)
        configure_rgba(model, names, mode, base_rgba, object_only)
        mujoco.mj_forward(model, data)
        if hasattr(renderer, "_scene_option"):
            renderer._scene_option.geomgroup[:] = 1
        renderer.update_scene(data, camera=camera_for(model, data, azimuth, object_only))
        panels.append(label_image(renderer.render(), title, font))
    gap = 8
    out = Image.new("RGB", (sum(panel.width for panel in panels) + gap * 2, max(panel.height for panel in panels)), "white")
    x = 0
    for panel in panels:
        out.paste(panel, (x, 0))
        x += panel.width + gap
    return np.asarray(out)


def normalize_row(row: dict[str, str]) -> dict[str, str]:
    task = row.get("source_scene_task") or row.get("task") or Path(row.get("scene_xml", "")).parent.name
    return {
        "source_scene_task": task,
        "object_key": row.get("object_key", ""),
        "object_category": row.get("object_category", ""),
        "person": row.get("person", ""),
        "template_status": row.get("template_status", row.get("draft_status", "")),
        "recommended_action": row.get("recommended_action", row.get("review_recommendation", "")),
        "proxy_policy": row.get("collision_policy", row.get("draft_policy", row.get("approved_collision_policy", ""))),
        "scene_xml": row.get("scene_xml", row.get("proxy_scene_xml", "")),
    }


def render_task(row: dict[str, str], out_root: Path, width: int, height: int, frames: int, fps: int, overwrite: bool, object_only: bool) -> dict[str, str]:
    base = normalize_row(row)
    task = base["source_scene_task"]
    scene = Path(base["scene_xml"])
    case_dir = out_root / "templates" / task
    suffix = "object_only_mesh_collision" if object_only else "mesh_collision"
    video_path = case_dir / f"{task}_{suffix}_review.mp4"
    sheet_path = case_dir / f"{task}_{suffix}_review_sheet.png"
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
        font = load_font(16)
        rendered = []
        key_images = []
        frame_count = max(int(frames), 1)
        azimuths = np.linspace(35.0, 395.0, frame_count, endpoint=False)
        key_slots = {0, frame_count // 4, frame_count // 2, (3 * frame_count) // 4}
        with mujoco.Renderer(model, height=height, width=width) as renderer:
            for idx, azimuth in enumerate(azimuths):
                image = render_frame(renderer, model, data, names, base_rgba, float(azimuth), font, object_only)
                rendered.append(image)
                if idx in key_slots:
                    key_images.append((azimuth, image))
        imageio.mimsave(video_path, rendered, fps=fps)
        tile_w, tile_h = rendered[0].shape[1], rendered[0].shape[0]
        cols = 2
        rows = int(np.ceil(len(key_images) / cols))
        label_h = 34
        title_h = 44
        sheet = Image.new("RGB", (cols * tile_w, title_h + rows * (tile_h + label_h)), "white")
        draw = ImageDraw.Draw(sheet)
        title_font = load_font(22)
        small_font = load_font(16)
        draw.text(
            (10, 10),
            f"{task} | {base['object_category']} | {base['template_status']} | {base['proxy_policy']}",
            fill=(0, 0, 0),
            font=title_font,
        )
        for idx, (azimuth, image) in enumerate(key_images):
            rr, cc = divmod(idx, cols)
            x = cc * tile_w
            y = title_h + rr * (tile_h + label_h)
            sheet.paste(Image.fromarray(image), (x, y))
            draw.text((x + 8, y + tile_h + 7), f"az={azimuth:.0f}", fill=(0, 0, 0), font=small_font)
        sheet.save(sheet_path)
        return {**base, "render_status": "pass", "render_notes": "rendered mesh/collision comparison", "video_path": str(video_path), "sheet_path": str(sheet_path)}
    except Exception as exc:  # noqa: BLE001
        return {**base, "render_status": "render_error", "render_notes": f"{type(exc).__name__}: {exc}", "video_path": "", "sheet_path": ""}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-tsv", type=Path, required=True, help="template_backlog.tsv or another TSV with source_scene_task and scene_xml")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--render-statuses", default="", help="optional comma-separated template_status filter")
    parser.add_argument("--object-only", action="store_true", help="hide robot and frame object mesh/collision only")
    parser.add_argument("--width", type=int, default=360)
    parser.add_argument("--height", type=int, default=300)
    parser.add_argument("--frames", type=int, default=18)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rows = read_tsv(args.input_tsv)
    statuses = {item.strip() for item in args.render_statuses.split(",") if item.strip()}
    if statuses:
        rows = [row for row in rows if normalize_row(row)["template_status"] in statuses]
    rendered = [
        render_task(row, args.out_dir, args.width, args.height, args.frames, args.fps, args.overwrite, args.object_only)
        for row in rows
    ]
    write_tsv(args.out_dir / "template_mesh_collision_review_manifest.tsv", rendered, FIELDS)
    write_json(args.out_dir / "template_mesh_collision_review_manifest.json", rendered)
    summary = {
        "rows": len(rendered),
        "render_status_counts": dict(Counter(row["render_status"] for row in rendered)),
        "template_status_counts": dict(Counter(row["template_status"] for row in rendered)),
        "object_category_counts": dict(Counter(row["object_category"] for row in rendered)),
        "out_dir": str(args.out_dir),
        "object_only": args.object_only,
    }
    write_json(args.out_dir / "template_mesh_collision_review_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
