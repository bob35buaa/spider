#!/usr/bin/env python3
"""Render S2 source-template review sheets/videos."""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path
import sys

SCRIPT_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "data_construction_v3")
for _path in (SCRIPT_ROOT / "lib", SCRIPT_ROOT / "state", SCRIPT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from common import SCHEMA_VERSION, json_dumps, read_tsv, timestamp, write_json, write_tsv


FIELDS = [
    "stage",
    "source_scene_task",
    "object_key",
    "object_category",
    "person",
    "template_status",
    "recommended_action",
    "render_status",
    "render_notes",
    "video_path",
    "sheet_path",
    "scene_xml",
    "schema_version",
    "updated_at",
]


def wanted_statuses(text: str) -> set[str]:
    return {item.strip() for item in text.split(",") if item.strip()}


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
    elif model.nbody > 1:
        look = data.xpos[1].copy()
        look[2] = max(float(look[2]), 0.55)
    cam.lookat[:] = look
    return cam


def render_template(
    row: dict[str, str],
    out_dir: Path,
    width: int,
    height: int,
    fps: int,
    frames: int,
    overwrite: bool,
) -> dict[str, Any]:
    task = row.get("source_scene_task") or row.get("task", "")
    scene = Path(row.get("scene_xml", "")).expanduser()
    case_dir = out_dir / "templates" / task
    case_dir.mkdir(parents=True, exist_ok=True)
    video = case_dir / f"{task}_source_template_orbit.mp4"
    sheet = case_dir / f"{task}_source_template_sheet.png"
    base = {
        "stage": "S2_template_visual_review",
        "source_scene_task": task,
        "object_key": row.get("object_key", ""),
        "object_category": row.get("object_category", ""),
        "person": row.get("person", ""),
        "template_status": row.get("template_status", ""),
        "recommended_action": row.get("recommended_action", ""),
        "scene_xml": str(scene),
        "schema_version": SCHEMA_VERSION,
        "updated_at": timestamp(),
    }
    if not scene.is_file():
        return {
            **base,
            "render_status": "not_rendered",
            "render_notes": "missing scene_xml",
            "video_path": "",
            "sheet_path": "",
        }

    try:
        if video.is_file() and sheet.is_file() and not overwrite:
            return {
                **base,
                "render_status": "pass",
                "render_notes": "existing visual package reused",
                "video_path": str(video),
                "sheet_path": str(sheet),
            }
        model = mujoco.MjModel.from_xml_path(str(scene))
        data = mujoco.MjData(model)
        frame_count = max(int(frames), 1)
        azimuths = np.linspace(35.0, 395.0, frame_count, endpoint=False)
        rendered: list[np.ndarray] = []
        key_images: list[tuple[str, np.ndarray]] = []
        key_slots = {0, frame_count // 4, frame_count // 2, (3 * frame_count) // 4}
        with mujoco.Renderer(model, height=height, width=width) as renderer:
            for idx, azimuth in enumerate(azimuths):
                mujoco.mj_resetData(model, data)
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera_for(model, data, float(azimuth)))
                image = renderer.render()
                rendered.append(image)
                if idx in key_slots:
                    key_images.append((f"az={azimuth:.0f}", image))
        imageio.mimsave(video, rendered, fps=fps)

        cols = 2
        rows_n = int(np.ceil(len(key_images) / cols))
        fig, axes = plt.subplots(rows_n, cols, figsize=(10, 4.5 * rows_n), squeeze=False, constrained_layout=True)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, (label, image) in zip(axes.ravel(), key_images):
            ax.imshow(image)
            ax.set_title(label)
        fig.suptitle(f"{task} | {row.get('template_status', '')} | {row.get('recommended_action', '')}")
        fig.savefig(sheet, dpi=150)
        plt.close(fig)
        return {
            **base,
            "render_status": "pass",
            "render_notes": "rendered",
            "video_path": str(video),
            "sheet_path": str(sheet),
        }
    except Exception as exc:  # noqa: BLE001 - keep per-template render evidence.
        return {
            **base,
            "render_status": "render_error",
            "render_notes": f"{type(exc).__name__}: {exc}",
            "video_path": "",
            "sheet_path": "",
        }


def markdown_summary(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# S2 source-template visual review",
        "",
        f"- rows: `{summary['rows']}`",
        f"- rendered rows: `{summary['rendered_rows']}`",
        "",
        "## render status",
        "",
        "| status | count |",
        "|---|---:|",
    ]
    for status, count in summary["render_status_counts"].items():
        lines.append(f"| `{status}` | {count} |")
    lines.extend(["", "## templates", "", "| task | template status | render status | sheet | video | notes |", "|---|---|---|---|---|---|"])
    for row in rows[:100]:
        sheet = Path(str(row.get("sheet_path", ""))).name if row.get("sheet_path") else ""
        video = Path(str(row.get("video_path", ""))).name if row.get("video_path") else ""
        lines.append(
            f"| `{row['source_scene_task']}` | `{row['template_status']}` | `{row['render_status']}` | `{sheet}` | `{video}` | `{row['render_notes']}` |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template-backlog-tsv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--render-statuses", default="clean,audit_fail,manual_review_required", help="comma-separated template_status values to render")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    statuses = wanted_statuses(args.render_statuses)
    out_dir = args.out_dir.resolve()
    rows: list[dict[str, Any]] = []
    for row in read_tsv(args.template_backlog_tsv):
        if row.get("template_status", "") in statuses:
            rows.append(render_template(row, out_dir, args.width, args.height, args.fps, args.frames, args.overwrite))
        else:
            rows.append(
                {
                    "stage": "S2_template_visual_review",
                    "source_scene_task": row.get("source_scene_task", ""),
                    "object_key": row.get("object_key", ""),
                    "object_category": row.get("object_category", ""),
                    "person": row.get("person", ""),
                    "template_status": row.get("template_status", ""),
                    "recommended_action": row.get("recommended_action", ""),
                    "render_status": "skipped",
                    "render_notes": "template_status not selected for rendering",
                    "video_path": "",
                    "sheet_path": "",
                    "scene_xml": row.get("scene_xml", ""),
                    "schema_version": SCHEMA_VERSION,
                    "updated_at": timestamp(),
                }
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(out_dir / "template_visual_manifest.tsv", rows, FIELDS)
    write_json(out_dir / "template_visual_manifest.json", rows)
    summary = {
        "stage": "S2_template_visual_review",
        "created_at": timestamp(),
        "schema_version": SCHEMA_VERSION,
        "rows": len(rows),
        "rendered_rows": sum(1 for row in rows if row["render_status"] == "pass"),
        "render_status_counts": dict(Counter(row["render_status"] for row in rows)),
        "out_dir": str(out_dir),
    }
    write_json(out_dir / "template_visual_summary.json", summary)
    (out_dir / "template_visual_summary.md").write_text(markdown_summary(summary, rows), encoding="utf-8")
    print(json_dumps(summary))


if __name__ == "__main__":
    main()
