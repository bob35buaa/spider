#!/usr/bin/env python3
"""Render S4 target-gate replay sheets/videos for visual QC review."""

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
    "case_id",
    "retarget_variant_id",
    "target_variant_id",
    "target_task",
    "target_gate_status",
    "render_status",
    "render_notes",
    "frames",
    "rendered_frames",
    "video_path",
    "sheet_path",
    "target_scene",
    "scene_act",
    "trajectory",
    "schema_version",
    "updated_at",
]


def wanted_statuses(text: str) -> set[str]:
    return {item.strip() for item in text.split(",") if item.strip()}


def camera_for(model: mujoco.MjModel, data: mujoco.MjData) -> mujoco.MjvCamera:
    mujoco.mj_forward(model, data)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = 55
    cam.elevation = -18
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


def qpos_from_traj(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    if "qpos" not in data:
        raise KeyError(f"trajectory missing qpos: {path}")
    qpos = np.asarray(data["qpos"], dtype=np.float64)
    if qpos.ndim != 2:
        raise ValueError(f"qpos must be 2D, got {qpos.shape}: {path}")
    return qpos


def render_case(
    row: dict[str, str],
    out_dir: Path,
    width: int,
    height: int,
    fps: int,
    max_video_frames: int,
    overwrite: bool,
) -> dict[str, Any]:
    case_id = row.get("case_id", "")
    scene = Path(row.get("target_scene", "")).expanduser()
    trajectory = Path(row.get("trajectory", "")).expanduser()
    case_dir = out_dir / "cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    video = case_dir / f"{case_id}_target_replay.mp4"
    sheet = case_dir / f"{case_id}_target_replay_sheet.png"
    base = {
        "stage": "S4_visual_qc_render",
        "case_id": case_id,
        "retarget_variant_id": row.get("retarget_variant_id", ""),
        "target_variant_id": row.get("target_variant_id", "ref_fk"),
        "target_task": row.get("target_task", ""),
        "target_gate_status": row.get("target_gate_status", ""),
        "target_scene": str(scene),
        "scene_act": row.get("scene_act", ""),
        "trajectory": str(trajectory),
        "schema_version": SCHEMA_VERSION,
        "updated_at": timestamp(),
    }
    if not scene.is_file() or not trajectory.is_file():
        return {
            **base,
            "render_status": "not_rendered",
            "render_notes": "missing target_scene or trajectory",
            "frames": "",
            "rendered_frames": "",
            "video_path": "",
            "sheet_path": "",
        }

    try:
        qpos = qpos_from_traj(trajectory)
        model = mujoco.MjModel.from_xml_path(str(scene))
        if qpos.shape[1] != model.nq:
            raise ValueError(f"qpos width {qpos.shape[1]} != model.nq {model.nq}")
        if video.is_file() and sheet.is_file() and not overwrite:
            return {
                **base,
                "render_status": "pass",
                "render_notes": "existing visual package reused",
                "frames": str(qpos.shape[0]),
                "rendered_frames": "",
                "video_path": str(video),
                "sheet_path": str(sheet),
            }

        data = mujoco.MjData(model)
        key_indices = sorted(set(np.linspace(0, qpos.shape[0] - 1, num=min(8, qpos.shape[0]), dtype=int).tolist()))
        video_indices = sorted(set(np.linspace(0, qpos.shape[0] - 1, num=min(max_video_frames, qpos.shape[0]), dtype=int).tolist()))
        key_images: list[tuple[int, np.ndarray]] = []
        frames: list[np.ndarray] = []
        with mujoco.Renderer(model, height=height, width=width) as renderer:
            for idx in sorted(set(key_indices + video_indices)):
                data.qpos[:] = qpos[idx]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera_for(model, data))
                image = renderer.render()
                if idx in video_indices:
                    frames.append(image)
                if idx in key_indices:
                    key_images.append((idx, image))
        if frames:
            imageio.mimsave(video, frames, fps=fps)

        cols = 4
        rows_n = int(np.ceil(len(key_images) / cols))
        fig, axes = plt.subplots(rows_n, cols, figsize=(16, 4 * rows_n), squeeze=False, constrained_layout=True)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, (idx, image) in zip(axes.ravel(), key_images):
            ax.imshow(image)
            ax.set_title(f"f{idx:04d}")
        fig.suptitle(f"{case_id} | {row.get('retarget_variant_id', '')}/{row.get('target_variant_id', '')}")
        fig.savefig(sheet, dpi=150)
        plt.close(fig)
        return {
            **base,
            "render_status": "pass",
            "render_notes": "rendered",
            "frames": str(qpos.shape[0]),
            "rendered_frames": str(len(frames)),
            "video_path": str(video),
            "sheet_path": str(sheet),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            **base,
            "render_status": "render_error",
            "render_notes": f"{type(exc).__name__}: {exc}",
            "frames": "",
            "rendered_frames": "",
            "video_path": "",
            "sheet_path": "",
        }


def markdown_summary(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# S4 visual QC render package",
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
    lines.extend(["", "## package rows", "", "| case | status | sheet | video | notes |", "|---|---|---|---|---|"])
    for row in rows[:100]:
        sheet = Path(str(row.get("sheet_path", ""))).name if row.get("sheet_path") else ""
        video = Path(str(row.get("video_path", ""))).name if row.get("video_path") else ""
        lines.append(f"| `{row['case_id']}` | `{row['render_status']}` | `{sheet}` | `{video}` | `{row['render_notes']}` |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-gate-manifest-tsv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--render-statuses", default="pass", help="comma-separated target_gate_status values to render")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--max-video-frames", type=int, default=240)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    statuses = wanted_statuses(args.render_statuses)
    out_dir = args.out_dir.resolve()
    rows: list[dict[str, Any]] = []
    for row in read_tsv(args.target_gate_manifest_tsv):
        if row.get("target_gate_status", "") in statuses:
            rows.append(render_case(row, out_dir, args.width, args.height, args.fps, args.max_video_frames, args.overwrite))
        else:
            rows.append(
                {
                    "stage": "S4_visual_qc_render",
                    "case_id": row.get("case_id", ""),
                    "retarget_variant_id": row.get("retarget_variant_id", ""),
                    "target_variant_id": row.get("target_variant_id", "ref_fk"),
                    "target_task": row.get("target_task", ""),
                    "target_gate_status": row.get("target_gate_status", ""),
                    "render_status": "skipped",
                    "render_notes": "target_gate_status not selected for rendering",
                    "frames": "",
                    "rendered_frames": "",
                    "video_path": "",
                    "sheet_path": "",
                    "target_scene": row.get("target_scene", ""),
                    "scene_act": row.get("scene_act", ""),
                    "trajectory": row.get("trajectory", ""),
                    "schema_version": SCHEMA_VERSION,
                    "updated_at": timestamp(),
                }
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(out_dir / "visual_qc_render_manifest.tsv", rows, FIELDS)
    write_json(out_dir / "visual_qc_render_manifest.json", rows)
    summary = {
        "stage": "S4_visual_qc_render",
        "created_at": timestamp(),
        "schema_version": SCHEMA_VERSION,
        "rows": len(rows),
        "rendered_rows": sum(1 for row in rows if row["render_status"] == "pass"),
        "render_status_counts": dict(Counter(row["render_status"] for row in rows)),
        "out_dir": str(out_dir),
    }
    write_json(out_dir / "visual_qc_render_summary.json", summary)
    (out_dir / "visual_qc_render_summary.md").write_text(markdown_summary(summary, rows), encoding="utf-8")
    print(json_dumps(summary))


if __name__ == "__main__":
    main()
