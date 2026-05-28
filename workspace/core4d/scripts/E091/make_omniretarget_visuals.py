#!/usr/bin/env python3
"""Visualize E091 OmniRetarget retargeted and trimmed NPZ outputs."""

from __future__ import annotations

import argparse
import csv
import json
import os
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


DEFAULT_V2_ROOT = Path("/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2")
DEFAULT_SPIDER_REPO = Path("/home/ubuntu/Workspace/spider")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def image_nonblank(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 2048:
        return False
    try:
        arr = plt.imread(path)
    except Exception:
        return False
    return bool(np.asarray(arr).std() > 1e-4)


def task_info(spider_repo: Path, task: str) -> dict[str, Any]:
    path = spider_repo / "example_datasets/processed/core4d/unitree_g1/humanoid_object" / task / "task_info.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def qpos_path(stage_dir: Path, stage: str) -> Path | None:
    files = sorted((stage_dir / stage).glob("*.npz"))
    return files[0] if files else None


def load_qpos(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    data = np.load(path, allow_pickle=True)
    meta = {key: data[key].tolist() if np.asarray(data[key]).shape == () else str(np.asarray(data[key]).shape) for key in data.files if key != "qpos"}
    return np.asarray(data["qpos"], dtype=np.float64), meta


def trim_window(stage_dir: Path) -> dict[str, Any]:
    path = stage_dir / "trim_window.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def render_frames(scene_xml: Path, qpos: np.ndarray, frame_indices: np.ndarray, cameras: list[str]) -> list[np.ndarray]:
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=640, height=420)
    frames: list[np.ndarray] = []
    for idx in frame_indices:
        data.qpos[:] = qpos[int(idx)]
        mujoco.mj_forward(model, data)
        views = []
        for camera in cameras:
            renderer.update_scene(data, camera=camera)
            views.append(renderer.render().copy())
        frames.append(np.concatenate(views, axis=1) if len(views) > 1 else views[0])
    renderer.close()
    return frames


def make_keyframes(scene_xml: Path, qpos: np.ndarray, out_path: Path, title: str) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    idxs = np.linspace(0, len(qpos) - 1, num=min(6, len(qpos)), dtype=int)
    try:
        frames = render_frames(scene_xml, qpos, idxs, ["track", "track2"])
    except Exception as exc:
        return f"render_failed: {exc}"
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True)
    for ax, img, idx in zip(axes.flat, frames, idxs):
        ax.imshow(img)
        ax.set_title(f"frame {int(idx)}")
        ax.axis("off")
    for ax in axes.flat[len(frames) :]:
        ax.axis("off")
    fig.suptitle(title)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return "ok"


def make_video(scene_xml: Path, qpos: np.ndarray, out_path: Path, fps: int) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stride = max(1, int(round(fps / 15)))
    idxs = np.arange(0, len(qpos), stride, dtype=int)
    try:
        frames = render_frames(scene_xml, qpos, idxs, ["track", "track2"])
        with imageio.get_writer(out_path, fps=min(15, fps), codec="libx264", quality=8) as writer:
            for frame in frames:
                writer.append_data(frame)
    except Exception as exc:
        return f"video_failed: {exc}"
    return "ok"


def make_timeline(
    task: str,
    retarget_qpos: np.ndarray,
    trimmed_qpos: np.ndarray | None,
    window: dict[str, Any],
    out_path: Path,
) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.arange(len(retarget_qpos))
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    axes[0].plot(frames, retarget_qpos[:, 2], label="retarget pelvis z", color="#333333")
    axes[0].plot(frames, retarget_qpos[:, 38], label="retarget object z", color="#4477aa")
    axes[0].set_ylabel("z (m)")
    axes[0].legend(loc="upper right")
    axes[0].set_title("Retargeted height signals")

    root_xy = np.linalg.norm(retarget_qpos[:, :2] - retarget_qpos[0, :2], axis=1)
    obj_xy = np.linalg.norm(retarget_qpos[:, 36:38] - retarget_qpos[0, 36:38], axis=1)
    axes[1].plot(frames, root_xy, label="root xy displacement", color="#cc3311")
    axes[1].plot(frames, obj_xy, label="object xy displacement", color="#228833")
    axes[1].set_ylabel("displacement (m)")
    axes[1].legend(loc="upper right")
    axes[1].set_title("XY displacement")

    if trimmed_qpos is not None:
        trimmed_frames = np.arange(len(trimmed_qpos))
        axes[2].plot(trimmed_frames, trimmed_qpos[:, 2], label="trimmed pelvis z", color="#333333")
        axes[2].plot(trimmed_frames, trimmed_qpos[:, 38], label="trimmed object z", color="#4477aa")
        axes[2].set_title("Trimmed segment height signals")
        axes[2].legend(loc="upper right")
    else:
        axes[2].text(0.5, 0.5, "No trimmed NPZ", ha="center", va="center", transform=axes[2].transAxes)

    start = int(window.get("start", window.get("trim_start", -1)) or -1)
    count = int(window.get("frames", window.get("trim_frames", 0)) or 0)
    if start >= 0 and count > 0:
        for ax in axes[:2]:
            ax.axvspan(start, start + count, color="#999999", alpha=0.18, label="trim window")
    axes[-1].set_xlabel("frame")
    fig.suptitle(f"{task} OmniRetarget timeline")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return "ok"


def visualize_case(stage_dir: Path, spider_repo: Path, v2_root: Path) -> dict[str, Any]:
    task = stage_dir.name.replace("holosoma_", "", 1)
    info = task_info(spider_repo, task)
    scene_rel = info.get("source_scene")
    source_scene = spider_repo / scene_rel if scene_rel else None
    retarget = qpos_path(stage_dir, "retargeted")
    trimmed = qpos_path(stage_dir, "trimmed")
    out_dir = v2_root / "visualizations/omniretarget"
    row: dict[str, Any] = {
        "task": task,
        "object_name": info.get("object_name", ""),
        "sequence": f"{info.get('date', '')}/{info.get('seq', '')}".strip("/"),
        "person": info.get("person", ""),
        "source_scene": str(source_scene) if source_scene else "",
        "retargeted_npz": str(retarget) if retarget else "",
        "trimmed_npz": str(trimmed) if trimmed else "",
        "retargeted_frames": 0,
        "trimmed_frames": 0,
        "retargeted_keyframes_png": "",
        "trimmed_keyframes_png": "",
        "timeline_png": "",
        "retargeted_mp4": "",
        "status": "",
        "render_status": "",
    }
    if retarget is None:
        row["status"] = "missing_retargeted_npz"
        return row
    if source_scene is None or not source_scene.is_file():
        row["status"] = "missing_source_scene"
        return row

    retarget_qpos, meta = load_qpos(retarget)
    trimmed_qpos = None
    if trimmed is not None:
        trimmed_qpos, _ = load_qpos(trimmed)
    fps = int(meta.get("fps", 30) or 30)
    row["retargeted_frames"] = int(len(retarget_qpos))
    row["trimmed_frames"] = int(len(trimmed_qpos)) if trimmed_qpos is not None else 0
    row["cost"] = meta.get("cost", "")

    retarget_keyframes = out_dir / f"{task}_retargeted_keyframes.png"
    timeline = out_dir / f"{task}_omniretarget_timeline.png"
    video = out_dir / f"{task}_retargeted.mp4"
    statuses = [
        make_keyframes(source_scene, retarget_qpos, retarget_keyframes, f"{task} OmniRetarget retargeted"),
        make_timeline(task, retarget_qpos, trimmed_qpos, trim_window(stage_dir), timeline),
        make_video(source_scene, retarget_qpos, video, fps=fps),
    ]
    row["retargeted_keyframes_png"] = str(retarget_keyframes)
    row["timeline_png"] = str(timeline)
    row["retargeted_mp4"] = str(video)
    if trimmed_qpos is not None:
        trimmed_keyframes = out_dir / f"{task}_trimmed_keyframes.png"
        statuses.append(make_keyframes(source_scene, trimmed_qpos, trimmed_keyframes, f"{task} OmniRetarget trimmed"))
        row["trimmed_keyframes_png"] = str(trimmed_keyframes)
    row["render_status"] = ",".join(statuses)
    row["status"] = "ok" if all(s == "ok" for s in statuses) else "partial"
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-root", type=Path, default=DEFAULT_V2_ROOT)
    parser.add_argument("--spider-repo", type=Path, default=DEFAULT_SPIDER_REPO)
    args = parser.parse_args()

    stage_root = args.v2_root / "results/stage2b_medium/results"
    rows = [
        visualize_case(path, args.spider_repo, args.v2_root)
        for path in sorted(stage_root.glob("holosoma_e091_*"))
        if path.is_dir()
    ]
    fields = [
        "task",
        "object_name",
        "sequence",
        "person",
        "status",
        "render_status",
        "source_scene",
        "retargeted_npz",
        "trimmed_npz",
        "retargeted_frames",
        "trimmed_frames",
        "cost",
        "retargeted_keyframes_png",
        "trimmed_keyframes_png",
        "timeline_png",
        "retargeted_mp4",
    ]
    results_dir = args.v2_root / "results/omniretarget_visuals"
    write_tsv(results_dir / "omniretarget_visual_manifest.tsv", rows, fields)
    write_json(results_dir / "omniretarget_visual_manifest.json", rows)

    pngs: list[Path] = []
    videos: list[Path] = []
    for row in rows:
        for key in ("retargeted_keyframes_png", "trimmed_keyframes_png", "timeline_png"):
            if row.get(key):
                pngs.append(Path(row[key]))
        if row.get("retargeted_mp4"):
            videos.append(Path(row["retargeted_mp4"]))
    png_records = [
        {"path": str(path), "exists": path.is_file(), "nonblank": image_nonblank(path), "bytes": path.stat().st_size if path.is_file() else 0}
        for path in pngs
    ]
    video_records = [
        {"path": str(path), "exists": path.is_file(), "bytes": path.stat().st_size if path.is_file() else 0}
        for path in videos
    ]
    summary = {
        "stage": "E091 OmniRetarget visuals",
        "case_count": len(rows),
        "ok_count": sum(1 for row in rows if row["status"] == "ok"),
        "missing_retargeted_count": sum(1 for row in rows if row["status"] == "missing_retargeted_npz"),
        "png_count": len(png_records),
        "nonblank_png_count": sum(1 for row in png_records if row["nonblank"]),
        "video_count": len(video_records),
        "video_existing_count": sum(1 for row in video_records if row["exists"] and row["bytes"] > 2048),
        "pngs": png_records,
        "videos": video_records,
    }
    write_json(results_dir / "summary.json", summary)
    lines = [
        "# E091 OmniRetarget Visuals",
        "",
        f"- Cases scanned: `{len(rows)}`",
        f"- OK visualizations: `{summary['ok_count']}`",
        f"- Missing retargeted NPZ: `{summary['missing_retargeted_count']}`",
        f"- Nonblank PNGs: `{summary['nonblank_png_count']}/{summary['png_count']}`",
        f"- MP4 videos: `{summary['video_existing_count']}/{summary['video_count']}`",
        "",
        "| task | status | retargeted frames | trimmed frames |",
        "|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| `{row['task']}` | {row['status']} | {row['retargeted_frames']} | {row['trimmed_frames']} |")
    (results_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"OmniRetarget visual rows: {len(rows)}")
    print(f"OK: {summary['ok_count']}")
    print(f"Nonblank PNGs: {summary['nonblank_png_count']}/{summary['png_count']}")
    print(f"MP4s: {summary['video_existing_count']}/{summary['video_count']}")


if __name__ == "__main__":
    main()
