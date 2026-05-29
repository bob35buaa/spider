#!/usr/bin/env python3
"""Regenerate visual review assets for E097 feature-mined candidates."""

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


SPIDER_REPO = Path("/home/ubuntu/Workspace/spider")
HOLOSOMA_V3 = Path("/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction")
DEFAULT_INPUT = (
    SPIDER_REPO
    / "workspace/core4d/results/E097/feature_candidate_mining/cases_e097_feature_candidate_pipeline.tsv"
)
DEFAULT_OUTPUT = SPIDER_REPO / "workspace/core4d/results/E097/visual_review"
D003_ROOT = HOLOSOMA_V3 / "results/d003_omniretarget_spider_production/results"
D002_RAW_ROOT = HOLOSOMA_V3 / "results/d002_stage1_raw_contact_v2/per_sequence"
PROCESSED_ROOT = SPIDER_REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"

LEGACY_STATUS = {
    "d003_box021_20231018_028_p1": "d003_infeasible_cvxpy",
    "d003_box021_20231018_028_p2": "d004_visual_reject_fall_prone",
    "d003_box021_20231020_020_p2": "d004_visual_pass",
    "d003_box021_20231011_035_p1": "d004_visual_pass",
    "d003_box021_20231018_030_p2": "d004_visual_pass_check_shortcut",
    "d003_box021_20231020_019_p2": "d004_visual_review",
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def image_nonblank(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 2048:
        return False
    try:
        arr = np.asarray(plt.imread(path))
    except Exception:
        return False
    return bool(arr.std() > 1e-4)


def video_readable(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 4096:
        return False
    try:
        reader = imageio.get_reader(path)
        frame = reader.get_data(0)
        reader.close()
    except Exception:
        return False
    return bool(np.asarray(frame).std() > 1e-4)


def d003_task(target_task: str) -> str:
    if target_task.startswith("e091_"):
        return "d003_" + target_task[len("e091_") :]
    return target_task


def stage_dir(task: str) -> Path:
    return D003_ROOT / f"holosoma_{task}"


def qpos_npz(case_dir: Path, stage: str) -> Path | None:
    files = sorted((case_dir / stage).glob("*.npz"))
    return files[0] if files else None


def load_qpos(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=True) as data:
        qpos = np.asarray(data["qpos"], dtype=np.float64)
        meta: dict[str, Any] = {}
        for key in data.files:
            if key == "qpos":
                continue
            value = data[key]
            meta[key] = value.item() if value.shape == () else str(value.shape)
    return qpos, meta


def task_info(task: str) -> dict[str, Any]:
    path = PROCESSED_ROOT / task / "task_info.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def source_scene(task: str) -> Path | None:
    info = task_info(task)
    rel = info.get("source_scene")
    if rel:
        path = SPIDER_REPO / rel
        if path.is_file():
            return path
    fallback = PROCESSED_ROOT / task / "scene.xml"
    return fallback if fallback.is_file() else None


def raw_contact_proxy(row: dict[str, str]) -> Path:
    object_key = row["object_name"].lower()
    return D002_RAW_ROOT / f"{row['date']}_{row['seq']}_{object_key}" / "raw_contact_proxy.npz"


def threshold_index(thresholds: np.ndarray, target_m: float = 0.03) -> int:
    arr = np.asarray(thresholds, dtype=float).reshape(-1)
    return int(np.argmin(np.abs(arr - target_m)))


def shade_active(ax: plt.Axes, frames: np.ndarray, active: np.ndarray) -> None:
    ax.fill_between(
        frames,
        0,
        1,
        where=active,
        transform=ax.get_xaxis_transform(),
        color="#999999",
        alpha=0.13,
        linewidth=0,
    )


def make_raw_contact_plot(row: dict[str, str], out_root: Path) -> dict[str, Any]:
    target = row["target_task"]
    out_path = out_root / "raw_contact" / f"{target}_raw_contact.png"
    proxy = raw_contact_proxy(row)
    result: dict[str, Any] = {
        "raw_contact_proxy": str(proxy),
        "raw_contact_png": str(out_path),
        "raw_contact_status": "",
        "raw_contact_nonblank": False,
        "raw_contact_frames": 0,
        "raw_both_contact_frac_3cm": "",
        "raw_active_frac": "",
    }
    if not proxy.is_file():
        result["raw_contact_status"] = "missing_raw_contact_proxy"
        return result

    with np.load(proxy, allow_pickle=True) as data:
        min_dist = np.asarray(data["min_dist_m"], dtype=float)
        masks = np.asarray(data["masks"], dtype=bool)
        active = np.asarray(data["active_mask"], dtype=bool)
        object_pos = np.asarray(data["object_pos"], dtype=float)
        persons = [str(x) for x in np.asarray(data["persons"]).tolist()]
        hands = [str(x) for x in np.asarray(data["hands"]).tolist()]
        thresholds = np.asarray(data["thresholds_m"], dtype=float)

    person = row["person"]
    if person in persons:
        person_idx = persons.index(person)
        status = "ok"
    else:
        person_idx = 0
        status = f"person_not_found_used_{persons[0]}"

    th_idx = threshold_index(thresholds, 0.03)
    th_cm = float(thresholds.reshape(-1)[th_idx] * 100.0)
    frames = np.arange(min_dist.shape[0])
    hand_dist_cm = min_dist[:, person_idx, :] * 100.0
    hand_masks = masks[:, person_idx, :, th_idx]
    both = np.logical_and(hand_masks[:, 0], hand_masks[:, 1])

    fig, axes = plt.subplots(4, 1, figsize=(13, 10), constrained_layout=True)
    fig.suptitle(
        f"{target} raw contact | {row['date']}-{row['seq']} {person} {row['object_name']} | "
        f"E097 score {row.get('e097_score', '')}"
    )

    for hand_idx, hand in enumerate(hands):
        axes[0].plot(frames, hand_dist_cm[:, hand_idx], label=f"{hand} min distance")
    axes[0].axhline(th_cm, color="#cc3311", linestyle="--", linewidth=1.2, label=f"{th_cm:.1f} cm")
    shade_active(axes[0], frames, active)
    axes[0].set_ylabel("distance (cm)")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(loc="upper right")
    axes[0].set_title("Target hand to object distance")

    axes[1].plot(frames, hand_masks[:, 0].astype(int), label=f"{hands[0]} <= {th_cm:.1f} cm")
    axes[1].plot(frames, hand_masks[:, 1].astype(int) + 1.2, label=f"{hands[1]} <= {th_cm:.1f} cm")
    axes[1].plot(frames, both.astype(int) + 2.4, label="both hands")
    axes[1].plot(frames, active.astype(int) + 3.6, label="object active")
    axes[1].set_yticks([0, 1.2, 2.4, 3.6])
    axes[1].set_yticklabels([hands[0], hands[1], "both", "active"])
    axes[1].set_ylim(-0.2, 4.9)
    axes[1].legend(loc="upper right")
    axes[1].set_title("3cm contact masks")

    for idx, label in enumerate(["x", "y", "z"]):
        axes[2].plot(frames, object_pos[:, idx], label=f"object {label}")
    shade_active(axes[2], frames, active)
    axes[2].set_ylabel("position (m)")
    axes[2].legend(loc="upper right")
    axes[2].set_title("Raw object trajectory")

    axes[3].plot(object_pos[:, 0], object_pos[:, 1], color="#4477aa")
    axes[3].scatter(object_pos[0, 0], object_pos[0, 1], color="#228833", label="start")
    axes[3].scatter(object_pos[-1, 0], object_pos[-1, 1], color="#cc3311", label="end")
    axes[3].axis("equal")
    axes[3].legend(loc="best")
    axes[3].set_xlabel("object x (m)")
    axes[3].set_ylabel("object y (m)")
    axes[3].set_title("Object XY path")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    result.update(
        {
            "raw_contact_status": status,
            "raw_contact_nonblank": image_nonblank(out_path),
            "raw_contact_frames": int(min_dist.shape[0]),
            "raw_both_contact_frac_3cm": round(float(both.mean()), 6),
            "raw_active_frac": round(float(active.mean()), 6),
        }
    )
    return result


def full_body_camera(model: mujoco.MjModel, data: mujoco.MjData) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    names: list[int] = []
    for idx in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, idx) or ""
        if name:
            names.append(idx)
    points = data.xpos[names] if names else data.xpos[1:]
    finite = points[np.isfinite(points).all(axis=1)]
    if finite.size == 0:
        center = np.array([0.0, 0.0, 0.8])
        span = 2.0
    else:
        lo = finite.min(axis=0)
        hi = finite.max(axis=0)
        center = (lo + hi) * 0.5
        span = float(np.max(hi - lo))
        center[2] = max(float(center[2]), 0.8)
    cam.lookat[:] = center
    cam.distance = max(2.2, span * 2.1)
    cam.azimuth = 135.0
    cam.elevation = -18.0
    return cam


def set_qpos(model: mujoco.MjModel, data: mujoco.MjData, qpos: np.ndarray) -> None:
    data.qpos[:] = 0.0
    n = min(model.nq, qpos.shape[-1])
    data.qpos[:n] = qpos[:n]
    mujoco.mj_forward(model, data)


def render_index(
    renderer: mujoco.Renderer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: np.ndarray,
    idx: int,
    use_track2: bool,
) -> np.ndarray:
    set_qpos(model, data, qpos[int(idx)])
    auto_cam = full_body_camera(model, data)
    renderer.update_scene(data, camera=auto_cam)
    auto = renderer.render().copy()
    if not use_track2:
        return auto
    renderer.update_scene(data, camera="track2")
    track2 = renderer.render().copy()
    return np.concatenate([auto, track2], axis=1)


def render_frames(scene_xml: Path, qpos: np.ndarray, frame_indices: np.ndarray) -> list[np.ndarray]:
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=640, height=432)
    camera_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, idx) for idx in range(model.ncam)
    }
    use_track2 = "track2" in camera_names
    frames = [render_index(renderer, model, data, qpos, int(idx), use_track2) for idx in frame_indices]
    renderer.close()
    return frames


def make_keyframes(scene_xml: Path, qpos: np.ndarray, out_path: Path, title: str) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    idxs = np.linspace(0, len(qpos) - 1, num=min(6, len(qpos)), dtype=int)
    try:
        frames = render_frames(scene_xml, qpos, idxs)
    except Exception as exc:  # noqa: BLE001 - keep manifest actionable.
        return f"render_failed:{exc}"

    fig, axes = plt.subplots(2, 3, figsize=(16, 7), constrained_layout=True)
    for ax, img, idx in zip(axes.flat, frames, idxs):
        ax.imshow(img)
        ax.set_title(f"frame {int(idx)}")
        ax.axis("off")
    for ax in axes.flat[len(frames) :]:
        ax.axis("off")
    fig.suptitle(title)
    fig.savefig(out_path, dpi=145)
    plt.close(fig)
    return "ok" if image_nonblank(out_path) else "blank_output"


def make_video(scene_xml: Path, qpos: np.ndarray, out_path: Path, fps: int) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    target_fps = min(15, max(1, fps))
    stride = max(1, int(round(fps / target_fps)))
    idxs = np.arange(0, len(qpos), stride, dtype=int)
    try:
        frames = render_frames(scene_xml, qpos, idxs)
        with imageio.get_writer(out_path, fps=target_fps, codec="libx264", quality=8) as writer:
            for frame in frames:
                writer.append_data(frame)
    except Exception as exc:  # noqa: BLE001 - keep manifest actionable.
        return f"video_failed:{exc}"
    return "ok" if video_readable(out_path) else "unreadable_video"


def make_timeline(
    target: str,
    retarget_qpos: np.ndarray,
    trimmed_qpos: np.ndarray | None,
    out_path: Path,
) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.arange(len(retarget_qpos))
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.5), sharex=False, constrained_layout=True)

    axes[0].plot(frames, retarget_qpos[:, 2], label="pelvis z", color="#332288")
    if retarget_qpos.shape[1] > 38:
        axes[0].plot(frames, retarget_qpos[:, 38], label="object z", color="#117733")
    axes[0].set_ylabel("z (m)")
    axes[0].set_title("Retargeted height")
    axes[0].legend(loc="best")

    root_xy = np.linalg.norm(retarget_qpos[:, :2] - retarget_qpos[0, :2], axis=1)
    axes[1].plot(frames, root_xy, label="root xy displacement", color="#cc6677")
    if retarget_qpos.shape[1] > 37:
        obj_xy = np.linalg.norm(retarget_qpos[:, 36:38] - retarget_qpos[0, 36:38], axis=1)
        axes[1].plot(frames, obj_xy, label="object xy displacement", color="#44aa99")
    axes[1].set_ylabel("m")
    axes[1].set_title("Retargeted displacement")
    axes[1].legend(loc="best")

    if trimmed_qpos is None:
        axes[2].text(0.5, 0.5, "No trimmed qpos", ha="center", va="center", transform=axes[2].transAxes)
        axes[2].axis("off")
    else:
        trimmed_frames = np.arange(len(trimmed_qpos))
        axes[2].plot(trimmed_frames, trimmed_qpos[:, 2], label="trimmed pelvis z", color="#332288")
        if trimmed_qpos.shape[1] > 38:
            axes[2].plot(trimmed_frames, trimmed_qpos[:, 38], label="trimmed object z", color="#117733")
        axes[2].set_xlabel("trimmed frame")
        axes[2].set_ylabel("z (m)")
        axes[2].set_title("Trimmed height")
        axes[2].legend(loc="best")

    fig.suptitle(f"{target} retarget timeline")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return "ok" if image_nonblank(out_path) else "blank_output"


def make_retarget_visuals(row: dict[str, str], out_root: Path) -> dict[str, Any]:
    target = row["target_task"]
    old_task = d003_task(target)
    case_dir = stage_dir(old_task)
    scene_xml = source_scene(old_task)
    retarget_npz = qpos_npz(case_dir, "retargeted")
    trimmed_npz = qpos_npz(case_dir, "trimmed")
    result: dict[str, Any] = {
        "d003_task": old_task,
        "legacy_status": LEGACY_STATUS.get(old_task, ""),
        "scene_xml": str(scene_xml) if scene_xml else "",
        "retargeted_npz": str(retarget_npz) if retarget_npz else "",
        "trimmed_npz": str(trimmed_npz) if trimmed_npz else "",
        "retargeted_frames": 0,
        "trimmed_frames": 0,
        "retargeted_keyframes_png": "",
        "trimmed_keyframes_png": "",
        "timeline_png": "",
        "retargeted_mp4": "",
        "trimmed_mp4": "",
        "retarget_status": "",
        "trimmed_status": "",
        "timeline_status": "",
        "retarget_cost": "na",
    }
    if retarget_npz is None:
        result["retarget_status"] = "missing_retargeted_npz"
        result["trimmed_status"] = "missing_trimmed_npz"
        result["timeline_status"] = "missing_retargeted_npz"
        return result
    if scene_xml is None or not scene_xml.is_file():
        result["retarget_status"] = "missing_scene_xml"
        result["trimmed_status"] = "missing_scene_xml"
        result["timeline_status"] = "missing_scene_xml"
        return result

    retarget_qpos, retarget_meta = load_qpos(retarget_npz)
    trimmed_qpos = None
    if trimmed_npz is not None:
        trimmed_qpos, _ = load_qpos(trimmed_npz)
    fps = int(retarget_meta.get("fps", 30) or 30)
    result["retargeted_frames"] = int(len(retarget_qpos))
    result["trimmed_frames"] = int(len(trimmed_qpos)) if trimmed_qpos is not None else 0
    if "cost" in retarget_meta:
        result["retarget_cost"] = retarget_meta["cost"]

    retarget_keyframes = out_root / "mujoco_keyframes" / f"{target}_retargeted_keyframes.png"
    retarget_mp4 = out_root / "mujoco_videos" / f"{target}_retargeted.mp4"
    timeline = out_root / "timelines" / f"{target}_timeline.png"
    result["retargeted_keyframes_png"] = str(retarget_keyframes)
    result["retargeted_mp4"] = str(retarget_mp4)
    result["timeline_png"] = str(timeline)
    result["retarget_status"] = ";".join(
        [
            make_keyframes(scene_xml, retarget_qpos, retarget_keyframes, f"{target} retargeted"),
            make_video(scene_xml, retarget_qpos, retarget_mp4, fps),
        ]
    )
    result["timeline_status"] = make_timeline(target, retarget_qpos, trimmed_qpos, timeline)

    if trimmed_qpos is None:
        result["trimmed_status"] = "missing_trimmed_npz"
    else:
        trimmed_keyframes = out_root / "mujoco_keyframes" / f"{target}_trimmed_keyframes.png"
        trimmed_mp4 = out_root / "mujoco_videos" / f"{target}_trimmed.mp4"
        result["trimmed_keyframes_png"] = str(trimmed_keyframes)
        result["trimmed_mp4"] = str(trimmed_mp4)
        result["trimmed_status"] = ";".join(
            [
                make_keyframes(scene_xml, trimmed_qpos, trimmed_keyframes, f"{target} trimmed"),
                make_video(scene_xml, trimmed_qpos, trimmed_mp4, fps),
            ]
        )
    return result


def make_summary(rows: list[dict[str, Any]], out_root: Path, input_path: Path) -> None:
    lines = [
        "# E097 visual review",
        "",
        "Generated assets are under this directory. MuJoCo panels use an auto-framed full-body camera plus `track2` when available.",
        "",
        "| target | legacy status | raw contact | retarget | trimmed | mp4 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        target = row["target_task"]
        raw = "ok" if row.get("raw_contact_nonblank") else row.get("raw_contact_status", "")
        retarget = row.get("retarget_status", "")
        trimmed = row.get("trimmed_status", "")
        mp4 = "yes" if row.get("retargeted_mp4") and video_readable(Path(row["retargeted_mp4"])) else "no"
        lines.append(
            f"| `{target}` | {row.get('legacy_status', '')} | {raw} | {retarget} | {trimmed} | {mp4} |"
        )
    lines.extend(
        [
            "",
            "Key files:",
            "",
            f"- Visual input TSV: `{input_path}`",
            f"- Manifest TSV: `{out_root / 'visual_manifest.tsv'}`",
            f"- Manifest JSON: `{out_root / 'visual_manifest.json'}`",
            f"- Raw contact plots: `{out_root / 'raw_contact'}`",
            f"- MuJoCo keyframes: `{out_root / 'mujoco_keyframes'}`",
            f"- MuJoCo videos: `{out_root / 'mujoco_videos'}`",
            f"- Timelines: `{out_root / 'timelines'}`",
            "",
        ]
    )
    (out_root / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--include-disabled", action="store_true")
    args = parser.parse_args()

    input_rows = read_tsv(args.input)
    selected = [
        row
        for row in input_rows
        if args.include_disabled or row.get("enabled", row.get("# enabled", "")) == "1"
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    visual_input = args.output / "visual_input_cases.tsv"
    write_tsv(visual_input, selected)

    rows: list[dict[str, Any]] = []
    for row in selected:
        merged: dict[str, Any] = dict(row)
        print(f"[E097 visual] {row['target_task']}", flush=True)
        merged.update(make_raw_contact_plot(row, args.output))
        merged.update(make_retarget_visuals(row, args.output))
        rows.append(merged)

    write_tsv(args.output / "visual_manifest.tsv", rows)
    write_json(args.output / "visual_manifest.json", rows)
    make_summary(rows, args.output, visual_input)
    print(f"Wrote {len(rows)} visual rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
