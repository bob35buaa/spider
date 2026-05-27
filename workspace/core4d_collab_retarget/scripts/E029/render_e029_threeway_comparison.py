#!/usr/bin/env python3
"""Render three-way comparison videos for E029 D003 Box021 candidates.

Panels:
  left   - original OmniRetarget kinematic trajectory
  middle - E018b-style canonical support-proxy rollout on the same D003 case
           (stored as E028 canonical_t02)
  right  - E029 best gate-eligible D6 locked sanity rollout
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import e029_common as common  # noqa: E402


RESULTS = common.REPO / "workspace/core4d_collab_retarget/results"
E028 = RESULTS / "E028"
E029 = RESULTS / "E029"
OUT_DIR = E029 / "compare_threeway"


def _read_e028_manifest() -> dict[str, dict[str, str]]:
    with (E028 / "manifest.tsv").open("r", encoding="utf-8", newline="") as f:
        return {row["variant"]: row for row in csv.DictReader(f, delimiter="\t")}


def _read_candidates() -> list[str]:
    return list(json.loads((E028 / "candidates.json").read_text(encoding="utf-8")))


def _flatten(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:
        return arr.reshape(-1, arr.shape[-1])
    return arr


def _sample_indices(length: int, target_count: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("cannot sample empty trajectory")
    return np.round(np.linspace(0, length - 1, target_count)).astype(int)


def _mocap_id(model: mujoco.MjModel, body_name: str) -> int | None:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        return None
    mid = int(model.body_mocapid[bid])
    return mid if mid >= 0 else None


def _object_quat(model: mujoco.MjModel, qpos: np.ndarray) -> np.ndarray | None:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    if bid < 0:
        return None
    jadr = int(model.body_jntadr[bid])
    if jadr < 0:
        return None
    qadr = int(model.jnt_qposadr[jadr])
    if qadr + 7 > len(qpos):
        return None
    return qpos[qadr + 3 : qadr + 7].copy()


def _update_scene(renderer: mujoco.Renderer, data: mujoco.MjData) -> None:
    try:
        renderer.update_scene(data, "track")
    except Exception:
        try:
            renderer.update_scene(data, "front")
        except Exception:
            renderer.update_scene(data, 0)


def _label(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 36), (20, 20, 20), -1)
    cv2.putText(
        out,
        text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    return out


def _render_qpos_panel(
    *,
    scene: Path,
    qpos: np.ndarray,
    label: str,
    target_count: int,
    width: int,
    height: int,
    mocap_body: str | None = None,
    mocap_pos: np.ndarray | None = None,
) -> list[np.ndarray]:
    model = mujoco.MjModel.from_xml_path(str(scene))
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    renderer = mujoco.Renderer(model, height=height, width=width)
    data = mujoco.MjData(model)
    mid = _mocap_id(model, mocap_body) if mocap_body else None
    qpos = _flatten(qpos).astype(np.float64)
    mocap_pos = _flatten(mocap_pos).astype(np.float64) if mocap_pos is not None else None
    frames: list[np.ndarray] = []
    for idx in _sample_indices(len(qpos), target_count):
        data.qpos[: model.nq] = qpos[idx, : model.nq]
        if mid is not None and mocap_pos is not None:
            mp_idx = min(idx, len(mocap_pos) - 1)
            data.mocap_pos[mid] = mocap_pos[mp_idx]
            quat = _object_quat(model, data.qpos)
            if quat is not None:
                data.mocap_quat[mid] = quat
        mujoco.mj_forward(model, data)
        _update_scene(renderer, data)
        frames.append(_label(renderer.render(), label))
    renderer.close()
    return frames


def _read_video_frames(
    path: Path,
    *,
    label: str,
    target_size: tuple[int, int],
    output_fps: int,
) -> list[np.ndarray]:
    reader = imageio.get_reader(str(path))
    meta = reader.get_meta_data()
    input_fps = float(meta.get("fps") or output_fps)
    frames = []
    width, height = target_size
    for frame in reader:
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        frames.append(_label(resized, label))
    reader.close()
    if not frames:
        raise ValueError(f"video has no frames: {path}")
    if input_fps > output_fps:
        target_count = max(1, round(len(frames) * output_fps / input_fps))
        frames = [frames[i] for i in _sample_indices(len(frames), target_count)]
    return frames


def _d6_video_path(source_task: str) -> Path:
    stem = f"E029_{source_task}_d6_locked_candidates_locked_raw_viz_sanity.mp4"
    path = E029 / "d6/sanity" / stem
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _e028_phys_qpos(variant: str) -> tuple[np.ndarray, np.ndarray | None]:
    data = np.load(E028 / f"{variant}_outdir/trajectory_mjwp.npz", allow_pickle=True)
    support = data["support_proxy_pos"] if "support_proxy_pos" in data.files else None
    return data["qpos"], support


def render_variant(
    variant: str,
    row: dict[str, str],
    *,
    out_dir: Path,
    width: int,
    height: int,
    fps: int,
) -> Path:
    source_task = row["source_task"]
    source_scene = common.BASE / source_task / "scene.xml"
    source_npz = Path(row["trajectory_path"])
    source_qpos = np.load(source_npz, allow_pickle=True)["qpos"]

    e028_scene = common.BASE / row["derived_task"] / f"{row['scene_name']}.xml"
    e028_qpos, e028_support = _e028_phys_qpos(variant)

    d6_video = _d6_video_path(source_task)
    right_frames = _read_video_frames(
        d6_video,
        label="E029 best: D6 locked",
        target_size=(width, height),
        output_fps=fps,
    )
    target_count = len(right_frames)

    left_frames = _render_qpos_panel(
        scene=source_scene,
        qpos=source_qpos,
        label="Original OmniRetarget",
        target_count=target_count,
        width=width,
        height=height,
    )
    middle_frames = _render_qpos_panel(
        scene=e028_scene,
        qpos=e028_qpos,
        label="E018b-style: E028 canonical",
        target_count=target_count,
        width=width,
        height=height,
        mocap_body="support_weld_anchor",
        mocap_pos=e028_support,
    )

    frame_count = min(len(left_frames), len(middle_frames), len(right_frames))
    frames = [
        np.concatenate([left_frames[i], middle_frames[i], right_frames[i]], axis=1)
        for i in range(frame_count)
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{source_task}_omniretarget_e018bstyle_e029d6locked_compare.mp4"
    imageio.mimsave(str(out), frames, fps=fps, codec="libx264", quality=8)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=368)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--variant", action="append", default=[])
    args = ap.parse_args()

    manifest = _read_e028_manifest()
    variants = args.variant or _read_candidates()
    rows: list[dict[str, str]] = []
    for variant in variants:
        row = manifest[variant]
        out = render_variant(
            variant,
            row,
            out_dir=args.out_dir,
            width=args.width,
            height=args.height,
            fps=args.fps,
        )
        rows.append(
            {
                "source_task": row["source_task"],
                "e028_variant": variant,
                "left": "Original OmniRetarget",
                "middle": "E018b-style canonical support proxy (E028)",
                "right": "E029 D6 locked raw sanity",
                "video_path": common.rel_or_abs(out),
            }
        )
        print(common.rel_or_abs(out))

    csv_path = args.out_dir / "threeway_comparison_index.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    md_path = args.out_dir / "threeway_comparison_index.md"
    lines = [
        "# E029 三列对比可视化",
        "",
        "左：原始 OmniRetarget；中：E018b-style canonical support proxy（同 case 的 E028 rollout）；右：E029 best D6 locked sanity。",
        "",
        f"输出帧率：{args.fps}fps；对右侧 sanity 视频按时间均匀抽帧，保持原视频时长。",
        "",
        "| Case | Video |",
        "|---|---|",
    ]
    for row in rows:
        lines.append(f"| `{row['source_task']}` | `{row['video_path']}` |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(common.rel_or_abs(csv_path))
    print(common.rel_or_abs(md_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
