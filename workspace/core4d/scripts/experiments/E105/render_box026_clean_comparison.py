#!/usr/bin/env python3
"""Build E105 Box026 clean rerun visual comparison artifacts."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/E105"))

from e105_common import TASK_ROOT, rel, read_variants  # noqa: E402


EEF_OFFSET = np.array([0.05, 0.0, 0.0], dtype=np.float64)


def repo_path(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return REPO / path


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def read_csv_dict(path: Path, key: str) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        return {row[key]: row for row in csv.DictReader(f)}


def parse_box_half(scene_xml: Path) -> np.ndarray:
    tree = ET.parse(scene_xml)
    for body in tree.getroot().iter("body"):
        if body.get("name") != "object":
            continue
        for geom in body.iter("geom"):
            name = geom.get("name", "")
            if geom.get("type") == "box" and "collision" in name.lower():
                return np.array([float(x) for x in geom.get("size", "").split()], dtype=np.float64)
    raise ValueError(f"object collision box not found in {scene_xml}")


def body_id(model: mujoco.MjModel, name: str) -> int:
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))


def rollout_qpos(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    qpos = data["qpos"]
    if qpos.ndim != 3 or qpos.shape[1] < 2:
        raise ValueError(f"expected qpos shape (T,2,nq), got {qpos.shape} in {npz_path}")
    return qpos[:, 0], qpos[:, 1]


def timeline(npz_path: Path, scene_xml: Path) -> dict[str, np.ndarray]:
    sim_q, ref_q = rollout_qpos(npz_path)
    T = min(len(sim_q), len(ref_q))
    sim_q = sim_q[:T]
    ref_q = ref_q[:T]

    half = parse_box_half(scene_xml)
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    mj = mujoco.MjData(model)
    obj_b = body_id(model, "object")
    wrist_l = body_id(model, "left_wrist_yaw_link")
    wrist_r = body_id(model, "right_wrist_yaw_link")

    obj_sim = np.zeros((T, 3), dtype=np.float64)
    obj_ref = np.zeros((T, 3), dtype=np.float64)
    pelvis = sim_q[:, 2].astype(np.float64)
    dist_l = np.zeros(T, dtype=np.float64)
    dist_r = np.zeros(T, dtype=np.float64)

    for t in range(T):
        mj.qpos[:] = sim_q[t]
        mujoco.mj_forward(model, mj)
        obj_pos = mj.xpos[obj_b].copy()
        obj_mat = mj.xmat[obj_b].reshape(3, 3).copy()
        obj_sim[t] = obj_pos
        for bid, store in [(wrist_l, dist_l), (wrist_r, dist_r)]:
            pos = mj.xpos[bid].copy()
            quat = mj.xquat[bid].copy()
            rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
            hand = pos + rot.apply(EEF_OFFSET)
            local = obj_mat.T @ (hand - obj_pos)
            outside = np.maximum(np.abs(local) - half, 0.0)
            store[t] = float(np.linalg.norm(outside))

        mj.qpos[:] = ref_q[t]
        mujoco.mj_forward(model, mj)
        obj_ref[t] = mj.xpos[obj_b].copy()

    return {
        "frame": np.arange(T),
        "obj_err": np.linalg.norm(obj_sim - obj_ref, axis=1),
        "pelvis_z": pelvis,
        "dist_l": dist_l,
        "dist_r": dist_r,
    }


def plot_timeline(name: str, series: dict[str, np.ndarray], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = series["frame"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(x, series["obj_err"], color="#1f77b4", lw=1.8)
    axes[0].axhline(0.10, color="#d62728", ls="--", lw=1.0)
    axes[0].set_ylabel("obj err m")
    axes[0].grid(alpha=0.25)

    axes[1].plot(x, series["pelvis_z"], color="#2ca02c", lw=1.8)
    axes[1].axhline(0.55, color="#d62728", ls="--", lw=1.0)
    axes[1].set_ylabel("pelvis z m")
    axes[1].grid(alpha=0.25)

    axes[2].plot(x, series["dist_l"], label="L", color="#9467bd", lw=1.5)
    axes[2].plot(x, series["dist_r"], label="R", color="#ff7f0e", lw=1.5)
    axes[2].axhline(0.08, color="#d62728", ls="--", lw=1.0)
    axes[2].set_ylabel("hand box dist m")
    axes[2].set_xlabel("frame")
    axes[2].legend(loc="upper right")
    axes[2].grid(alpha=0.25)

    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def frame_at(video: Path, ratio: float) -> Image.Image | None:
    if not video.is_file():
        return None
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release()
        return None
    idx = max(0, min(n - 1, int(round((n - 1) * ratio))))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def labeled_panel(img: Image.Image | None, label: str, size: tuple[int, int]) -> Image.Image:
    w, h = size
    if img is None:
        panel = Image.new("RGB", size, (32, 32, 32))
        d = ImageDraw.Draw(panel)
        d.text((18, h // 2 - 10), "missing video/frame", fill=(230, 230, 230), font=font(22))
    else:
        panel = img.copy()
        panel.thumbnail((w, h), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", size, (12, 12, 12))
        canvas.paste(panel, ((w - panel.width) // 2, (h - panel.height) // 2))
        panel = canvas
    d = ImageDraw.Draw(panel, "RGBA")
    d.rectangle([(0, 0), (w, 34)], fill=(0, 0, 0, 180))
    d.text((10, 7), label, fill=(255, 255, 255), font=font(20))
    return panel


def comparison_sheet(
    new_variant: str,
    new_video: Path,
    old_label: str,
    old_video: Path | None,
    out_path: Path,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ratios = [("early", 0.15), ("mid", 0.50), ("late", 0.85)]
    cell = (480, 300)
    rows: list[Image.Image] = []
    for label, ratio in ratios:
        old_img = frame_at(old_video, ratio) if old_video else None
        new_img = frame_at(new_video, ratio)
        left = labeled_panel(old_img, f"{old_label or 'no old baseline'} | {label}", cell)
        right = labeled_panel(new_img, f"{new_variant} | {label}", cell)
        row = Image.new("RGB", (cell[0] * 2, cell[1]), (255, 255, 255))
        row.paste(left, (0, 0))
        row.paste(right, (cell[0], 0))
        rows.append(row)
    sheet = Image.new("RGB", (cell[0] * 2, cell[1] * len(rows)), (255, 255, 255))
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    sheet.save(out_path, quality=92)
    return out_path


def old_video_for(old_variant: str) -> Path | None:
    if not old_variant:
        return None
    candidates = [
        REPO / "workspace/core4d/results/E092/spider_dyn/full" / f"{old_variant}_full.mp4",
        REPO / "workspace/core4d/results/E094/cem/full" / f"{old_variant}_full_autocam.mp4",
        REPO / "workspace/core4d/results/E094/cem/full" / f"{old_variant}_full.mp4",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def old_npz_scene_for(old_variant: str) -> tuple[Path | None, Path | None]:
    if not old_variant:
        return None, None
    roots = [
        REPO / "workspace/core4d/results/E092/spider_dyn/full",
        REPO / "workspace/core4d/results/E094/cem/full",
    ]
    for root in roots:
        out_dir = root / f"{old_variant}_outdir_full"
        npz = out_dir / "trajectory_mjwp_act.npz"
        cfg = out_dir / "config_act.yaml"
        if npz.is_file() and cfg.is_file():
            for line in cfg.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.strip().startswith("model_path:"):
                    raw = line.split(":", 1)[1].strip().strip("'\"")
                    scene = repo_path(raw)
                    if scene.is_file():
                        return npz, scene
            return npz, None
    return None, None


def write_review(rows: list[dict[str, str]], metrics: dict[str, dict[str, str]], out_dir: Path) -> None:
    lines = [
        "# E105 Box026 Clean Rerun Visual Review",
        "",
        "Scope: clean-scene Box026 full CEM reruns after E103 source template rebuild.",
        "",
        "## Artifacts",
        "",
        "| variant | route | old baseline | status | metrics | sheet | timeline |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        variant = row["variant"]
        m = metrics.get(variant, {})
        metric_text = ""
        if m:
            metric_text = (
                f"contact {float(m.get('contact_frac_either', 0)) * 100:.1f}%, "
                f"obj {float(m.get('obj_err_mean_m', 0)):.3f}/{float(m.get('obj_err_max_m', 0)):.3f}m, "
                f"pelvis {float(m.get('pelvis_min_m', 0)):.3f}m, "
                f"leg-intf {float(m.get('leg_box_interference_frac', 0)) * 100:.1f}%"
            )
        sheet = out_dir / "sheets" / f"{variant}_old_vs_new.jpg"
        timeline_png = out_dir / "timelines" / f"{variant}_timeline.png"
        lines.append(
            f"| `{variant}` | `{row['target_route']}` | `{row.get('old_variant', '')}` | "
            f"{m.get('work_status', '')}/{m.get('work_status_lowerbody_strict', '')} | {metric_text} | "
            f"[sheet]({rel(sheet)}) | [timeline]({rel(timeline_png)}) |"
        )
    lines += [
        "",
        "## Visual QC Notes",
        "",
        "- Sheets compare early/mid/late frames. For E105 fingertip secondary ablations, the left column is intentionally blank because there is no historical full-CEM Box026 E101 baseline.",
        "- Timeline thresholds: object mean target band is shown with 10cm reference, pelvis work threshold with 55cm reference, and hand-box proximity with 8cm reference.",
        "- Status is shown as upper-body/replay gate status followed by lower-body strict status. Lower-body strict follows E026/E081 leg-box interference <=5%.",
        "- Final interpretation should use this visual package together with `comparison/box026_clean_vs_old_comparison.md`.",
        "",
    ]
    (out_dir / "REVIEW.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="full")
    parser.add_argument("--results", type=Path, default=REPO / "workspace/core4d/results/E105/cem/full")
    parser.add_argument("--out-dir", type=Path, default=REPO / "workspace/core4d/results/E105/visuals/box026_clean_rerun")
    parser.add_argument("--eval-csv", type=Path, default=REPO / "workspace/core4d/results/E105/cem/full/full_eval_summary.csv")
    parser.add_argument("variants", nargs="*")
    args = parser.parse_args()

    results_dir = repo_path(args.results)
    out_dir = repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = read_variants()
    selected_names = set(args.variants) if args.variants else {row["variant"] for row in variants}
    rows = [row for row in variants if row["variant"] in selected_names]
    metrics = read_csv_dict(repo_path(args.eval_csv), "variant")

    for row in rows:
        variant = row["variant"]
        new_video = results_dir / f"{variant}_{args.stage}.mp4"
        old_video = old_video_for(row.get("old_variant", ""))
        comparison_sheet(
            variant,
            new_video,
            row.get("old_variant", ""),
            old_video,
            out_dir / "sheets" / f"{variant}_old_vs_new.jpg",
        )

        new_npz = results_dir / f"{variant}_outdir_{args.stage}/trajectory_mjwp_act.npz"
        scene = TASK_ROOT / row["derived_task"] / "scene_act.xml"
        if new_npz.is_file() and scene.is_file():
            plot_timeline(variant, timeline(new_npz, scene), out_dir / "timelines" / f"{variant}_timeline.png")
        old_npz, old_scene = old_npz_scene_for(row.get("old_variant", ""))
        if old_npz and old_scene:
            plot_timeline(
                row["old_variant"],
                timeline(old_npz, old_scene),
                out_dir / "timelines" / f"{row['old_variant']}_old_timeline.png",
            )
        print(f"[OK] {variant}")

    write_review(rows, metrics, out_dir)
    print(f"review -> {rel(out_dir / 'REVIEW.md')}")


if __name__ == "__main__":
    main()
