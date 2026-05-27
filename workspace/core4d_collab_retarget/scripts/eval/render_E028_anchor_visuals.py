#!/usr/bin/env python3
"""Render anchor diagnostics for the E028 D003 Box021 candidates.

This is intentionally narrower than the rollout videos: it visualizes the
actual support-proxy anchor used by E028, plus the object-local contact points
that drove the side-face estimate.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import imageio
import matplotlib
import mujoco
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[4]
BASE = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
RESULTS = REPO / "workspace/core4d_collab_retarget/results/E028"
MANIFEST = RESULTS / "manifest.tsv"
CANDIDATES = RESULTS / "candidates.json"
OUT_DIR = RESULTS / "anchor_visual"

FACES = ["+x", "-x", "+y", "-y", "+z", "-z"]
SIDE_FACES = ["+x", "-x", "+y", "-y"]


def _rel(path: Path) -> Path:
    return path.resolve().relative_to(REPO)


@dataclass(frozen=True)
class Marker:
    label: str
    point_local: np.ndarray
    rgba: tuple[float, float, float, float]
    bgr: tuple[int, int, int]


def _read_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return {row["variant"]: row for row in csv.DictReader(f, delimiter="\t")}


def _read_candidates(path: Path) -> list[str]:
    return list(json.loads(path.read_text(encoding="utf-8")))


def _quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q / np.clip(np.linalg.norm(q), 1e-8, None)
    v = np.asarray(v, dtype=np.float64)
    qvec = q[1:]
    t = 2.0 * np.cross(qvec, v)
    return v + q[0] * t + np.cross(qvec, t)


def _quat_apply_inv(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qi = np.asarray(q, dtype=np.float64).copy()
    qi = qi / np.clip(np.linalg.norm(qi), 1e-8, None)
    qi[1:] *= -1.0
    return _quat_apply(qi, v)


def _object_qadr(model: mujoco.MjModel) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    if body_id < 0:
        raise ValueError("model has no body named object")
    joint_id = int(model.body_jntadr[body_id])
    if joint_id < 0:
        raise ValueError("object body has no joint")
    return int(model.jnt_qposadr[joint_id])


def _face_label(point: np.ndarray, half: np.ndarray) -> str:
    norm = np.abs(point) / np.clip(half, 1e-6, None)
    axis = int(np.argmax(norm))
    sign = "+" if point[axis] >= 0.0 else "-"
    return f"{sign}{'xyz'[axis]}"


def _mask_for_person(mask_npz: Path, person_idx: int, target_len: int) -> np.ndarray | None:
    if not mask_npz.is_file():
        return None
    data = np.load(mask_npz, allow_pickle=True)
    key = "spider_contact_mask_3cm" if "spider_contact_mask_3cm" in data else "eval_contact_mask_3cm"
    if key not in data:
        return None
    raw = data[key]
    if raw.ndim != 3:
        return None
    person_idx = min(max(person_idx, 0), raw.shape[1] - 1)
    mask = raw[:, person_idx, :2].astype(bool)
    if len(mask) == target_len:
        return mask
    if len(mask) == 0:
        return None
    idx = np.round(np.linspace(0, len(mask) - 1, target_len)).astype(int)
    return mask[idx]


def _load_case(row: dict[str, str]) -> tuple[mujoco.MjModel, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    task_dir = BASE / row["source_task"]
    scene = task_dir / "scene.xml"
    traj_path = task_dir / "0/trajectory_kinematic.npz"
    model = mujoco.MjModel.from_xml_path(str(scene))
    traj = np.load(traj_path)
    qpos = traj["qpos"].reshape(-1, model.nq).astype(np.float64)
    contact_pos = traj["contact_pos"].reshape(qpos.shape[0], 2, 3).astype(np.float64)
    contact = traj["contact"].reshape(qpos.shape[0], 2).astype(bool) if "contact" in traj else None
    mask = _mask_for_person(Path(row["mask_path_source"]), int(row["person_idx"]), qpos.shape[0])
    if contact is not None and mask is not None:
        active = np.logical_or(contact, mask)
    elif contact is not None:
        active = contact
    elif mask is not None:
        active = mask
    else:
        active = np.ones((qpos.shape[0], 2), dtype=bool)
    return model, qpos, contact_pos, active, mask


def _contact_points_local(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    contact_pos: np.ndarray,
    active: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[list[np.ndarray]]]:
    qadr = _object_qadr(model)
    points: list[np.ndarray] = []
    hands: list[int] = []
    per_frame: list[list[np.ndarray]] = []
    for i, q in enumerate(qpos):
        obj_pos = q[qadr : qadr + 3].astype(np.float64)
        obj_quat = q[qadr + 3 : qadr + 7].astype(np.float64)
        frame_points: list[np.ndarray] = []
        for hand_idx in range(2):
            if not bool(active[i, hand_idx]):
                continue
            world = contact_pos[i, hand_idx]
            if not np.all(np.isfinite(world)):
                continue
            local = _quat_apply_inv(obj_quat, world - obj_pos)
            if not np.all(np.isfinite(local)):
                continue
            points.append(local)
            hands.append(hand_idx)
            frame_points.append(world)
        per_frame.append(frame_points)
    if not points:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.int64), per_frame
    return np.asarray(points, dtype=np.float64), np.asarray(hands, dtype=np.int64), per_frame


def _axis_limits(half: np.ndarray, axis_a: int, axis_b: int) -> tuple[tuple[float, float], tuple[float, float]]:
    pad_a = max(0.08, float(half[axis_a]) * 0.35)
    pad_b = max(0.08, float(half[axis_b]) * 0.35)
    return (
        (-float(half[axis_a]) - pad_a, float(half[axis_a]) + pad_a),
        (-float(half[axis_b]) - pad_b, float(half[axis_b]) + pad_b),
    )


def _draw_box(ax: plt.Axes, half: np.ndarray, axis_a: int, axis_b: int) -> None:
    xs = [-half[axis_a], half[axis_a], half[axis_a], -half[axis_a], -half[axis_a]]
    ys = [-half[axis_b], -half[axis_b], half[axis_b], half[axis_b], -half[axis_b]]
    ax.plot(xs, ys, color="0.35", linewidth=1.2)
    lim_a, lim_b = _axis_limits(half, axis_a, axis_b)
    ax.set_xlim(*lim_a)
    ax.set_ylim(*lim_b)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="0.88", linewidth=0.6)


def _plot_cloud(
    variant: str,
    row: dict[str, str],
    half: np.ndarray,
    anchor: np.ndarray,
    points: np.ndarray,
    hands: np.ndarray,
    face_counts: dict[str, int],
    out_png: Path,
) -> None:
    colors = np.array(["tab:blue", "tab:red"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), dpi=150)
    views = [
        ("XY top", 0, 1, "x", "y"),
        ("XZ front", 0, 2, "x", "z"),
        ("YZ side", 1, 2, "y", "z"),
    ]
    for ax, (title, axis_a, axis_b, xlabel, ylabel) in zip(axes, views):
        _draw_box(ax, half, axis_a, axis_b)
        if len(points):
            for hand_idx in (0, 1):
                sel = hands == hand_idx
                if np.any(sel):
                    ax.scatter(
                        points[sel, axis_a],
                        points[sel, axis_b],
                        s=8,
                        alpha=0.38,
                        color=colors[hand_idx],
                        label=f"hand{hand_idx}",
                    )
        ax.scatter(
            [anchor[axis_a]],
            [anchor[axis_b]],
            marker="*",
            s=220,
            color="#f0c400",
            edgecolors="black",
            linewidths=0.8,
            label="support anchor",
            zorder=5,
        )
        ax.axhline(float(anchor[axis_b]), color="#d8ad00", linestyle="--", linewidth=0.8, alpha=0.65)
        ax.axvline(float(anchor[axis_a]), color="#d8ad00", linestyle="--", linewidth=0.8, alpha=0.65)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    axes[0].legend(loc="upper right", fontsize=8)
    counts_text = ", ".join(f"{face}:{face_counts.get(face, 0)}" for face in FACES)
    fig.suptitle(
        (
            f"{variant}\n"
            f"anchor={row['anchor_face']} local=[{anchor[0]:+.3f},{anchor[1]:+.3f},{anchor[2]:+.3f}] "
            f"review={row['anchor_face_review']} {row['anchor_face_review_reason']}"
        ),
        fontsize=10,
    )
    fig.text(
        0.5,
        0.015,
        (
            f"points={len(points)} | side_frac={row['anchor_side_face_frac']} | "
            f"margin={row['anchor_side_margin']} | z_frac={row['anchor_z_face_frac']} | {counts_text}"
        ),
        ha="center",
        fontsize=8,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.04, 1, 0.88))
    fig.savefig(out_png)
    plt.close(fig)


def _marker(label: str, point: np.ndarray, rgba: tuple[float, float, float, float]) -> Marker:
    bgr = tuple(int(255 * c) for c in (rgba[2], rgba[1], rgba[0]))
    return Marker(label, np.asarray(point, dtype=np.float64), rgba, bgr)


def _add_spheres(
    scene: mujoco.MjvScene,
    positions: list[np.ndarray],
    colors: list[tuple[float, float, float, float]],
    radius: float,
) -> None:
    for pos, rgba in zip(positions, colors):
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([radius, 0.0, 0.0], dtype=np.float64),
            np.asarray(pos, dtype=np.float64),
            np.eye(3, dtype=np.float64).reshape(-1),
            np.asarray(rgba, dtype=np.float32),
        )
        scene.ngeom += 1


def _top_camera(obj_pos: np.ndarray, distance: float) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = obj_pos
    cam.distance = distance
    cam.azimuth = 90.0
    cam.elevation = -90.0
    return cam


def _draw_legend(
    image: np.ndarray,
    title: str,
    panel: str,
    row: dict[str, str],
    frame_idx: int,
    frame_count: int,
) -> np.ndarray:
    out = image.copy()
    cv2.putText(out, f"{title} | {panel}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (35, 35, 35), 2)
    cv2.putText(
        out,
        f"frame {frame_idx + 1}/{frame_count} face={row['anchor_face']} review={row['anchor_face_review']}",
        (10, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (70, 70, 70),
        1,
    )
    cv2.rectangle(out, (12, 62), (28, 78), (0, 196, 240), thickness=-1)
    cv2.putText(out, "support anchor", (36, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 120, 180), 1)
    cv2.rectangle(out, (12, 84), (28, 100), (230, 90, 40), thickness=-1)
    cv2.putText(out, "ref active contact_pos", (36, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (170, 70, 35), 1)
    return out


def _render_video(
    variant: str,
    row: dict[str, str],
    model: mujoco.MjModel,
    qpos: np.ndarray,
    contact_points_world: list[list[np.ndarray]],
    anchor: np.ndarray,
    half: np.ndarray,
    out_mp4: Path,
    sheet_jpg: Path,
    width: int,
    height: int,
    fps: int,
) -> None:
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    renderer = mujoco.Renderer(model, height=height, width=width)
    data = mujoco.MjData(model)
    qadr = _object_qadr(model)
    obj_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    marker_radius = float(max(0.025, min(0.055, np.min(half[:2]) * 0.22)))
    contact_radius = marker_radius * 0.48
    top_distance = float(max(1.2, 4.2 * np.linalg.norm(half[:2])))
    frames: list[np.ndarray] = []
    for i, q in enumerate(qpos):
        data.qpos[: model.nq] = q[: model.nq]
        mujoco.mj_forward(model, data)
        obj_pos = data.xpos[obj_body].copy()
        obj_quat = q[qadr + 3 : qadr + 7].astype(np.float64)
        anchor_world = obj_pos + _quat_apply(obj_quat, anchor)
        contacts = contact_points_world[i]
        for camera_name, panel in (("track", "front"), (_top_camera(obj_pos, top_distance), "top")):
            renderer.update_scene(data, camera_name)
            _add_spheres(renderer.scene, [anchor_world], [(0.95, 0.78, 0.02, 1.0)], marker_radius)
            if contacts:
                _add_spheres(renderer.scene, contacts, [(0.05, 0.45, 1.0, 0.85)] * len(contacts), contact_radius)
            image = renderer.render()
            image = _draw_legend(image, variant, panel, row, i, len(qpos))
            if panel == "front":
                front = image
            else:
                top = image
        frames.append(np.concatenate([front, top], axis=1))
    renderer.close()
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_mp4), frames, fps=fps, codec="libx264", quality=8)
    sample_idx = np.linspace(0, len(frames) - 1, min(6, len(frames))).round().astype(int)
    thumbs = [cv2.resize(frames[i], (width, height), interpolation=cv2.INTER_AREA) for i in sample_idx]
    while len(thumbs) < 6:
        thumbs.append(np.zeros_like(thumbs[0]))
    sheet = np.concatenate(
        [np.concatenate(thumbs[:3], axis=1), np.concatenate(thumbs[3:6], axis=1)],
        axis=0,
    )
    cv2.imwrite(str(sheet_jpg), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def _write_montage(paths: list[Path], out_path: Path) -> None:
    images = [cv2.imread(str(path)) for path in paths if path.is_file()]
    if not images:
        return
    target_w = 720
    resized: list[np.ndarray] = []
    for image in images:
        h, w = image.shape[:2]
        scale = target_w / float(w)
        resized.append(cv2.resize(image, (target_w, int(h * scale)), interpolation=cv2.INTER_AREA))
    max_h = max(img.shape[0] for img in resized)
    padded: list[np.ndarray] = []
    for image in resized:
        if image.shape[0] < max_h:
            pad = np.full((max_h - image.shape[0], target_w, 3), 255, dtype=np.uint8)
            image = np.concatenate([image, pad], axis=0)
        padded.append(image)
    rows = [np.concatenate(padded[i : i + 2], axis=1) for i in range(0, len(padded), 2)]
    target_row_w = rows[0].shape[1]
    if rows[-1].shape[1] < target_row_w:
        rows[-1] = cv2.copyMakeBorder(
            rows[-1],
            0,
            0,
            0,
            target_row_w - rows[-1].shape[1],
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )
    montage = np.concatenate(rows, axis=0)
    cv2.imwrite(str(out_path), montage)


def render_all(
    cases: list[str],
    *,
    manifest_path: Path,
    out_dir: Path,
    width: int,
    height: int,
    fps: int,
    force: bool,
) -> None:
    manifest = _read_manifest(manifest_path)
    summary_rows: list[dict[str, object]] = []
    cloud_paths: list[Path] = []
    index_lines = ["# Candidate Anchor Visuals", ""]
    for variant in cases:
        if variant not in manifest:
            raise KeyError(f"{variant} not found in {manifest_path}")
        row = manifest[variant]
        anchor = np.array(
            [
                float(row["support_proxy_point_local_x"]),
                float(row["support_proxy_point_local_y"]),
                float(row["support_proxy_point_local_z"]),
            ],
            dtype=np.float64,
        )
        half = np.array(
            [float(row["object_half_x"]), float(row["object_half_y"]), float(row["object_half_z"])],
            dtype=np.float64,
        )
        model, qpos, contact_pos, active, _mask = _load_case(row)
        points, hands, per_frame_world = _contact_points_local(model, qpos, contact_pos, active)
        labels = [_face_label(point, half) for point in points]
        face_counts = {face: labels.count(face) for face in FACES}
        side_counts = {face: face_counts[face] for face in SIDE_FACES}
        selected_face_count = face_counts.get(row["anchor_face"], 0)
        top_face = max(face_counts, key=lambda face: face_counts[face]) if face_counts else ""
        side_face = max(side_counts, key=lambda face: side_counts[face]) if side_counts else ""
        side_rank = sorted(SIDE_FACES, key=lambda face: side_counts.get(face, 0), reverse=True)
        side_margin_count = side_counts.get(side_rank[0], 0) - side_counts.get(side_rank[1], 0)
        labels_np = np.asarray(labels)
        selected = labels_np == row["anchor_face"]
        if np.any(selected):
            selected_centroid = points[selected].mean(axis=0)
            selected_std = points[selected].std(axis=0)
            anchor_to_centroid = float(np.linalg.norm(anchor - selected_centroid))
            anchor_minus_centroid_z = float(anchor[2] - selected_centroid[2])
        else:
            selected_centroid = np.full(3, np.nan, dtype=np.float64)
            selected_std = np.full(3, np.nan, dtype=np.float64)
            anchor_to_centroid = float("nan")
            anchor_minus_centroid_z = float("nan")
        cloud_png = out_dir / f"{variant}_anchor_cloud.png"
        video_mp4 = out_dir / f"{variant}_anchor_motion.mp4"
        sheet_jpg = out_dir / f"{variant}_anchor_motion_sheet.jpg"
        if force or not cloud_png.is_file():
            _plot_cloud(variant, row, half, anchor, points, hands, face_counts, cloud_png)
        if force or not video_mp4.is_file() or not sheet_jpg.is_file():
            _render_video(
                variant,
                row,
                model,
                qpos,
                per_frame_world,
                anchor,
                half,
                video_mp4,
                sheet_jpg,
                width,
                height,
                fps,
            )
        cloud_paths.append(cloud_png)
        total = max(1, len(points))
        recomputed_side_frac = side_counts.get(side_face, 0) / total
        recomputed_margin = side_margin_count / total
        summary_rows.append(
            {
                "variant": variant,
                "source_task": row["source_task"],
                "anchor_face": row["anchor_face"],
                "anchor_local": f"[{anchor[0]:+.6f},{anchor[1]:+.6f},{anchor[2]:+.6f}]",
                "manifest_review": row["anchor_face_review"],
                "manifest_review_reason": row["anchor_face_review_reason"],
                "manifest_side_frac": row["anchor_side_face_frac"],
                "manifest_side_margin": row["anchor_side_margin"],
                "contact_points_used_manifest": row["contact_points_used"],
                "contact_points_recomputed": len(points),
                "recomputed_top_face": top_face,
                "recomputed_side_face": side_face,
                "recomputed_side_frac": f"{recomputed_side_frac:.8g}",
                "recomputed_side_margin": f"{recomputed_margin:.8g}",
                "selected_face_count": selected_face_count,
                "selected_face_centroid_x": f"{selected_centroid[0]:.8g}",
                "selected_face_centroid_y": f"{selected_centroid[1]:.8g}",
                "selected_face_centroid_z": f"{selected_centroid[2]:.8g}",
                "selected_face_std_x": f"{selected_std[0]:.8g}",
                "selected_face_std_y": f"{selected_std[1]:.8g}",
                "selected_face_std_z": f"{selected_std[2]:.8g}",
                "anchor_to_selected_face_centroid_m": f"{anchor_to_centroid:.8g}",
                "anchor_minus_selected_face_centroid_z_m": f"{anchor_minus_centroid_z:.8g}",
                "face_count_pos_x": face_counts.get("+x", 0),
                "face_count_neg_x": face_counts.get("-x", 0),
                "face_count_pos_y": face_counts.get("+y", 0),
                "face_count_neg_y": face_counts.get("-y", 0),
                "face_count_pos_z": face_counts.get("+z", 0),
                "face_count_neg_z": face_counts.get("-z", 0),
                "cloud_png": str(_rel(cloud_png)),
                "motion_mp4": str(_rel(video_mp4)),
                "motion_sheet": str(_rel(sheet_jpg)),
            }
        )
        index_lines.extend(
            [
                f"## {variant}",
                "",
                f"- cloud: `{_rel(cloud_png)}`",
                f"- motion: `{_rel(video_mp4)}`",
                f"- sheet: `{_rel(sheet_jpg)}`",
                (
                    f"- anchor `{row['anchor_face']}` local "
                    f"`[{anchor[0]:+.3f}, {anchor[1]:+.3f}, {anchor[2]:+.3f}]`, "
                    f"review=`{row['anchor_face_review']}` "
                    f"`{row['anchor_face_review_reason']}`"
                ),
                (
                    f"- recomputed points={len(points)}, top={top_face}, side={side_face}, "
                    f"side_frac={recomputed_side_frac:.3f}, margin={recomputed_margin:.3f}"
                ),
                "",
            ]
        )
        print(f"[OK] {variant}: cloud={_rel(cloud_png)} video={_rel(video_mp4)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "anchor_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    montage = out_dir / "candidate_anchor_cloud_montage.jpg"
    _write_montage(cloud_paths, montage)
    index_lines.insert(2, f"- summary: `{_rel(summary_csv)}`")
    index_lines.insert(3, f"- montage: `{_rel(montage)}`")
    index_lines.insert(4, "")
    (out_dir / "anchor_visual_eval.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    print(f"Summary saved: {_rel(summary_csv)}")
    print(f"Index saved: {_rel(out_dir / 'anchor_visual_eval.md')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", default=[], help="Variant to render. Defaults to candidates.json.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=432)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--candidates", type=Path, default=CANDIDATES)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cases = args.case or _read_candidates(args.candidates)
    render_all(
        cases,
        manifest_path=args.manifest,
        out_dir=args.out_dir,
        width=args.width,
        height=args.height,
        fps=args.fps,
        force=args.force,
    )


if __name__ == "__main__":
    main()
