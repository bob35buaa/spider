#!/usr/bin/env python3
"""Generate E091 D005b visual-QC artifacts for medium-box candidates."""

from __future__ import annotations

import argparse
import csv
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import mujoco
import numpy as np


DEFAULT_V2_ROOT = Path("/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2")
DEFAULT_SPIDER_REPO = Path("/home/ubuntu/Workspace/spider")
EEF_OFFSET = np.array([0.05, 0.0, 0.0], dtype=np.float64)


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


def parse_object_collision(scene_xml: Path) -> tuple[np.ndarray, np.ndarray]:
    tree = ET.parse(scene_xml)
    root = tree.getroot()
    for body in root.iter("body"):
        if body.attrib.get("name") != "object":
            continue
        for geom in body.iter("geom"):
            name = geom.attrib.get("name", "").lower()
            if geom.attrib.get("type") == "box" and "collision" in name:
                size = np.array([float(x) for x in geom.attrib["size"].split()], dtype=np.float64)
                pos = np.array([float(x) for x in geom.attrib.get("pos", "0 0 0").split()], dtype=np.float64)
                return size, pos
    raise ValueError(f"Could not find object collision box in {scene_xml}")


def task_info(task_dir: Path) -> dict[str, str]:
    path = task_dir / "task_info.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: str(v) for k, v in data.items() if isinstance(v, (str, int, float))}


def hand_frame_stats(loc: np.ndarray, half: np.ndarray, obj_mats: np.ndarray) -> dict[str, np.ndarray]:
    norm = loc / half
    ax = np.argmax(np.abs(norm), axis=1)
    raw_sgn = np.take_along_axis(norm, ax[:, None], axis=1).squeeze(1)
    sgn = np.sign(raw_sgn)
    sgn[sgn == 0.0] = 1.0
    chosen = np.take_along_axis(loc, ax[:, None], axis=1).squeeze(1)
    signed = sgn * chosen - half[ax]

    local_world_up = np.einsum("tji,j->ti", obj_mats, np.array([0.0, 0.0, 1.0]))
    top_ax = np.argmax(np.abs(local_world_up), axis=1)
    top_sgn = np.sign(np.take_along_axis(local_world_up, top_ax[:, None], axis=1).squeeze(1))
    top_sgn[top_sgn == 0.0] = 1.0
    top_hit = (ax == top_ax) & (sgn == top_sgn)
    legacy_local_z_hit = (ax == 2) & (sgn > 0)
    support_hit = top_hit | legacy_local_z_hit
    inside = np.all(np.abs(loc) < half, axis=1)

    return {
        "axis": ax,
        "sign": sgn,
        "signed_dist": signed,
        "inside": inside,
        "top_hit": top_hit,
        "legacy_local_z_hit": legacy_local_z_hit,
        "support_hit": support_hit,
    }


def compute_case(task_dir: Path) -> dict[str, Any]:
    scene = task_dir / "scene.xml"
    traj = task_dir / "0" / "trajectory_kinematic.npz"
    if not scene.is_file() or not traj.is_file():
        raise FileNotFoundError(f"Missing scene or trajectory for {task_dir.name}")

    half, collision_pos = parse_object_collision(scene)
    npz = np.load(traj, allow_pickle=True)
    qpos = np.asarray(npz["qpos"], dtype=np.float64)
    contact = np.asarray(npz["contact"], dtype=np.float64) if "contact" in npz else None
    contact_pos = np.asarray(npz["contact_pos"], dtype=np.float64) if "contact_pos" in npz else None
    t_count = int(qpos.shape[0])

    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    bid_l = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
    bid_r = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")

    obj_mats = np.zeros((t_count, 3, 3), dtype=np.float64)
    l_loc = np.zeros((t_count, 3), dtype=np.float64)
    r_loc = np.zeros((t_count, 3), dtype=np.float64)
    contact_loc = np.full((t_count, 2, 3), np.nan, dtype=np.float64)
    l_world_z = np.zeros(t_count, dtype=np.float64)
    r_world_z = np.zeros(t_count, dtype=np.float64)
    obj_z = np.zeros(t_count, dtype=np.float64)

    for t in range(t_count):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        obj_pos = data.xpos[obj_bid].copy()
        obj_mat = data.xmat[obj_bid].reshape(3, 3).copy()
        obj_mats[t] = obj_mat
        obj_z[t] = obj_pos[2]
        for bid, out, z_out in ((bid_l, l_loc, l_world_z), (bid_r, r_loc, r_world_z)):
            wrist_pos = data.xpos[bid].copy()
            wrist_mat = data.xmat[bid].reshape(3, 3).copy()
            eef_world = wrist_pos + wrist_mat @ EEF_OFFSET
            out[t] = obj_mat.T @ (eef_world - obj_pos) - collision_pos
            z_out[t] = eef_world[2]
        if contact_pos is not None:
            for h in range(min(2, contact_pos.shape[1])):
                contact_loc[t, h] = obj_mat.T @ (contact_pos[t, h] - obj_pos) - collision_pos

    return {
        "task": task_dir.name,
        "task_dir": task_dir,
        "scene": scene,
        "trajectory": traj,
        "qpos": qpos,
        "T": t_count,
        "half": half,
        "collision_pos": collision_pos,
        "L_loc": l_loc,
        "R_loc": r_loc,
        "contact": contact,
        "contact_loc": contact_loc,
        "pelvis_z": qpos[:, 2],
        "obj_z": obj_z,
        "L_world_z": l_world_z,
        "R_world_z": r_world_z,
        "L_frame": hand_frame_stats(l_loc, half, obj_mats),
        "R_frame": hand_frame_stats(r_loc, half, obj_mats),
    }


def plot_overlay(case: dict[str, Any], gate_row: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    half = case["half"]
    t = np.arange(case["T"])
    out_path = out_dir / f"{case['task']}_object_local_overlay.png"

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    panels = [("local X", "local Y", 0, 1), ("local X", "local Z", 0, 2), ("local Y", "local Z", 1, 2)]
    for ax, (xlabel, ylabel, ix, iy) in zip(axes, panels):
        ax.add_patch(
            Rectangle(
                (-half[ix], -half[iy]),
                2.0 * half[ix],
                2.0 * half[iy],
                fill=False,
                edgecolor="black",
                linewidth=1.5,
                label="collision box",
            )
        )
        ax.scatter(case["L_loc"][:, ix], case["L_loc"][:, iy], s=16, c=t, cmap="Reds", alpha=0.65, label="L eef")
        ax.scatter(case["R_loc"][:, ix], case["R_loc"][:, iy], s=16, c=t, cmap="Blues", alpha=0.65, label="R eef")
        c_loc = case["contact_loc"]
        if np.isfinite(c_loc).any():
            ax.scatter(c_loc[:, 0, ix], c_loc[:, 0, iy], s=12, c="#ff9900", marker="x", alpha=0.45, label="L contact")
            ax.scatter(c_loc[:, 1, ix], c_loc[:, 1, iy], s=12, c="#008855", marker="x", alpha=0.45, label="R contact")
        lim = max(float(np.max(half)) * 1.75, 0.28)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel(f"{xlabel} (m)")
        ax.set_ylabel(f"{ylabel} (m)")
    axes[0].legend(loc="upper right", fontsize=8)
    decision = "PASS" if gate_row.get("gate_pass") else "REJECT"
    reasons = ",".join(gate_row.get("gate_reject_reasons") or [])
    fig.suptitle(
        f"{case['task']} | object-local EEF/contact overlay | {decision}"
        + (f" ({reasons})" if reasons else "")
        + f" | half={np.round(half, 3).tolist()}"
    )
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_timeline(case: dict[str, Any], gate_row: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = np.arange(case["T"])
    out_path = out_dir / f"{case['task']}_d005b_timeline.png"
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True, constrained_layout=True)

    axes[0].plot(frames, case["pelvis_z"], color="#333333", label="pelvis z")
    axes[0].plot(frames, case["obj_z"], color="#888888", label="object center z")
    axes[0].plot(frames, case["L_world_z"], color="#cc3311", label="L eef z")
    axes[0].plot(frames, case["R_world_z"], color="#4477aa", label="R eef z")
    axes[0].axhline(0.60, color="#222222", linestyle="--", linewidth=1.0, label="pelvis min gate")
    axes[0].set_ylabel("world z (m)")
    axes[0].legend(loc="upper right", ncol=2)
    axes[0].set_title("Height signals")

    axes[1].plot(frames, case["L_frame"]["signed_dist"], color="#cc3311", label="L signed dist")
    axes[1].plot(frames, case["R_frame"]["signed_dist"], color="#4477aa", label="R signed dist")
    axes[1].axhline(0.03, color="#222222", linestyle="--", linewidth=1.0, label="3cm gate")
    axes[1].axhline(0.0, color="#999999", linestyle=":", linewidth=1.0)
    axes[1].set_ylabel("distance (m)")
    axes[1].legend(loc="upper right")
    axes[1].set_title("Signed distance to selected object face")

    axes[2].plot(frames, case["L_frame"]["inside"].astype(int), color="#cc3311", label="L inside")
    axes[2].plot(frames, case["R_frame"]["inside"].astype(int) + 1.2, color="#4477aa", label="R inside")
    axes[2].plot(frames, case["L_frame"]["support_hit"].astype(int) + 2.4, color="#ee7733", label="L support face")
    axes[2].plot(frames, case["R_frame"]["support_hit"].astype(int) + 3.6, color="#009988", label="R support face")
    axes[2].set_yticks([0, 1.2, 2.4, 3.6])
    axes[2].set_yticklabels(["L inside", "R inside", "L support", "R support"])
    axes[2].set_ylim(-0.2, 4.8)
    axes[2].legend(loc="upper right", ncol=2)
    axes[2].set_title("Gate classifications per frame")

    if case["contact"] is not None:
        contact = case["contact"]
        axes[3].plot(frames, contact[:, 0], color="#cc3311", label="L contact")
        axes[3].plot(frames, contact[:, 1] + 1.2, color="#4477aa", label="R contact")
        axes[3].set_yticks([0, 1.2])
        axes[3].set_yticklabels(["L contact", "R contact"])
    axes[3].set_xlabel("frame")
    axes[3].set_title("SPIDER contact mask")
    axes[3].legend(loc="upper right")

    decision = "PASS" if gate_row.get("gate_pass") else "REJECT"
    fig.suptitle(f"{case['task']} | D005b timeline | {decision}")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def render_keyframes(case: dict[str, Any], out_dir: Path, camera: str = "track2") -> tuple[Path, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{case['task']}_keyframes.png"
    idxs = np.linspace(0, case["T"] - 1, num=min(6, case["T"]), dtype=int)
    try:
        model = mujoco.MjModel.from_xml_path(str(case["scene"]))
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, width=480, height=360)
        images = []
        for idx in idxs:
            data.qpos[:] = case["qpos"][idx]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            images.append(renderer.render())
        renderer.close()
    except Exception as exc:
        return out_path, f"render_failed: {exc}"

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for ax, img, idx in zip(axes.flat, images, idxs):
        ax.imshow(img)
        ax.set_title(f"frame {int(idx)}")
        ax.axis("off")
    for ax in axes.flat[len(images) :]:
        ax.axis("off")
    fig.suptitle(f"{case['task']} MuJoCo keyframes ({camera})")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path, "ok"


def flatten_gate_rows(gate_data: dict[str, Any], task_to_paths: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, row in gate_data["results"].items():
        task = row.get("task", label)
        if "error" in row:
            rows.append({"task": task, "gate_pass": False, "error": row["error"]})
            continue
        info = task_to_paths.get(task, {})
        sequence = info.get("sequence", "")
        if not sequence and (info.get("date") or info.get("seq")):
            sequence = f"{info.get('date', '')}/{info.get('seq', '')}".strip("/")
        l = row["L"]
        r = row["R"]
        rows.append(
            {
                "task": task,
                "object_name": info.get("object_name", ""),
                "sequence": sequence,
                "person": info.get("person", ""),
                "T": row["T"],
                "gate_pass": bool(row["gate_pass"]),
                "reject_reasons": ",".join(row.get("gate_reject_reasons") or []),
                "pelvis_z_min": round(float(row["pelvis_z_min"]), 6),
                "L_inside_frac": round(float(l["inside_box_frac"]), 6),
                "R_inside_frac": round(float(r["inside_box_frac"]), 6),
                "L_signed_dist_mean_m": round(float(l["signed_dist_mean_m"]), 6),
                "R_signed_dist_mean_m": round(float(r["signed_dist_mean_m"]), 6),
                "L_support_face_frac": round(float(l["support_face_frac"]), 6),
                "R_support_face_frac": round(float(r["support_face_frac"]), 6),
                "support_face_frac_either": round(max(float(l["support_face_frac"]), float(r["support_face_frac"])), 6),
                "L_wrist_below_pelvis_gap_m": round(float(l["wrist_below_pelvis_gap_m"]), 6),
                "R_wrist_below_pelvis_gap_m": round(float(r["wrist_below_pelvis_gap_m"]), 6),
                "overlay_png": info.get("overlay_png", ""),
                "timeline_png": info.get("timeline_png", ""),
                "keyframes_png": info.get("keyframes_png", ""),
            }
        )
    return rows


def plot_dashboard(rows: list[dict[str, Any]], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for r in rows if "error" not in r]
    paths: list[Path] = []
    if not rows:
        return paths

    labels = [str(r["task"]).replace("e091_", "") for r in rows]
    colors = ["#228833" if r["gate_pass"] else "#cc3311" for r in rows]
    support = np.array([float(r["support_face_frac_either"]) for r in rows])
    inside = np.array([max(float(r["L_inside_frac"]), float(r["R_inside_frac"])) for r in rows])
    pelvis = np.array([float(r["pelvis_z_min"]) for r in rows])

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    ax.scatter(inside * 100.0, support * 100.0, s=90, c=colors, edgecolor="black")
    for x, y, label in zip(inside * 100.0, support * 100.0, labels):
        ax.annotate(label, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.axvline(10.0, color="#222222", linestyle="--", linewidth=1)
    ax.axhline(30.0, color="#222222", linestyle="--", linewidth=1)
    ax.set_xlabel("max wrist inside box (%)")
    ax.set_ylabel("max support-face fraction (%)")
    ax.set_title("D005b inside vs support")
    path = out_dir / "d005b_inside_vs_support.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    ax.scatter(support * 100.0, pelvis, s=90, c=colors, edgecolor="black")
    for x, y, label in zip(support * 100.0, pelvis, labels):
        ax.annotate(label, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.axvline(30.0, color="#222222", linestyle="--", linewidth=1)
    ax.axhline(0.60, color="#222222", linestyle="--", linewidth=1)
    ax.set_xlabel("max support-face fraction (%)")
    ax.set_ylabel("pelvis z min (m)")
    ax.set_title("D005b pelvis vs support")
    path = out_dir / "d005b_pelvis_vs_support.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)
    return paths


def write_markdown(root: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    path = root / "results/visual_qc/summary.md"
    lines = [
        "# E091 Medium-Box Visual QC",
        "",
        f"- D005b rows: `{len(rows)}`",
        f"- D005b pass: `{sum(1 for r in rows if r.get('gate_pass'))}`",
        f"- Nonblank PNGs: `{summary['nonblank_png_count']}/{summary['png_count']}`",
        "",
        "| task | decision | T | inside max | support max | pelvis min | reasons |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        if "error" in row:
            lines.append(f"| `{row['task']}` | ERROR | | | | | {row['error']} |")
            continue
        decision = "PASS" if row["gate_pass"] else "REJECT"
        inside = max(float(row["L_inside_frac"]), float(row["R_inside_frac"]))
        lines.append(
            f"| `{row['task']}` | {decision} | {row['T']} | "
            f"{inside:.3f} | {float(row['support_face_frac_either']):.3f} | "
            f"{float(row['pelvis_z_min']):.3f} | {row['reject_reasons']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-root", type=Path, default=DEFAULT_V2_ROOT)
    parser.add_argument("--spider-repo", type=Path, default=DEFAULT_SPIDER_REPO)
    parser.add_argument(
        "--d005b-json",
        type=Path,
        default=DEFAULT_V2_ROOT / "results/d005b_g1_feasibility/d005b_summary.json",
    )
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--render-camera", default="track2")
    parser.add_argument("--skip-render", action="store_true")
    args = parser.parse_args()

    gate_data = json.loads(args.d005b_json.read_text(encoding="utf-8"))
    tasks = args.tasks or [row.get("task", label) for label, row in gate_data["results"].items()]
    tasks_dir = args.spider_repo / "example_datasets/processed/core4d/unitree_g1/humanoid_object"

    visual_dir = args.v2_root / "visualizations/d005b"
    frame_dir = args.v2_root / "visualizations/keyframes"
    task_to_paths: dict[str, dict[str, str]] = {}
    png_paths: list[Path] = []
    render_status: dict[str, str] = {}

    for task in tasks:
        if task not in {r.get("task", label) for label, r in gate_data["results"].items()}:
            continue
        task_dir = tasks_dir / task
        case = compute_case(task_dir)
        gate_row = next(r for label, r in gate_data["results"].items() if r.get("task", label) == task)
        overlay = plot_overlay(case, gate_row, visual_dir)
        timeline = plot_timeline(case, gate_row, visual_dir)
        png_paths.extend([overlay, timeline])
        keyframes = Path("")
        status = "skipped"
        if not args.skip_render:
            keyframes, status = render_keyframes(case, frame_dir, args.render_camera)
            if keyframes:
                png_paths.append(keyframes)
        info = task_info(task_dir)
        info.update(
            {
                "overlay_png": str(overlay),
                "timeline_png": str(timeline),
                "keyframes_png": str(keyframes) if keyframes else "",
            }
        )
        task_to_paths[task] = info
        render_status[task] = status

    rows = flatten_gate_rows(gate_data, task_to_paths)
    fields = [
        "task",
        "object_name",
        "sequence",
        "person",
        "T",
        "gate_pass",
        "reject_reasons",
        "pelvis_z_min",
        "L_inside_frac",
        "R_inside_frac",
        "L_signed_dist_mean_m",
        "R_signed_dist_mean_m",
        "L_support_face_frac",
        "R_support_face_frac",
        "support_face_frac_either",
        "L_wrist_below_pelvis_gap_m",
        "R_wrist_below_pelvis_gap_m",
        "overlay_png",
        "timeline_png",
        "keyframes_png",
    ]
    write_tsv(args.v2_root / "results/d005b_g1_feasibility/d005b_summary.tsv", rows, fields)
    dashboard_paths = plot_dashboard(rows, args.v2_root / "visualizations/dashboard")
    png_paths.extend(dashboard_paths)

    png_records = [
        {"path": str(path), "exists": path.is_file(), "nonblank": image_nonblank(path), "bytes": path.stat().st_size if path.is_file() else 0}
        for path in png_paths
        if str(path)
    ]
    summary = {
        "stage": "E091 medium-box visual QC",
        "d005b_json": str(args.d005b_json),
        "tasks": tasks,
        "d005b_rows": len(rows),
        "d005b_pass_count": sum(1 for r in rows if r.get("gate_pass")),
        "png_count": len(png_records),
        "nonblank_png_count": sum(1 for r in png_records if r["nonblank"]),
        "render_status": render_status,
        "pngs": png_records,
        "dashboard_pngs": [str(p) for p in dashboard_paths],
    }
    write_json(args.v2_root / "results/visual_qc/summary.json", summary)
    write_tsv(
        args.v2_root / "results/visual_qc/png_manifest.tsv",
        png_records,
        ["path", "exists", "nonblank", "bytes"],
    )
    write_markdown(args.v2_root, rows, summary)
    print(f"Wrote {len(rows)} D005b rows")
    print(f"PNGs nonblank: {summary['nonblank_png_count']}/{summary['png_count']}")
    print(f"D005b pass count: {summary['d005b_pass_count']}")


if __name__ == "__main__":
    main()
