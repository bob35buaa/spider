#!/usr/bin/env python3
"""Build E094 G1-handbox-aware external contact targets."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import Counter
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


REPO = Path(__file__).resolve().parents[4]
E093_DIR = REPO / "workspace/core4d/scripts/E093"
if str(E093_DIR) not in sys.path:
    sys.path.insert(0, str(E093_DIR))

from audit_contact_geometry import (  # noqa: E402
    EEF_OFFSET,
    HANDBOX_URDF,
    HANDS,
    HAND_PREFIX,
    face_stats,
    finite_stat,
    generate_raw_targets,
    name2id,
    parse_handbox_urdf,
    world_to_local,
)


E093_ROOT = REPO / "workspace/core4d/results/E093/contact_geometry"
DEFAULT_MANIFEST = E093_ROOT / "case_manifest.tsv"
OUT_ROOT = REPO / "workspace/core4d/results/E094/handbox_target_projection"
DEFAULT_CASE_IDS = [
    "box023_p2",
    "box004_083_p2",
    "box021_d003_029_p2",
    "box026_039_p2",
    "box026_135_p2",
]

SUMMARY_FIELDS = [
    "case_id",
    "object_group",
    "task",
    "role",
    "hand",
    "T",
    "raw_active_frac",
    "filled_frac",
    "old_wrist5_support_frac",
    "old_wrist5_inside_frac",
    "support_patch_support_frac",
    "support_patch_inside_frac",
    "reward_target_support_frac",
    "reward_target_inside_frac",
    "old_wrist5_to_raw_mean_m",
    "old_wrist5_to_raw_p90_m",
    "support_patch_to_raw_mean_m",
    "support_patch_to_raw_p90_m",
    "reward_delta_mean_m",
    "reward_delta_p90_m",
    "handbox_gap_to_patch_mean_m",
    "handbox_gap_to_patch_p90_abs_m",
    "support_patch_face_counts",
    "reward_target_face_counts",
    "gate_status",
    "gate_reasons",
    "target_npz",
]

PER_FRAME_FIELDS = [
    "case_id",
    "object_group",
    "task",
    "role",
    "frame",
    "raw_frame",
    "hand",
    "raw_active",
    "filled",
    "raw_local_x",
    "raw_local_y",
    "raw_local_z",
    "old_wrist5_local_x",
    "old_wrist5_local_y",
    "old_wrist5_local_z",
    "support_patch_local_x",
    "support_patch_local_y",
    "support_patch_local_z",
    "reward_target_local_x",
    "reward_target_local_y",
    "reward_target_local_z",
    "handbox_closest_local_x",
    "handbox_closest_local_y",
    "handbox_closest_local_z",
    "old_wrist5_to_raw_m",
    "support_patch_to_raw_m",
    "reward_delta_m",
    "handbox_gap_to_patch_m",
    "old_wrist5_face",
    "support_patch_face",
    "reward_target_face",
    "old_wrist5_support",
    "support_patch_support",
    "reward_target_support",
    "old_wrist5_inside",
    "support_patch_inside",
    "reward_target_inside",
]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True, default=jsonable) + "\n", encoding="utf-8")


def jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, Counter):
        return dict(obj)
    raise TypeError(type(obj).__name__)


def image_nonblank(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 2048:
        return False
    try:
        arr = plt.imread(path)
    except Exception:
        return False
    return bool(np.asarray(arr).std() > 1e-4)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def support_patch(raw_local: np.ndarray, half: np.ndarray, obj_mat: np.ndarray, inset: float) -> tuple[np.ndarray, str]:
    local_world_up = obj_mat.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    axis = int(np.argmax(np.abs(local_world_up)))
    sign = 1.0 if local_world_up[axis] >= 0.0 else -1.0
    patch = np.clip(raw_local, -half + inset, half - inset)
    patch[axis] = sign * half[axis]
    face = f"{'+' if sign > 0 else '-'}{'xyz'[axis]}"
    return patch, face


def face_stats_tol(local: np.ndarray, half: np.ndarray, obj_mat: np.ndarray, eps: float = 1e-5) -> dict[str, Any]:
    if not np.isfinite(local).all():
        return {"face": "", "inside": False, "support": False, "signed_dist": float("nan")}
    norm = local / half
    axis = int(np.argmax(np.abs(norm)))
    sign = 1.0 if local[axis] >= 0.0 else -1.0
    face = f"{'+' if sign > 0 else '-'}{'xyz'[axis]}"
    signed = sign * local[axis] - half[axis]
    local_world_up = obj_mat.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    top_axis = int(np.argmax(np.abs(local_world_up)))
    top_sign = 1.0 if local_world_up[top_axis] >= 0.0 else -1.0
    top_hit = axis == top_axis and sign == top_sign
    legacy_local_z_hit = axis == 2 and sign > 0
    inside = bool(np.all(np.abs(local) < (half - eps)))
    return {"face": face, "inside": inside, "support": bool(top_hit or legacy_local_z_hit), "signed_dist": float(signed)}


def closest_point_on_obb(point_world: np.ndarray, center_world: np.ndarray, mat_world: np.ndarray, half: np.ndarray) -> np.ndarray:
    local = mat_world.T @ (point_world - center_world)
    closest_local = np.clip(local, -half, half)
    return center_world + mat_world @ closest_local


def fill_nearest(values: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    filled = values.copy()
    filled_mask = np.zeros(valid.shape, dtype=bool)
    T = values.shape[0]
    for hi in range(values.shape[1]):
        valid_idx = np.where(valid[:, hi] & np.isfinite(values[:, hi]).all(axis=1))[0]
        if valid_idx.size == 0:
            raise ValueError(f"no valid projected target for {HANDS[hi]}")
        bad = ~np.isfinite(filled[:, hi]).all(axis=1)
        for t in np.where(bad)[0]:
            nearest = int(valid_idx[np.argmin(np.abs(valid_idx - t))])
            filled[t, hi] = filled[nearest, hi]
            filled_mask[t, hi] = True
        if valid_idx[0] > 0:
            filled_mask[: valid_idx[0], hi] = True
        if valid_idx[-1] < T - 1:
            filled_mask[valid_idx[-1] + 1 :, hi] = True
    return filled, filled_mask


def local_vec(row: dict[str, Any], prefix: str) -> np.ndarray:
    return np.array([row[f"{prefix}_x"], row[f"{prefix}_y"], row[f"{prefix}_z"]], dtype=np.float64)


def plot_object_local(case: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{case['case_id']}_{case['task']}_projection.png"
    half = case["half"]
    frames = np.arange(case["T"])
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), constrained_layout=True)
    panels = [("local X", "local Y", 0, 1), ("local X", "local Z", 0, 2), ("local Y", "local Z", 1, 2)]
    colors = {"left": "#cc3311", "right": "#0077bb"}
    for ax, (xlabel, ylabel, ix, iy) in zip(axes, panels):
        ax.add_patch(Rectangle((-half[ix], -half[iy]), 2 * half[ix], 2 * half[iy], fill=False, edgecolor="black", linewidth=1.4, label="collision box"))
        for hand in HANDS:
            arr = case["arrays"][hand]
            raw = arr["raw"]
            old = arr["old_wrist5"]
            patch = arr["support_patch"]
            reward = arr["reward_target"]
            active = arr["raw_active"]
            color = colors[hand]
            if active.any():
                ax.scatter(raw[active, ix], raw[active, iy], s=12, c=color, marker=".", alpha=0.30, label=f"{HAND_PREFIX[hand]} raw")
            ax.scatter(old[:, ix], old[:, iy], s=8, c=color, marker="x", alpha=0.18, label=f"{HAND_PREFIX[hand]} old wrist5")
            ax.scatter(patch[:, ix], patch[:, iy], s=13, c=color, marker="s", alpha=0.42, label=f"{HAND_PREFIX[hand]} support patch")
            ax.scatter(reward[:, ix], reward[:, iy], s=12, c=frames, cmap="viridis", marker="^", alpha=0.35, label=f"{HAND_PREFIX[hand]} reward target")
        lim = max(float(np.max(half)) * 1.9, 0.42)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel(f"{xlabel} (m)")
        ax.set_ylabel(f"{ylabel} (m)")
    axes[0].legend(loc="upper right", fontsize=7, ncols=2)
    fig.suptitle(f"{case['case_id']} | {case['task']} | E094 handbox-aware target projection")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_timeline(case: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{case['case_id']}_{case['task']}_projection_timeline.png"
    frames = np.arange(case["T"])
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True, constrained_layout=True)
    colors = {"left": "#cc3311", "right": "#0077bb"}
    for hand in HANDS:
        arr = case["arrays"][hand]
        color = colors[hand]
        label = HAND_PREFIX[hand]
        axes[0].plot(frames, arr["old_to_raw"], color=color, linestyle="-", alpha=0.65, label=f"{label} old->raw")
        axes[0].plot(frames, arr["patch_to_raw"], color=color, linestyle="--", alpha=0.75, label=f"{label} patch->raw")
        axes[1].plot(frames, arr["reward_delta"], color=color, alpha=0.8, label=f"{label} reward delta")
        axes[1].plot(frames, np.abs(arr["handbox_gap"]), color=color, linestyle=":", alpha=0.8, label=f"{label} |handbox gap|")
        axes[2].plot(frames, arr["old_support"].astype(float) + (0.0 if hand == "left" else 0.15), color=color, alpha=0.45, label=f"{label} old support")
        axes[2].plot(frames, arr["patch_support"].astype(float) + (1.25 if hand == "left" else 1.40), color=color, linestyle="--", alpha=0.75, label=f"{label} patch support")
        axes[3].plot(frames, arr["old_inside"].astype(float) + (0.0 if hand == "left" else 0.15), color=color, alpha=0.45, label=f"{label} old inside")
        axes[3].plot(frames, arr["reward_inside"].astype(float) + (1.25 if hand == "left" else 1.40), color=color, linestyle="--", alpha=0.75, label=f"{label} reward inside")
        axes[3].plot(frames, arr["filled"].astype(float) + (2.5 if hand == "left" else 2.65), color=color, linestyle=":", alpha=0.65, label=f"{label} filled")
    axes[0].set_title("Distance to raw contact")
    axes[0].set_ylabel("m")
    axes[1].set_title("Reward target move and handbox gap to support patch")
    axes[1].set_ylabel("m")
    axes[2].set_title("Support face indicators")
    axes[2].set_yticks([0, 0.15, 1.25, 1.40])
    axes[2].set_yticklabels(["L old", "R old", "L patch", "R patch"])
    axes[3].set_title("Inside/fill indicators")
    axes[3].set_yticks([0, 0.15, 1.25, 1.40, 2.5, 2.65])
    axes[3].set_yticklabels(["L old", "R old", "L rew", "R rew", "L fill", "R fill"])
    axes[3].set_xlabel("frame")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8, ncols=2)
    fig.suptitle(f"{case['case_id']} | {case['task']} | E094 projection timeline")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def summarize_hand(case: dict[str, Any], hand: str, target_npz: Path) -> dict[str, Any]:
    arr = case["arrays"][hand]
    raw_active = arr["raw_active"]
    old_support = arr["old_support"]
    old_inside = arr["old_inside"]
    patch_support = arr["patch_support"]
    patch_inside = arr["patch_inside"]
    reward_support = arr["reward_support"]
    reward_inside = arr["reward_inside"]
    reasons = []
    if float(arr["filled"].mean()) > 0.55:
        reasons.append("filled_gt55pct")
    if float(reward_inside.mean()) > 0.05:
        reasons.append("reward_inside_gt5pct")
    if finite_stat(arr["reward_delta"], "p90") > 0.35:
        reasons.append("reward_delta_p90_gt35cm")
    if float(reward_support.mean()) + 0.05 < float(old_support.mean()):
        reasons.append("reward_support_regression_gt5pct")
    if "FAIL" in str(case["role"]) and float(reward_support.mean()) < 0.50:
        reasons.append("fail_case_reward_support_lt50pct")
    status = "PASS" if not reasons else "REVIEW"
    return {
        "case_id": case["case_id"],
        "object_group": case["object_group"],
        "task": case["task"],
        "role": case["role"],
        "hand": hand,
        "T": case["T"],
        "raw_active_frac": round(float(raw_active.mean()), 6),
        "filled_frac": round(float(arr["filled"].mean()), 6),
        "old_wrist5_support_frac": round(float(old_support.mean()), 6),
        "old_wrist5_inside_frac": round(float(old_inside.mean()), 6),
        "support_patch_support_frac": round(float(patch_support.mean()), 6),
        "support_patch_inside_frac": round(float(patch_inside.mean()), 6),
        "reward_target_support_frac": round(float(reward_support.mean()), 6),
        "reward_target_inside_frac": round(float(reward_inside.mean()), 6),
        "old_wrist5_to_raw_mean_m": round(finite_stat(arr["old_to_raw"], "mean"), 6),
        "old_wrist5_to_raw_p90_m": round(finite_stat(arr["old_to_raw"], "p90"), 6),
        "support_patch_to_raw_mean_m": round(finite_stat(arr["patch_to_raw"], "mean"), 6),
        "support_patch_to_raw_p90_m": round(finite_stat(arr["patch_to_raw"], "p90"), 6),
        "reward_delta_mean_m": round(finite_stat(arr["reward_delta"], "mean"), 6),
        "reward_delta_p90_m": round(finite_stat(arr["reward_delta"], "p90"), 6),
        "handbox_gap_to_patch_mean_m": round(finite_stat(arr["handbox_gap"], "mean"), 6),
        "handbox_gap_to_patch_p90_abs_m": round(finite_stat(arr["handbox_gap"], "p90_abs"), 6),
        "support_patch_face_counts": json.dumps(Counter(arr["patch_face"]), sort_keys=True),
        "reward_target_face_counts": json.dumps(Counter(arr["reward_face"]), sort_keys=True),
        "gate_status": status,
        "gate_reasons": ",".join(reasons),
        "target_npz": rel(target_npz),
    }


def write_summary_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# E094 handbox-aware target projection summary",
        "",
        "| case | hand | old support | patch support | reward inside | reward delta p90 | handbox gap p90 | gate | reasons |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['case_id']}` | {row['hand']} | "
            f"{float(row['old_wrist5_support_frac']) * 100:.1f}% | "
            f"{float(row['support_patch_support_frac']) * 100:.1f}% | "
            f"{float(row['reward_target_inside_frac']) * 100:.1f}% | "
            f"{float(row['reward_delta_p90_m']):.3f}m | "
            f"{float(row['handbox_gap_to_patch_p90_abs_m']):.3f}m | "
            f"{row['gate_status']} | {row['gate_reasons']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compute_case(
    row: dict[str, str],
    out_root: Path,
    handbox: dict[str, dict[str, np.ndarray]],
    inset: float,
    sample_count: int,
    reward_mode: str,
    keep_old_distance: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    model = mujoco.MjModel.from_xml_path(row["scene_xml"])
    data = mujoco.MjData(model)
    traj_npz = np.load(row["trajectory_npz"], allow_pickle=True)
    qpos = traj_npz["qpos"].astype(np.float64)
    T = int(qpos.shape[0])
    obj_bid = name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    obj_gid = name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_collision")
    visual_gid = name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_visual")
    wrist_ids = {
        "left": name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link"),
        "right": name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link"),
    }
    half = model.geom_size[obj_gid].copy()
    raw = generate_raw_targets(row, half, model.geom_pos[visual_gid].copy(), model.geom_quat[visual_gid].copy(), T, sample_count)

    patch_raw = np.full((T, 2, 3), np.nan, dtype=np.float64)
    reward_raw = np.full((T, 2, 3), np.nan, dtype=np.float64)
    old_wrist5 = np.full((T, 2, 3), np.nan, dtype=np.float64)
    handbox_closest = np.full((T, 2, 3), np.nan, dtype=np.float64)
    valid = raw["qpos_mask"].copy()
    raw_active = raw["qpos_mask"].copy()
    raw_qpos = raw["raw_qpos"].copy()
    obj_mats = np.zeros((T, 3, 3), dtype=np.float64)
    obj_pos_arr = np.zeros((T, 3), dtype=np.float64)

    for t in range(T):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        obj_pos = data.xpos[obj_bid].copy()
        obj_mat = data.xmat[obj_bid].reshape(3, 3).copy()
        obj_pos_arr[t] = obj_pos
        obj_mats[t] = obj_mat
        for hi, hand in enumerate(HANDS):
            wrist_pos = data.xpos[wrist_ids[hand]].copy()
            wrist_mat = data.xmat[wrist_ids[hand]].reshape(3, 3).copy()
            wrist5_world = wrist_pos + wrist_mat @ EEF_OFFSET
            old_wrist5[t, hi] = world_to_local(obj_pos, obj_mat, wrist5_world)
            raw_local = raw_qpos[t, hi]
            if reward_mode == "adaptive_support" and not (raw_active[t, hi] and np.isfinite(raw_local).all()):
                patch_raw[t, hi] = old_wrist5[t, hi]
                reward_raw[t, hi] = old_wrist5[t, hi]
                handbox_closest[t, hi] = old_wrist5[t, hi]
                valid[t, hi] = True
                continue
            if not (raw_active[t, hi] and np.isfinite(raw_local).all()):
                continue
            old_s = face_stats_tol(old_wrist5[t, hi], half, obj_mat)
            old_to_raw = float(np.linalg.norm(old_wrist5[t, hi] - raw_local))
            patch_local, _face = support_patch(raw_local, half, obj_mat, inset)
            patch_world = obj_pos + obj_mat @ patch_local
            hb = handbox[hand]
            hb_center = wrist_pos + wrist_mat @ hb["offset"]
            hb_mat = wrist_mat @ hb["rot"]
            closest_world = closest_point_on_obb(patch_world, hb_center, hb_mat, hb["half"])
            if reward_mode == "support_patch":
                reward_world = patch_world
            elif reward_mode == "handbox_compensated":
                reward_world = patch_world + (wrist5_world - closest_world)
            elif reward_mode == "adaptive_support":
                keep_old = (not old_s["inside"]) and (bool(old_s["support"]) or old_to_raw <= keep_old_distance)
                reward_world = wrist5_world if keep_old else patch_world
            else:
                raise ValueError(f"unsupported reward_mode={reward_mode}")
            patch_raw[t, hi] = patch_local
            reward_raw[t, hi] = world_to_local(obj_pos, obj_mat, reward_world)
            handbox_closest[t, hi] = world_to_local(obj_pos, obj_mat, closest_world)

    patch_filled, filled_patch = fill_nearest(patch_raw, valid)
    reward_filled, filled_reward = fill_nearest(reward_raw, valid)
    closest_filled, _ = fill_nearest(handbox_closest, valid)
    filled = filled_patch | filled_reward | ~valid

    target_dir = out_root / "targets"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_npz = target_dir / f"{row['case_id']}_{row['task']}_{reward_mode}_targets.npz"
    np.savez(
        target_npz,
        spider_contact_target_object_local=reward_filled.astype(np.float32),
        eval_contact_target_object_local=reward_filled.astype(np.float32),
        support_patch_object_local=patch_filled.astype(np.float32),
        raw_contact_target_object_local=raw_qpos.astype(np.float32),
        old_wrist5_object_local=old_wrist5.astype(np.float32),
        handbox_closest_object_local=closest_filled.astype(np.float32),
        raw_active_mask=raw_active.astype(np.bool_),
        filled_mask=filled.astype(np.bool_),
        case_id=np.array(row["case_id"]),
        task=np.array(row["task"]),
        projection_mode=np.array(f"support_face_preserve_tangent_{reward_mode}"),
        contact_hdmi_eef_offset=EEF_OFFSET.astype(np.float32),
    )

    per_frame: list[dict[str, Any]] = []
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for hi, hand in enumerate(HANDS):
        old_stats = []
        patch_stats = []
        reward_stats = []
        old_to_raw = np.full(T, np.nan)
        patch_to_raw = np.full(T, np.nan)
        reward_delta = np.full(T, np.nan)
        handbox_gap = np.full(T, np.nan)
        for t in range(T):
            obj_mat = obj_mats[t]
            raw_local = raw_qpos[t, hi]
            old_local = old_wrist5[t, hi]
            patch_local = patch_filled[t, hi]
            reward_local = reward_filled[t, hi]
            closest_local = closest_filled[t, hi]
            old_s = face_stats_tol(old_local, half, obj_mat)
            patch_s = face_stats_tol(patch_local, half, obj_mat)
            reward_s = face_stats_tol(reward_local, half, obj_mat)
            old_stats.append(old_s)
            patch_stats.append(patch_s)
            reward_stats.append(reward_s)
            if raw_active[t, hi] and np.isfinite(raw_local).all():
                old_to_raw[t] = float(np.linalg.norm(old_local - raw_local))
                patch_to_raw[t] = float(np.linalg.norm(patch_local - raw_local))
            reward_delta[t] = float(np.linalg.norm(reward_local - old_local))
            handbox_gap[t] = float(np.linalg.norm(closest_local - patch_local))

            item = {
                "case_id": row["case_id"],
                "object_group": row["object_group"],
                "task": row["task"],
                "role": row["role"],
                "frame": t,
                "raw_frame": int(raw["raw_idx_for_qpos"][t]),
                "hand": hand,
                "raw_active": bool(raw_active[t, hi]),
                "filled": bool(filled[t, hi]),
                "old_wrist5_to_raw_m": old_to_raw[t],
                "support_patch_to_raw_m": patch_to_raw[t],
                "reward_delta_m": reward_delta[t],
                "handbox_gap_to_patch_m": handbox_gap[t],
                "old_wrist5_face": old_s["face"],
                "support_patch_face": patch_s["face"],
                "reward_target_face": reward_s["face"],
                "old_wrist5_support": old_s["support"],
                "support_patch_support": patch_s["support"],
                "reward_target_support": reward_s["support"],
                "old_wrist5_inside": old_s["inside"],
                "support_patch_inside": patch_s["inside"],
                "reward_target_inside": reward_s["inside"],
            }
            for prefix, vec in (
                ("raw_local", raw_local),
                ("old_wrist5_local", old_local),
                ("support_patch_local", patch_local),
                ("reward_target_local", reward_local),
                ("handbox_closest_local", closest_local),
            ):
                item[f"{prefix}_x"] = float(vec[0]) if np.isfinite(vec[0]) else float("nan")
                item[f"{prefix}_y"] = float(vec[1]) if np.isfinite(vec[1]) else float("nan")
                item[f"{prefix}_z"] = float(vec[2]) if np.isfinite(vec[2]) else float("nan")
            per_frame.append(item)

        arrays[hand] = {
            "raw": raw_qpos[:, hi],
            "old_wrist5": old_wrist5[:, hi],
            "support_patch": patch_filled[:, hi],
            "reward_target": reward_filled[:, hi],
            "handbox_closest": closest_filled[:, hi],
            "raw_active": raw_active[:, hi],
            "filled": filled[:, hi],
            "old_to_raw": old_to_raw,
            "patch_to_raw": patch_to_raw,
            "reward_delta": reward_delta,
            "handbox_gap": handbox_gap,
            "old_support": np.array([s["support"] for s in old_stats], dtype=bool),
            "patch_support": np.array([s["support"] for s in patch_stats], dtype=bool),
            "reward_support": np.array([s["support"] for s in reward_stats], dtype=bool),
            "old_inside": np.array([s["inside"] for s in old_stats], dtype=bool),
            "patch_inside": np.array([s["inside"] for s in patch_stats], dtype=bool),
            "reward_inside": np.array([s["inside"] for s in reward_stats], dtype=bool),
            "patch_face": np.array([s["face"] for s in patch_stats], dtype=object),
            "reward_face": np.array([s["face"] for s in reward_stats], dtype=object),
        }

    case = {
        "case_id": row["case_id"],
        "object_group": row["object_group"],
        "task": row["task"],
        "role": row["role"],
        "reward_mode": reward_mode,
        "T": T,
        "half": half,
        "target_npz": target_npz,
        "arrays": arrays,
    }
    plot_object_local(case, out_root / "visuals/object_local")
    plot_timeline(case, out_root / "visuals/timeline")
    summary = [summarize_hand(case, hand, target_npz) for hand in HANDS]
    return case, summary, per_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--case-ids", nargs="*", default=DEFAULT_CASE_IDS)
    parser.add_argument("--sample-count", type=int, default=6000)
    parser.add_argument("--surface-inset", type=float, default=0.015)
    parser.add_argument("--reward-mode", choices=["adaptive_support", "support_patch", "handbox_compensated"], default="adaptive_support")
    parser.add_argument("--keep-old-distance", type=float, default=0.30)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.force and args.out_root.exists():
        shutil.rmtree(args.out_root)
    args.out_root.mkdir(parents=True, exist_ok=True)

    wanted = set(args.case_ids)
    manifest = [r for r in read_tsv(args.manifest) if str(r.get("ready", "")).lower() == "true" and r["case_id"] in wanted]
    missing = sorted(wanted - {r["case_id"] for r in manifest})
    if missing:
        raise SystemExit(f"Missing ready E093 manifest rows for {missing}")

    handbox = parse_handbox_urdf(HANDBOX_URDF)
    summary_rows: list[dict[str, Any]] = []
    per_frame_rows: list[dict[str, Any]] = []
    case_meta = []
    for row in manifest:
        print(f"[E094-proj] {row['case_id']} {row['task']}")
        case, summary, per_frame = compute_case(row, args.out_root, handbox, args.surface_inset, args.sample_count, args.reward_mode, args.keep_old_distance)
        summary_rows.extend(summary)
        per_frame_rows.extend(per_frame)
        case_meta.append({"case_id": case["case_id"], "task": case["task"], "reward_mode": case["reward_mode"], "target_npz": rel(case["target_npz"]), "T": case["T"]})

    write_tsv(args.out_root / "projection_summary.csv", summary_rows, SUMMARY_FIELDS)
    write_tsv(args.out_root / "per_frame_projection.csv", per_frame_rows, PER_FRAME_FIELDS)
    write_json(args.out_root / "projection_summary.json", summary_rows)
    write_json(args.out_root / "case_targets.json", case_meta)
    write_summary_md(args.out_root / "projection_summary.md", summary_rows)

    pngs = sorted((args.out_root / "visuals").glob("**/*.png"))
    qc = {
        "num_cases": len(manifest),
        "num_summary_rows": len(summary_rows),
        "num_per_frame_rows": len(per_frame_rows),
        "num_png": len(pngs),
        "num_png_nonblank": sum(image_nonblank(path) for path in pngs),
        "reward_mode": args.reward_mode,
        "keep_old_distance_m": args.keep_old_distance,
    }
    write_json(args.out_root / "projection_qc.json", qc)
    print(f"[E094-proj] wrote {rel(args.out_root)}")
    print(f"[E094-proj] qc={qc}")


if __name__ == "__main__":
    main()
