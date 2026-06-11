#!/usr/bin/env python3
"""Audit raw contact targets against G1 wrist/sphere/box hand proxies."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import xml.etree.ElementTree as ET
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
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OUT_ROOT = REPO / "workspace/core4d/results/E093/contact_geometry"
DEFAULT_MANIFEST = OUT_ROOT / "case_manifest.tsv"
HANDBOX_URDF = Path("/home/ubuntu/Workspace/holosoma/src/holosoma/holosoma/data/robots/g1/main_mesh_collision_handbox_m5.urdf")

HANDS = ("left", "right")
HAND_PREFIX = {"left": "L", "right": "R"}
HAND_RANGES = {
    "left": np.arange(4700, 5500, dtype=np.int64),
    "right": np.arange(7500, 8150, dtype=np.int64),
}
FINGERTIP_IDS = {
    "left": np.array([5361, 4933, 5058, 5169, 5286], dtype=np.int64),
    "right": np.array([8079, 7669, 7794, 7905, 8022], dtype=np.int64),
}
PERSONS = ("person1", "person2")
EEF_OFFSET = np.array([0.05, 0.0, 0.0], dtype=np.float64)

# Historical HDMI 3-box patch values: MuJoCo half-extents in wrist_yaw_link frame.
THREEBOX = {
    "left": [
        ("lh", np.array([0.02, 0.0, 0.0]), np.array([0.05, 0.025, 0.025]), np.eye(3)),
        ("lh2", np.array([0.09, 0.0, 0.0]), np.array([0.05, 0.01, 0.05]), np.eye(3)),
        (
            "lh3",
            np.array([0.15, -0.01, 0.0]),
            np.array([0.025, 0.01, 0.05]),
            Rotation.from_quat([0.0, 0.0, -0.198669, 0.980067]).as_matrix(),
        ),
    ],
    "right": [
        ("rh", np.array([0.02, 0.0, 0.0]), np.array([0.05, 0.025, 0.025]), np.eye(3)),
        ("rh2", np.array([0.09, 0.0, 0.0]), np.array([0.05, 0.01, 0.05]), np.eye(3)),
        (
            "rh3",
            np.array([0.15, 0.01, 0.0]),
            np.array([0.025, 0.01, 0.05]),
            Rotation.from_quat([0.0, 0.0, 0.198669, 0.980067]).as_matrix(),
        ),
    ],
}


SUMMARY_FIELDS = [
    "case_id",
    "object_group",
    "task",
    "role",
    "hand",
    "T",
    "raw_active_frac",
    "wrist5_to_raw_mean_m",
    "wrist5_to_raw_median_m",
    "wrist5_to_raw_p90_m",
    "wrist5_to_raw_max_m",
    "wrist5_vs_contact_pos_mean_m",
    "raw_support_frac",
    "wrist5_support_frac",
    "wrist5_inside_frac",
    "raw_inside_frac",
    "sphere_surface_gap_mean_m",
    "sphere_surface_gap_p90_abs_m",
    "threebox_surface_gap_mean_m",
    "threebox_surface_gap_p90_abs_m",
    "handbox_surface_gap_mean_m",
    "handbox_surface_gap_p90_abs_m",
    "best_proxy_by_distance",
    "raw_face_counts",
    "wrist5_face_counts",
    "bad_proxy_flags",
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
    "raw_local_x",
    "raw_local_y",
    "raw_local_z",
    "contact_pos_local_x",
    "contact_pos_local_y",
    "contact_pos_local_z",
    "wrist_local_x",
    "wrist_local_y",
    "wrist_local_z",
    "wrist5_local_x",
    "wrist5_local_y",
    "wrist5_local_z",
    "sphere_center_local_x",
    "sphere_center_local_y",
    "sphere_center_local_z",
    "contact_site_local_x",
    "contact_site_local_y",
    "contact_site_local_z",
    "handbox_center_local_x",
    "handbox_center_local_y",
    "handbox_center_local_z",
    "threebox_best_center_local_x",
    "threebox_best_center_local_y",
    "threebox_best_center_local_z",
    "wrist5_to_raw_m",
    "wrist5_vs_contact_pos_m",
    "sphere_surface_gap_m",
    "threebox_surface_gap_m",
    "handbox_surface_gap_m",
    "raw_face",
    "wrist5_face",
    "raw_support",
    "wrist5_support",
    "raw_inside",
    "wrist5_inside",
    "threebox_best_name",
    "best_proxy_by_distance",
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


def finite_mean(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def finite_stat(values: np.ndarray, stat: str) -> float:
    values = values[np.isfinite(values)]
    if not values.size:
        return float("nan")
    if stat == "mean":
        return float(values.mean())
    if stat == "median":
        return float(np.median(values))
    if stat == "p90":
        return float(np.quantile(values, 0.90))
    if stat == "p90_abs":
        return float(np.quantile(np.abs(values), 0.90))
    if stat == "max":
        return float(values.max())
    raise ValueError(stat)


def quat_wxyz_to_mat(quat: np.ndarray) -> np.ndarray:
    return Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()


def name2id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise KeyError(f"missing {name}")
    return int(idx)


def parse_handbox_urdf(path: Path) -> dict[str, dict[str, np.ndarray]]:
    root = ET.parse(path).getroot()
    out: dict[str, dict[str, np.ndarray]] = {}
    link_sizes: dict[str, np.ndarray] = {}
    for link in root.findall("link"):
        name = link.attrib.get("name", "")
        if name not in ("left_handbox_link", "right_handbox_link"):
            continue
        box = link.find("./collision/geometry/box")
        if box is None:
            raise ValueError(f"missing handbox box geometry for {name}")
        link_sizes[name] = np.array([float(x) for x in box.attrib["size"].split()], dtype=np.float64)
    for joint in root.findall("joint"):
        child = joint.find("child")
        origin = joint.find("origin")
        if child is None or origin is None:
            continue
        child_name = child.attrib.get("link", "")
        if child_name not in link_sizes:
            continue
        side = "left" if child_name.startswith("left") else "right"
        offset = np.array([float(x) for x in origin.attrib["xyz"].split()], dtype=np.float64)
        out[side] = {
            "offset": offset,
            "half": link_sizes[child_name] * 0.5,
            "rot": np.eye(3, dtype=np.float64),
            "source_link": child_name,
        }
    if set(out) != {"left", "right"}:
        raise ValueError(f"incomplete handbox parse from {path}: {sorted(out)}")
    return out


def load_person(seq_dir: Path, person: str) -> dict[str, Any]:
    path = seq_dir / f"{person}_poses.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=True)["arr_0"].item()


def project_to_box_surface(local: np.ndarray, half: np.ndarray) -> np.ndarray:
    abs_local = np.abs(local)
    outside = abs_local - half
    if np.any(outside > 0.0):
        axis = int(np.argmax(outside))
    else:
        axis = int(np.argmin(half - abs_local))
    projected = np.clip(local, -half, half)
    projected[axis] = (1.0 if local[axis] >= 0.0 else -1.0) * half[axis]
    return projected


def resize_index(source_len: int, target_len: int) -> np.ndarray:
    if source_len == target_len:
        return np.arange(target_len, dtype=np.int64)
    return np.round(np.linspace(0, source_len - 1, target_len)).astype(np.int64)


def qpos_raw_mapping(mask_npz: np.lib.npyio.NpzFile, person_idx: int, target_len: int) -> tuple[np.ndarray, np.ndarray, str]:
    if "spider_contact_mask_3cm" in mask_npz:
        source = mask_npz["spider_contact_mask_3cm"][:, person_idx, :].astype(bool)
        idx = resize_index(source.shape[0], target_len)
        trim_start = int(mask_npz["trim_start"]) if "trim_start" in mask_npz else 0
        raw_idx = trim_start + idx
        return source[idx], raw_idx.astype(np.int64), "spider_contact_mask_3cm"
    if "eval_contact_mask_3cm" in mask_npz:
        source = mask_npz["eval_contact_mask_3cm"][:, person_idx, :].astype(bool)
        idx = resize_index(source.shape[0], target_len)
        if "eval_raw_idx" in mask_npz:
            raw_idx = mask_npz["eval_raw_idx"].astype(np.int64)[idx]
        else:
            raw_idx = idx
        return source[idx], raw_idx.astype(np.int64), "eval_contact_mask_3cm"
    source = mask_npz["raw_contact_mask_3cm"][:, person_idx, :].astype(bool)
    idx = resize_index(source.shape[0], target_len)
    return source[idx], idx.astype(np.int64), "raw_contact_mask_3cm"


def generate_raw_targets(
    row: dict[str, str],
    half: np.ndarray,
    visual_pos: np.ndarray,
    visual_quat: np.ndarray,
    target_len: int,
    sample_count: int,
) -> dict[str, Any]:
    person_idx = int(row["person_idx"])
    mask_path = Path(row["mask_path"])
    audit = json.loads(Path(row["audit_summary"]).read_text(encoding="utf-8"))
    seq_dir = Path(audit["seq_dir"])
    mesh_path = Path(audit["mesh"])
    threshold = float(audit.get("threshold_m", 0.03))
    sample_count = int(audit.get("sample_count", sample_count) or sample_count)
    sample_count = min(sample_count, 30000)

    mask_npz = np.load(mask_path, allow_pickle=True)
    raw_mask = mask_npz["raw_contact_mask_3cm"][:, person_idx, :].astype(bool)
    qpos_mask, raw_idx_for_qpos, mask_key = qpos_raw_mapping(mask_npz, person_idx, target_len)

    mesh_rot = quat_wxyz_to_mat(visual_quat)
    mesh = trimesh.load(mesh_path, process=False)
    deterministic_offset = sum((i + 1) * ord(ch) for i, ch in enumerate(row["case_id"])) % 1000
    rng = np.random.default_rng(9300 + deterministic_offset)
    surface_points, _ = trimesh.sample.sample_surface(mesh, sample_count, seed=rng)
    obj_poses = np.load(seq_dir / "smooth_objposes.npy")
    person = load_person(seq_dir, PERSONS[person_idx])
    vertices = person["vertices"]
    n_raw = min(raw_mask.shape[0], obj_poses.shape[0], vertices.shape[0])

    raw_target = np.full((raw_mask.shape[0], 2, 3), np.nan, dtype=np.float64)
    raw_source = np.full((raw_mask.shape[0], 2), "", dtype=object)
    for raw_f in range(n_raw):
        r_raw = obj_poses[raw_f, :3, :3]
        t_raw = obj_poses[raw_f, :3, 3]
        obj_world = surface_points @ r_raw.T + t_raw
        tree = cKDTree(obj_world)
        verts = vertices[raw_f]
        for hi, hand in enumerate(HANDS):
            if not raw_mask[raw_f, hi]:
                continue
            hand_verts = verts[HAND_RANGES[hand]]
            dists, nn_idx = tree.query(hand_verts, k=1)
            close = dists < threshold
            if close.any():
                target_raw_local = surface_points[nn_idx[close]].mean(axis=0)
                source = "close_centroid"
            else:
                tip_verts = verts[FINGERTIP_IDS[hand]]
                tip_dists, tip_nn_idx = tree.query(tip_verts, k=1)
                target_raw_local = surface_points[tip_nn_idx[int(np.argmin(tip_dists))]]
                source = "tip_fallback"
            # Match E085: raw mesh local -> MuJoCo object body frame via the
            # object_visual fixed transform, then project to collision box.
            visual_body = (target_raw_local - visual_pos) @ mesh_rot
            raw_target[raw_f, hi] = project_to_box_surface(visual_body, half)
            raw_source[raw_f, hi] = source

    raw_qpos = np.full((target_len, 2, 3), np.nan, dtype=np.float64)
    valid_raw_idx = np.clip(raw_idx_for_qpos, 0, raw_target.shape[0] - 1)
    raw_qpos[:] = raw_target[valid_raw_idx]
    return {
        "raw_qpos": raw_qpos,
        "qpos_mask": qpos_mask,
        "raw_idx_for_qpos": raw_idx_for_qpos,
        "mask_key": mask_key,
        "raw_source": raw_source,
        "sample_count": sample_count,
    }


def face_stats(local: np.ndarray, half: np.ndarray, obj_mat: np.ndarray) -> dict[str, Any]:
    if not np.isfinite(local).all():
        return {"face": "", "inside": False, "support": False, "signed_dist": float("nan")}
    norm = local / half
    axis = int(np.argmax(np.abs(norm)))
    sign = 1.0 if local[axis] >= 0.0 else -1.0
    face = f"{'+' if sign > 0 else '-'}{'xyz'[axis]}"
    signed = sign * local[axis] - half[axis]
    local_world_up = obj_mat.T @ np.array([0.0, 0.0, 1.0])
    top_axis = int(np.argmax(np.abs(local_world_up)))
    top_sign = 1.0 if local_world_up[top_axis] >= 0.0 else -1.0
    top_hit = axis == top_axis and sign == top_sign
    legacy_local_z_hit = axis == 2 and sign > 0
    inside = bool(np.all(np.abs(local) < half))
    return {
        "face": face,
        "inside": inside,
        "support": bool(top_hit or legacy_local_z_hit),
        "signed_dist": float(signed),
    }


def world_to_local(obj_pos: np.ndarray, obj_mat: np.ndarray, point: np.ndarray) -> np.ndarray:
    return obj_mat.T @ (point - obj_pos)


def obb_signed_gap(point_world: np.ndarray, center_world: np.ndarray, mat_world: np.ndarray, half: np.ndarray) -> float:
    p = mat_world.T @ (point_world - center_world)
    q = np.abs(p) - half
    outside = np.maximum(q, 0.0)
    outside_dist = float(np.linalg.norm(outside))
    if outside_dist > 0.0:
        return outside_dist
    return float(np.max(q))


def obb_corners(center_world: np.ndarray, mat_world: np.ndarray, half: np.ndarray) -> np.ndarray:
    corners = np.array(
        [[sx, sy, sz] for sx in (-half[0], half[0]) for sy in (-half[1], half[1]) for sz in (-half[2], half[2])],
        dtype=np.float64,
    )
    return corners @ mat_world.T + center_world


def draw_box_projection(ax: plt.Axes, corners_local: np.ndarray, ix: int, iy: int, color: str, alpha: float, label: str | None = None) -> None:
    edges = [(0, 1), (0, 2), (0, 4), (3, 1), (3, 2), (3, 7), (5, 1), (5, 4), (5, 7), (6, 2), (6, 4), (6, 7)]
    for ei, (a, b) in enumerate(edges):
        ax.plot(
            [corners_local[a, ix], corners_local[b, ix]],
            [corners_local[a, iy], corners_local[b, iy]],
            color=color,
            alpha=alpha,
            linewidth=1.0,
            label=label if ei == 0 else None,
        )


def image_nonblank(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 2048:
        return False
    try:
        arr = plt.imread(path)
    except Exception:
        return False
    return bool(np.asarray(arr).std() > 1e-4)


def compute_case(row: dict[str, str], handbox: dict[str, dict[str, np.ndarray]], sample_count: int) -> dict[str, Any]:
    scene = Path(row["scene_xml"])
    traj = Path(row["trajectory_npz"])
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    qpos = np.load(traj, allow_pickle=True)["qpos"].astype(np.float64)
    traj_npz = np.load(traj, allow_pickle=True)
    contact_pos = traj_npz["contact_pos"].astype(np.float64) if "contact_pos" in traj_npz else None

    obj_bid = name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    obj_gid = name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_collision")
    visual_gid = name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_visual")
    half = model.geom_size[obj_gid].copy()
    visual_pos = model.geom_pos[visual_gid].copy()
    visual_quat = model.geom_quat[visual_gid].copy()
    wrist_ids = {
        "left": name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link"),
        "right": name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link"),
    }
    site_ids = {
        "left": name2id(model, mujoco.mjtObj.mjOBJ_SITE, "contact_left_hand"),
        "right": name2id(model, mujoco.mjtObj.mjOBJ_SITE, "contact_right_hand"),
    }
    geom_ids = {"left": name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "lh"), "right": name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "rh")}
    raw = generate_raw_targets(row, half, visual_pos, visual_quat, qpos.shape[0], sample_count)

    per_frame: list[dict[str, Any]] = []
    proxy_corners: dict[str, dict[str, list[np.ndarray]]] = {"handbox": {"left": [], "right": []}, "threebox": {"left": [], "right": []}}

    for t in range(qpos.shape[0]):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        obj_pos = data.xpos[obj_bid].copy()
        obj_mat = data.xmat[obj_bid].reshape(3, 3).copy()
        raw_idx = int(raw["raw_idx_for_qpos"][t])

        for hi, hand in enumerate(HANDS):
            wrist_pos = data.xpos[wrist_ids[hand]].copy()
            wrist_mat = data.xmat[wrist_ids[hand]].reshape(3, 3).copy()
            wrist5_world = wrist_pos + wrist_mat @ EEF_OFFSET
            sphere_center = data.geom_xpos[geom_ids[hand]].copy()
            sphere_radius = float(model.geom_size[geom_ids[hand], 0])
            site_world = data.site_xpos[site_ids[hand]].copy()
            raw_local = raw["raw_qpos"][t, hi]
            raw_world = obj_pos + obj_mat @ raw_local if np.isfinite(raw_local).all() else np.full(3, np.nan)
            raw_active = bool(raw["qpos_mask"][t, hi] and np.isfinite(raw_local).all())

            hb = handbox[hand]
            handbox_center_world = wrist_pos + wrist_mat @ hb["offset"]
            handbox_mat_world = wrist_mat @ hb["rot"]
            handbox_gap = obb_signed_gap(raw_world, handbox_center_world, handbox_mat_world, hb["half"]) if raw_active else float("nan")
            handbox_corners = obb_corners(handbox_center_world, handbox_mat_world, hb["half"])

            three_best_name = ""
            three_best_gap = float("nan")
            three_best_center = np.full(3, np.nan)
            three_best_corners = None
            if raw_active:
                best_abs = float("inf")
                for name, center_off, box_half, box_rot in THREEBOX[hand]:
                    center_world = wrist_pos + wrist_mat @ center_off
                    mat_world = wrist_mat @ box_rot
                    gap = obb_signed_gap(raw_world, center_world, mat_world, box_half)
                    if abs(gap) < best_abs:
                        best_abs = abs(gap)
                        three_best_gap = gap
                        three_best_name = name
                        three_best_center = center_world
                        three_best_corners = obb_corners(center_world, mat_world, box_half)
            if three_best_corners is None:
                name, center_off, box_half, box_rot = THREEBOX[hand][-1]
                three_best_name = name
                three_best_center = wrist_pos + wrist_mat @ center_off
                three_best_corners = obb_corners(three_best_center, wrist_mat @ box_rot, box_half)

            contact_local = np.full(3, np.nan)
            if contact_pos is not None and contact_pos.ndim == 3 and hi < contact_pos.shape[1]:
                contact_local = world_to_local(obj_pos, obj_mat, contact_pos[t, hi])

            wrist_local = world_to_local(obj_pos, obj_mat, wrist_pos)
            wrist5_local = world_to_local(obj_pos, obj_mat, wrist5_world)
            sphere_local = world_to_local(obj_pos, obj_mat, sphere_center)
            site_local = world_to_local(obj_pos, obj_mat, site_world)
            handbox_local = world_to_local(obj_pos, obj_mat, handbox_center_world)
            threebox_local = world_to_local(obj_pos, obj_mat, three_best_center)

            raw_stats = face_stats(raw_local, half, obj_mat)
            wrist5_stats = face_stats(wrist5_local, half, obj_mat)
            wrist5_to_raw = float(np.linalg.norm(wrist5_world - raw_world)) if raw_active else float("nan")
            wrist5_vs_contact = float(np.linalg.norm(wrist5_local - contact_local)) if np.isfinite(contact_local).all() else float("nan")
            sphere_gap = float(np.linalg.norm(raw_world - sphere_center) - sphere_radius) if raw_active else float("nan")
            candidates = {
                "wrist5_point": wrist5_to_raw,
                "sphere_surface": abs(sphere_gap) if np.isfinite(sphere_gap) else float("nan"),
                "threebox_surface": abs(three_best_gap) if np.isfinite(three_best_gap) else float("nan"),
                "handbox_surface": abs(handbox_gap) if np.isfinite(handbox_gap) else float("nan"),
            }
            finite_candidates = {k: v for k, v in candidates.items() if np.isfinite(v)}
            best_proxy = min(finite_candidates, key=finite_candidates.get) if finite_candidates else ""

            row_out: dict[str, Any] = {
                "case_id": row["case_id"],
                "object_group": row["object_group"],
                "task": row["task"],
                "role": row["role"],
                "frame": t,
                "raw_frame": raw_idx,
                "hand": hand,
                "raw_active": raw_active,
                "wrist5_to_raw_m": wrist5_to_raw,
                "wrist5_vs_contact_pos_m": wrist5_vs_contact,
                "sphere_surface_gap_m": sphere_gap,
                "threebox_surface_gap_m": three_best_gap,
                "handbox_surface_gap_m": handbox_gap,
                "raw_face": raw_stats["face"],
                "wrist5_face": wrist5_stats["face"],
                "raw_support": bool(raw_stats["support"]),
                "wrist5_support": bool(wrist5_stats["support"]),
                "raw_inside": bool(raw_stats["inside"]),
                "wrist5_inside": bool(wrist5_stats["inside"]),
                "threebox_best_name": three_best_name,
                "best_proxy_by_distance": best_proxy,
            }
            for prefix, vec in [
                ("raw_local", raw_local),
                ("contact_pos_local", contact_local),
                ("wrist_local", wrist_local),
                ("wrist5_local", wrist5_local),
                ("sphere_center_local", sphere_local),
                ("contact_site_local", site_local),
                ("handbox_center_local", handbox_local),
                ("threebox_best_center_local", threebox_local),
            ]:
                row_out[f"{prefix}_x"] = float(vec[0])
                row_out[f"{prefix}_y"] = float(vec[1])
                row_out[f"{prefix}_z"] = float(vec[2])
            per_frame.append(row_out)

            if t in np.linspace(0, qpos.shape[0] - 1, num=min(6, qpos.shape[0]), dtype=int):
                proxy_corners["handbox"][hand].append(np.stack([world_to_local(obj_pos, obj_mat, c) for c in handbox_corners]))
                proxy_corners["threebox"][hand].append(np.stack([world_to_local(obj_pos, obj_mat, c) for c in three_best_corners]))

    return {
        "row": row,
        "qpos": qpos,
        "half": half,
        "per_frame": per_frame,
        "raw_meta": {k: v for k, v in raw.items() if k not in ("raw_qpos", "qpos_mask")},
        "proxy_corners": proxy_corners,
    }


def summarize_case(case: dict[str, Any]) -> list[dict[str, Any]]:
    rows = case["per_frame"]
    out: list[dict[str, Any]] = []
    for hand in HANDS:
        hrows = [r for r in rows if r["hand"] == hand]
        active = np.array([bool(r["raw_active"]) for r in hrows], dtype=bool)
        vals = lambda key: np.array([float(r[key]) for r in hrows if r["raw_active"]], dtype=np.float64)
        wrist5 = vals("wrist5_to_raw_m")
        contact_pos = vals("wrist5_vs_contact_pos_m")
        sphere = vals("sphere_surface_gap_m")
        three = vals("threebox_surface_gap_m")
        handbox = vals("handbox_surface_gap_m")
        raw_support = np.array([bool(r["raw_support"]) for r in hrows if r["raw_active"]], dtype=bool)
        raw_inside = np.array([bool(r["raw_inside"]) for r in hrows if r["raw_active"]], dtype=bool)
        wrist_support = np.array([bool(r["wrist5_support"]) for r in hrows], dtype=bool)
        wrist_inside = np.array([bool(r["wrist5_inside"]) for r in hrows], dtype=bool)
        raw_faces = Counter(r["raw_face"] for r in hrows if r["raw_active"] and r["raw_face"])
        wrist_faces = Counter(r["wrist5_face"] for r in hrows if r["wrist5_face"])
        best = Counter(r["best_proxy_by_distance"] for r in hrows if r["raw_active"] and r["best_proxy_by_distance"])
        flags: list[str] = []
        if finite_stat(wrist5, "mean") > 0.20:
            flags.append("wrist5_raw_delta_gt20cm")
        if float(wrist_inside.mean()) > 0.10:
            flags.append("wrist5_inside_gt10pct")
        if float(wrist_support.mean()) < 0.30:
            flags.append("wrist5_support_lt30pct")
        if finite_stat(sphere, "p90_abs") > 0.10:
            flags.append("sphere_gap_p90_gt10cm")
        if finite_stat(handbox, "p90_abs") > 0.10:
            flags.append("handbox_gap_p90_gt10cm")
        row0 = case["row"]
        out.append(
            {
                "case_id": row0["case_id"],
                "object_group": row0["object_group"],
                "task": row0["task"],
                "role": row0["role"],
                "hand": hand,
                "T": len(hrows),
                "raw_active_frac": round(float(active.mean()), 6),
                "wrist5_to_raw_mean_m": round(finite_stat(wrist5, "mean"), 6),
                "wrist5_to_raw_median_m": round(finite_stat(wrist5, "median"), 6),
                "wrist5_to_raw_p90_m": round(finite_stat(wrist5, "p90"), 6),
                "wrist5_to_raw_max_m": round(finite_stat(wrist5, "max"), 6),
                "wrist5_vs_contact_pos_mean_m": round(finite_stat(contact_pos, "mean"), 6),
                "raw_support_frac": round(float(raw_support.mean()) if raw_support.size else float("nan"), 6),
                "wrist5_support_frac": round(float(wrist_support.mean()), 6),
                "wrist5_inside_frac": round(float(wrist_inside.mean()), 6),
                "raw_inside_frac": round(float(raw_inside.mean()) if raw_inside.size else float("nan"), 6),
                "sphere_surface_gap_mean_m": round(finite_stat(sphere, "mean"), 6),
                "sphere_surface_gap_p90_abs_m": round(finite_stat(sphere, "p90_abs"), 6),
                "threebox_surface_gap_mean_m": round(finite_stat(three, "mean"), 6),
                "threebox_surface_gap_p90_abs_m": round(finite_stat(three, "p90_abs"), 6),
                "handbox_surface_gap_mean_m": round(finite_stat(handbox, "mean"), 6),
                "handbox_surface_gap_p90_abs_m": round(finite_stat(handbox, "p90_abs"), 6),
                "best_proxy_by_distance": dict(best),
                "raw_face_counts": dict(raw_faces),
                "wrist5_face_counts": dict(wrist_faces),
                "bad_proxy_flags": ",".join(flags),
            }
        )
    return out


def plot_overlay(case: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    row = case["row"]
    half = case["half"]
    rows = case["per_frame"]
    out_path = out_dir / f"{row['case_id']}_{row['task']}_overlay.png"
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)
    panels = [("local X", "local Y", 0, 1), ("local X", "local Z", 0, 2), ("local Y", "local Z", 1, 2)]
    tvals = np.array([r["frame"] for r in rows if r["hand"] == "left"], dtype=float)
    for ax, (xlabel, ylabel, ix, iy) in zip(axes, panels):
        ax.add_patch(Rectangle((-half[ix], -half[iy]), 2 * half[ix], 2 * half[iy], fill=False, edgecolor="black", linewidth=1.4, label="object collision"))
        for hand, color_raw, color_wrist, marker in [
            ("left", "#ee7733", "#cc3311", "o"),
            ("right", "#009988", "#4477aa", "s"),
        ]:
            hrows = [r for r in rows if r["hand"] == hand]
            frames = np.array([r["frame"] for r in hrows], dtype=float)
            raw_x = np.array([r[f"raw_local_{'xyz'[ix]}"] for r in hrows], dtype=float)
            raw_y = np.array([r[f"raw_local_{'xyz'[iy]}"] for r in hrows], dtype=float)
            raw_active = np.array([bool(r["raw_active"]) for r in hrows], dtype=bool)
            wrist_x = np.array([r[f"wrist5_local_{'xyz'[ix]}"] for r in hrows], dtype=float)
            wrist_y = np.array([r[f"wrist5_local_{'xyz'[iy]}"] for r in hrows], dtype=float)
            sphere_x = np.array([r[f"sphere_center_local_{'xyz'[ix]}"] for r in hrows], dtype=float)
            sphere_y = np.array([r[f"sphere_center_local_{'xyz'[iy]}"] for r in hrows], dtype=float)
            hb_x = np.array([r[f"handbox_center_local_{'xyz'[ix]}"] for r in hrows], dtype=float)
            hb_y = np.array([r[f"handbox_center_local_{'xyz'[iy]}"] for r in hrows], dtype=float)
            ax.scatter(wrist_x, wrist_y, s=13, c=frames, cmap="Reds" if hand == "left" else "Blues", alpha=0.45, marker=marker, label=f"{HAND_PREFIX[hand]} wrist+5cm")
            ax.scatter(raw_x[raw_active], raw_y[raw_active], s=24, c=color_raw, alpha=0.75, marker="x", label=f"{HAND_PREFIX[hand]} raw")
            ax.scatter(sphere_x, sphere_y, s=8, c=color_wrist, alpha=0.22, marker=".", label=f"{HAND_PREFIX[hand]} sphere center")
            ax.scatter(hb_x[:: max(len(hb_x) // 20, 1)], hb_y[:: max(len(hb_y) // 20, 1)], s=18, c=color_wrist, alpha=0.45, marker="^", label=f"{HAND_PREFIX[hand]} handbox center")
            for corners in case["proxy_corners"]["handbox"][hand]:
                draw_box_projection(ax, corners, ix, iy, color_wrist, 0.18, None)
            for corners in case["proxy_corners"]["threebox"][hand]:
                draw_box_projection(ax, corners, ix, iy, "#aa4499", 0.15, None)
        lim = max(float(np.max(half)) * 1.65, 0.28)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel(f"{xlabel} (m)")
        ax.set_ylabel(f"{ylabel} (m)")
    axes[0].legend(loc="upper right", fontsize=7, ncol=1)
    fig.suptitle(f"{row['case_id']} | {row['task']} | raw vs wrist+5cm vs sphere/3-box/handbox")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_timeline(case: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    row = case["row"]
    rows = case["per_frame"]
    out_path = out_dir / f"{row['case_id']}_{row['task']}_timeline.png"
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True, constrained_layout=True)
    for hand, color in [("left", "#cc3311"), ("right", "#4477aa")]:
        hrows = [r for r in rows if r["hand"] == hand]
        frames = np.array([r["frame"] for r in hrows], dtype=int)
        label = HAND_PREFIX[hand]
        axes[0].plot(frames, [r["wrist5_to_raw_m"] for r in hrows], color=color, label=f"{label} wrist5->raw")
        axes[1].plot(frames, [r["sphere_surface_gap_m"] for r in hrows], color=color, label=f"{label} sphere gap")
        axes[1].plot(frames, [r["handbox_surface_gap_m"] for r in hrows], color=color, linestyle="--", label=f"{label} handbox gap")
        axes[1].plot(frames, [r["threebox_surface_gap_m"] for r in hrows], color=color, linestyle=":", label=f"{label} 3-box gap")
        axes[2].plot(frames, np.array([r["wrist5_inside"] for r in hrows], dtype=float) + (0 if hand == "left" else 1.2), color=color, label=f"{label} wrist inside")
        axes[2].plot(frames, np.array([r["wrist5_support"] for r in hrows], dtype=float) + (2.4 if hand == "left" else 3.6), color=color, linestyle="--", label=f"{label} wrist support")
        axes[3].plot(frames, np.array([r["raw_active"] for r in hrows], dtype=float) + (0 if hand == "left" else 1.2), color=color, label=f"{label} raw active")
        axes[3].plot(frames, np.array([r["raw_support"] for r in hrows], dtype=float) + (2.4 if hand == "left" else 3.6), color=color, linestyle="--", label=f"{label} raw support")
    axes[0].axhline(0.20, color="#222222", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("distance (m)")
    axes[0].set_title("Current reward target distance to raw contact")
    axes[1].axhline(0.0, color="#222222", linestyle=":", linewidth=1.0)
    axes[1].set_ylabel("signed gap (m)")
    axes[1].set_title("Proxy surface signed gap to raw contact")
    axes[2].set_yticks([0, 1.2, 2.4, 3.6])
    axes[2].set_yticklabels(["L inside", "R inside", "L support", "R support"])
    axes[2].set_title("wrist+5cm object-face classification")
    axes[3].set_yticks([0, 1.2, 2.4, 3.6])
    axes[3].set_yticklabels(["L active", "R active", "L raw support", "R raw support"])
    axes[3].set_title("raw contact mask/support")
    axes[3].set_xlabel("frame")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.suptitle(f"{row['case_id']} | {row['task']}")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_dashboard(summary_rows: list[dict[str, Any]], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    labels = [f"{r['case_id']}:{HAND_PREFIX[r['hand']]}" for r in summary_rows]
    x = np.arange(len(labels))
    wrist = np.array([float(r["wrist5_to_raw_mean_m"]) for r in summary_rows])
    sphere = np.array([float(r["sphere_surface_gap_p90_abs_m"]) for r in summary_rows])
    handbox = np.array([float(r["handbox_surface_gap_p90_abs_m"]) for r in summary_rows])
    three = np.array([float(r["threebox_surface_gap_p90_abs_m"]) for r in summary_rows])

    fig, ax = plt.subplots(figsize=(15, 6), constrained_layout=True)
    ax.bar(x - 0.3, wrist, width=0.2, label="wrist5->raw mean")
    ax.bar(x - 0.1, sphere, width=0.2, label="sphere p90 |gap|")
    ax.bar(x + 0.1, handbox, width=0.2, label="handbox p90 |gap|")
    ax.bar(x + 0.3, three, width=0.2, label="3-box p90 |gap|")
    ax.axhline(0.10, color="#222222", linestyle="--", linewidth=1.0)
    ax.axhline(0.20, color="#555555", linestyle=":", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("m")
    ax.set_title("E093 contact geometry distance/gap summary")
    ax.legend()
    path = out_dir / "contact_geometry_distance_dashboard.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    support = np.array([float(r["wrist5_support_frac"]) for r in summary_rows])
    inside = np.array([float(r["wrist5_inside_frac"]) for r in summary_rows])
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    ax.scatter(inside * 100, support * 100, s=80, c=wrist, cmap="viridis", edgecolor="black")
    for xi, yi, label in zip(inside * 100, support * 100, labels):
        ax.annotate(label, (xi, yi), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.axvline(10, color="#222222", linestyle="--", linewidth=1.0)
    ax.axhline(30, color="#222222", linestyle="--", linewidth=1.0)
    ax.set_xlabel("wrist+5cm inside box (%)")
    ax.set_ylabel("wrist+5cm support face (%)")
    ax.set_title("E093 inside/support vs raw offset color")
    path = out_dir / "contact_geometry_inside_support_dashboard.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)
    return paths


def write_markdown(path: Path, rows: list[dict[str, Any]], png_records: list[dict[str, Any]]) -> None:
    lines = [
        "# E093 Contact Geometry Summary",
        "",
        f"- summary rows: `{len(rows)}`",
        f"- nonblank PNGs: `{sum(1 for r in png_records if r['nonblank'])}/{len(png_records)}`",
        "",
        "| case | hand | wrist5->raw mean | wrist5 inside | wrist5 support | sphere p90 | handbox p90 | 3-box p90 | flags |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['case_id']}` | {HAND_PREFIX[r['hand']]} | {float(r['wrist5_to_raw_mean_m']):.3f} | "
            f"{float(r['wrist5_inside_frac']):.3f} | {float(r['wrist5_support_frac']):.3f} | "
            f"{float(r['sphere_surface_gap_p90_abs_m']):.3f} | {float(r['handbox_surface_gap_p90_abs_m']):.3f} | "
            f"{float(r['threebox_surface_gap_p90_abs_m']):.3f} | {r['bad_proxy_flags']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--sample-count", type=int, default=20000)
    args = parser.parse_args()

    rows = [r for r in read_tsv(args.manifest) if str(r.get("ready", "")).lower() == "true"]
    handbox = parse_handbox_urdf(HANDBOX_URDF)
    all_summary: list[dict[str, Any]] = []
    all_per_frame: list[dict[str, Any]] = []
    png_paths: list[Path] = []
    case_meta: dict[str, Any] = {
        "handbox_urdf": str(HANDBOX_URDF),
        "handbox": handbox,
        "threebox_source": "workspace/core4d/scripts/convert/patch_hand_3box.py constants",
    }

    for row in rows:
        print(f"[E093] auditing {row['case_id']} {row['task']}")
        case = compute_case(row, handbox, args.sample_count)
        all_per_frame.extend(case["per_frame"])
        all_summary.extend(summarize_case(case))
        png_paths.append(plot_overlay(case, args.out_root / "visuals/object_local"))
        png_paths.append(plot_timeline(case, args.out_root / "visuals/timeline"))
        case_meta[row["case_id"]] = {"raw_meta": case["raw_meta"], "T": int(case["qpos"].shape[0]), "half": case["half"]}

    png_paths.extend(plot_dashboard(all_summary, args.out_root / "visuals/dashboard"))
    write_tsv(args.out_root / "per_frame_points.csv", all_per_frame, PER_FRAME_FIELDS)
    write_tsv(args.out_root / "geometry_summary.csv", all_summary, SUMMARY_FIELDS)
    write_json(args.out_root / "geometry_summary.json", {"summary": all_summary, "meta": case_meta})
    png_records = [
        {"path": str(path), "exists": path.is_file(), "nonblank": image_nonblank(path), "bytes": path.stat().st_size if path.is_file() else 0}
        for path in png_paths
    ]
    write_tsv(args.out_root / "visuals/png_manifest.tsv", png_records, ["path", "exists", "nonblank", "bytes"])
    write_markdown(args.out_root / "geometry_summary.md", all_summary, png_records)
    print(f"[E093] summary rows: {len(all_summary)}")
    print(f"[E093] per-frame rows: {len(all_per_frame)}")
    print(f"[E093] PNG nonblank: {sum(1 for r in png_records if r['nonblank'])}/{len(png_records)}")


if __name__ == "__main__":
    main()
