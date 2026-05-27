#!/usr/bin/env python3
"""Shared helpers for E029 COLA-style support-body experiments."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[4]
BASE = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
E028_RESULTS = REPO / "workspace/core4d_collab_retarget/results/E028"
E029_RESULTS = REPO / "workspace/core4d_collab_retarget/results/E029"
OVERRIDE_DIR = REPO / "examples/config/override"
MANIFEST = E028_RESULTS / "manifest.tsv"
CANDIDATES = E028_RESULTS / "candidates.json"

FACES = ["+x", "-x", "+y", "-y", "+z", "-z"]
SIDE_FACES = ["+x", "-x", "+y", "-y"]
AXES = "xyz"
WORLD_AXES = "xyz"


def rel_or_abs(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO))
    except ValueError:
        return str(resolved)


def read_manifest(path: Path = MANIFEST) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return {row["variant"]: row for row in csv.DictReader(f, delimiter="\t")}


def read_candidates(path: Path = CANDIDATES) -> list[str]:
    return list(json.loads(path.read_text(encoding="utf-8")))


def candidate_rows(
    *,
    manifest_path: Path = MANIFEST,
    candidates_path: Path = CANDIDATES,
) -> list[dict[str, str]]:
    manifest = read_manifest(manifest_path)
    rows: list[dict[str, str]] = []
    missing: list[str] = []
    for variant in read_candidates(candidates_path):
        row = manifest.get(variant)
        if row is None:
            missing.append(variant)
        else:
            rows.append(row)
    if missing:
        raise KeyError(f"Candidates missing from manifest: {missing}")
    return rows


def as_float(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def anchor_local(row: dict[str, str]) -> np.ndarray:
    return np.array(
        [
            as_float(row, "support_proxy_point_local_x"),
            as_float(row, "support_proxy_point_local_y"),
            as_float(row, "support_proxy_point_local_z"),
        ],
        dtype=np.float64,
    )


def object_half(row: dict[str, str]) -> np.ndarray:
    return np.array(
        [
            as_float(row, "object_half_x"),
            as_float(row, "object_half_y"),
            as_float(row, "object_half_z"),
        ],
        dtype=np.float64,
    )


def source_task_dir(row: dict[str, str]) -> Path:
    return BASE / row["source_task"]


def derived_task_dir(row: dict[str, str]) -> Path:
    return BASE / row["derived_task"]


def scene_path(row: dict[str, str]) -> Path:
    return derived_task_dir(row) / f"{row['scene_name']}.xml"


def source_scene_path(row: dict[str, str]) -> Path:
    return source_task_dir(row) / "scene.xml"


def override_path(row: dict[str, str]) -> Path:
    return OVERRIDE_DIR / f"core4d_collab_{row['variant']}.yaml"


def result_npz_path(row: dict[str, str]) -> Path:
    return E028_RESULTS / f"{row['variant']}.npz"


def counterpart_source_task(source_task: str) -> str | None:
    match = re.search(r"_p([12])$", source_task)
    if not match:
        return None
    other = "2" if match.group(1) == "1" else "1"
    return f"{source_task[: match.start(1)]}{other}"


def parse_flat_yaml(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line or line.startswith(" ") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key:
            values[key] = value
    return values


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q / np.clip(np.linalg.norm(q), 1e-8, None)
    v = np.asarray(v, dtype=np.float64)
    qvec = q[1:]
    t = 2.0 * np.cross(qvec, v)
    return v + q[0] * t + np.cross(qvec, t)


def quat_apply_inv(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qi = np.asarray(q, dtype=np.float64).copy()
    qi = qi / np.clip(np.linalg.norm(qi), 1e-8, None)
    qi[1:] *= -1.0
    return quat_apply(qi, v)


def object_body_id(model: mujoco.MjModel) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    if body_id < 0:
        raise ValueError("model has no body named object")
    return int(body_id)


def body_id(model: mujoco.MjModel, name: str) -> int:
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))


def object_qadr(model: mujoco.MjModel) -> int:
    obj_body = object_body_id(model)
    joint_id = int(model.body_jntadr[obj_body])
    if joint_id < 0:
        raise ValueError("object body has no joint")
    return int(model.jnt_qposadr[joint_id])


def object_last_freejoint(model: mujoco.MjModel) -> bool:
    try:
        qadr = object_qadr(model)
    except ValueError:
        return False
    return qadr == model.nq - 7 and int(model.jnt_type[int(model.body_jntadr[object_body_id(model)])]) == int(
        mujoco.mjtJoint.mjJNT_FREE
    )


def face_label(point: np.ndarray, half: np.ndarray) -> str:
    norm = np.abs(point) / np.clip(half, 1e-6, None)
    axis = int(np.argmax(norm))
    sign = "+" if point[axis] >= 0.0 else "-"
    return f"{sign}{AXES[axis]}"


def face_counts(points: np.ndarray, half: np.ndarray) -> dict[str, int]:
    labels = [face_label(point, half) for point in points]
    return {face: labels.count(face) for face in FACES}


def top_face(counts: dict[str, int], faces: list[str] | tuple[str, ...] = tuple(FACES)) -> str:
    if not counts:
        return ""
    return max(faces, key=lambda face: counts.get(face, 0))


def side_margin(counts: dict[str, int], total: int) -> tuple[str, float, float]:
    if total <= 0:
        return "", 0.0, 0.0
    ranked = sorted(SIDE_FACES, key=lambda face: counts.get(face, 0), reverse=True)
    best = ranked[0]
    side_frac = counts.get(best, 0) / total
    margin = (counts.get(ranked[0], 0) - counts.get(ranked[1], 0)) / total
    return best, side_frac, margin


def load_contact_mask(mask_npz: Path, person_idx: int, target_len: int | None = None) -> np.ndarray | None:
    if not mask_npz.is_file():
        return None
    data = np.load(mask_npz, allow_pickle=True)
    key = "spider_contact_mask_3cm" if "spider_contact_mask_3cm" in data.files else "eval_contact_mask_3cm"
    if key not in data.files:
        return None
    raw = data[key]
    if raw.ndim != 3:
        return None
    person_idx = min(max(person_idx, 0), raw.shape[1] - 1)
    mask = raw[:, person_idx, :2].astype(bool)
    if target_len is None or len(mask) == target_len:
        return mask
    if len(mask) == 0:
        return None
    idx = np.round(np.linspace(0, len(mask) - 1, target_len)).astype(int)
    return mask[idx]


def load_source_case(
    task_name: str,
    *,
    mask_npz: Path | None = None,
    person_idx: int = 0,
) -> tuple[mujoco.MjModel, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    task_dir = BASE / task_name
    scene = task_dir / "scene.xml"
    traj_path = task_dir / "0/trajectory_kinematic.npz"
    if not scene.is_file():
        raise FileNotFoundError(scene)
    if not traj_path.is_file():
        raise FileNotFoundError(traj_path)
    model = mujoco.MjModel.from_xml_path(str(scene))
    traj = np.load(traj_path)
    qpos = traj["qpos"].reshape(-1, model.nq).astype(np.float64)
    contact_pos = traj["contact_pos"].reshape(qpos.shape[0], 2, 3).astype(np.float64)
    contact = traj["contact"].reshape(qpos.shape[0], 2).astype(bool) if "contact" in traj.files else None
    mask = load_contact_mask(mask_npz, person_idx, qpos.shape[0]) if mask_npz is not None else None
    if contact is not None and mask is not None:
        active = np.logical_or(contact, mask)
    elif contact is not None:
        active = contact
    elif mask is not None:
        active = mask
    else:
        active = np.ones((qpos.shape[0], 2), dtype=bool)
    return model, qpos, contact_pos, active, mask


def contact_points_local(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    contact_pos: np.ndarray,
    active: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qadr = object_qadr(model)
    points: list[np.ndarray] = []
    hands: list[int] = []
    frames: list[int] = []
    for frame_idx, q in enumerate(qpos):
        obj_pos = q[qadr : qadr + 3].astype(np.float64)
        obj_quat = q[qadr + 3 : qadr + 7].astype(np.float64)
        for hand_idx in range(2):
            if not bool(active[frame_idx, hand_idx]):
                continue
            world = contact_pos[frame_idx, hand_idx]
            if not np.all(np.isfinite(world)):
                continue
            local = quat_apply_inv(obj_quat, world - obj_pos)
            if not np.all(np.isfinite(local)):
                continue
            points.append(local)
            hands.append(hand_idx)
            frames.append(frame_idx)
    if not points:
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )
    return np.asarray(points), np.asarray(hands, dtype=np.int64), np.asarray(frames, dtype=np.int64)


def robust_centroid(points: np.ndarray, trim: float = 0.1) -> np.ndarray:
    if len(points) == 0:
        return np.full(3, np.nan, dtype=np.float64)
    if len(points) < 10:
        return np.median(points, axis=0)
    lo = np.quantile(points, trim, axis=0)
    hi = np.quantile(points, 1.0 - trim, axis=0)
    keep = np.all((points >= lo) & (points <= hi), axis=1)
    if not np.any(keep):
        return np.median(points, axis=0)
    return points[keep].mean(axis=0)


def point_fmt(point: np.ndarray) -> str:
    if len(point) != 3 or not np.all(np.isfinite(point)):
        return "nan,nan,nan"
    return f"{point[0]:+.6f},{point[1]:+.6f},{point[2]:+.6f}"

