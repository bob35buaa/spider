#!/usr/bin/env python3
"""Generate E085 raw CORE4D contact targets in MuJoCo object body frame."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
EVAL_DIR = REPO / "workspace/core4d/scripts/eval"
DEBUG_DIR = REPO / "workspace/core4d/scripts/debug"
for path in (EVAL_DIR, DEBUG_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval_E078 import load_ref  # noqa: E402


PROC = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
DEFAULT_CASES = REPO / "workspace/core4d/scripts/E085/target_cases.tsv"
DEFAULT_OUT = REPO / "workspace/core4d/results/E085/raw_targets"

PERSONS = ("person1", "person2")
HANDS = ("left", "right")
HAND_RANGES = {
    "left": np.arange(4700, 5500, dtype=np.int64),
    "right": np.arange(7500, 8150, dtype=np.int64),
}
FINGERTIP_IDS = {
    "left": np.array([5361, 4933, 5058, 5169, 5286], dtype=np.int64),
    "right": np.array([8079, 7669, 7794, 7905, 8022], dtype=np.int64),
}
FACE_NAMES = ("+x", "-x", "+y", "-y", "+z", "-z", "")
FACE_TO_ID = {name: i for i, name in enumerate(FACE_NAMES)}


def jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(type(obj).__name__)


def read_tsv(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row and not row.get("variant", "").startswith("#"):
                rows.append(row)
    return rows


def load_person(seq_dir: Path, person: str) -> dict:
    path = seq_dir / f"{person}_poses.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=True)["arr_0"].item()


def name2id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise KeyError(f"missing {obj_type}: {name}")
    return int(idx)


def resize_time(arr: np.ndarray, target_len: int) -> np.ndarray:
    if arr.shape[0] == target_len:
        return arr.copy()
    idx = np.round(np.linspace(0, arr.shape[0] - 1, target_len)).astype(np.int64)
    return arr[idx]


def quat_wxyz_to_mat(quat: np.ndarray) -> np.ndarray:
    return Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()


def box_surface_axis(local: np.ndarray, half: np.ndarray) -> int:
    abs_local = np.abs(local)
    outside = abs_local - half
    if np.any(outside > 0.0):
        return int(np.argmax(outside))
    margin = half - abs_local
    return int(np.argmin(margin))


def face_label(local: np.ndarray, half: np.ndarray) -> str:
    axis = box_surface_axis(local, half)
    sign = "+" if local[axis] >= 0 else "-"
    return f"{sign}{'xyz'[axis]}"


def vertical_fraction(
    local: np.ndarray, obj_pos: np.ndarray, obj_mat: np.ndarray, half: np.ndarray
) -> float:
    corners = np.array(
        [
            [sx, sy, sz]
            for sx in (-half[0], half[0])
            for sy in (-half[1], half[1])
            for sz in (-half[2], half[2])
        ],
        dtype=float,
    )
    corner_z = (corners @ obj_mat.T + obj_pos)[:, 2]
    world = local @ obj_mat.T + obj_pos
    return float((world[2] - corner_z.min()) / max(float(np.ptp(corner_z)), 1e-9))


def surface_dist(local: np.ndarray, half: np.ndarray) -> float:
    return float(np.linalg.norm(local - np.clip(local, -half, half)))


def project_to_box_surface(local: np.ndarray, half: np.ndarray) -> np.ndarray:
    """Project a semantic raw visual-mesh point to the nearest collision face."""
    axis = box_surface_axis(local, half)
    projected = np.clip(local, -half, half)
    sign = 1.0 if local[axis] >= 0.0 else -1.0
    projected[axis] = sign * half[axis]
    return projected


def load_scene_geometry(task: str) -> dict:
    scene = PROC / task / "scene_act.xml"
    model = mujoco.MjModel.from_xml_path(str(scene))
    obj_gid = name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_collision")
    visual_gid = name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_visual")
    return {
        "scene": scene,
        "half": model.geom_size[obj_gid].copy(),
        "visual_pos": model.geom_pos[visual_gid].copy(),
        "visual_quat": model.geom_quat[visual_gid].copy(),
    }


def fill_targets(target: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    filled = target.copy()
    stats: dict[str, int] = {}
    T = target.shape[0]
    for hi, hand in enumerate(HANDS):
        valid_idx = np.where(valid[:, hi] & np.isfinite(target[:, hi]).all(axis=1))[0]
        if valid_idx.size == 0:
            raise ValueError(f"no valid target frames for {hand}")
        nan_or_invalid = ~np.isfinite(filled[:, hi]).all(axis=1)
        for t in np.where(nan_or_invalid)[0]:
            nearest = valid_idx[np.argmin(np.abs(valid_idx - t))]
            filled[t, hi] = target[nearest, hi]
        stats[f"{hand}_filled_frames"] = int(nan_or_invalid.sum())
        stats[f"{hand}_valid_frames"] = int(valid_idx.size)
        stats[f"{hand}_first_valid"] = int(valid_idx[0])
        stats[f"{hand}_last_valid"] = int(valid_idx[-1])
    return filled, stats


def generate_case(row: dict[str, str], out_root: Path, threshold: float, sample_count: int, seed: int) -> dict:
    variant = row["variant"]
    task = row["task"]
    person_idx = int(row["person_idx"])
    mask_path = Path(row["mask_path"])
    if not mask_path.is_absolute():
        mask_path = REPO / mask_path
    audit = json.loads((mask_path.parent / "audit_summary_3cm.json").read_text())
    seq_dir = Path(audit["seq_dir"])
    mesh_path = Path(audit["mesh"])

    geom = load_scene_geometry(task)
    half = geom["half"]
    visual_pos = geom["visual_pos"]
    visual_quat = geom["visual_quat"]
    mesh_rot = quat_wxyz_to_mat(visual_quat)

    mesh = trimesh.load(mesh_path, process=False)
    rng = np.random.default_rng(seed)
    surface_points, _ = trimesh.sample.sample_surface(mesh, sample_count, seed=rng)
    body_surface_points = (surface_points - visual_pos) @ mesh_rot

    mask_npz = np.load(mask_path, allow_pickle=True)
    raw_mask = mask_npz["raw_contact_mask_3cm"][:, person_idx, :].astype(bool)
    raw_min_dist = mask_npz["raw_min_dist_m"][:, person_idx, :]
    raw_tip_min_dist = mask_npz["raw_tip_min_dist_m"][:, person_idx, :]
    raw_vertex_count = mask_npz["raw_vertex_count_lt_thresh"][:, person_idx, :]
    obj_poses = np.load(seq_dir / "smooth_objposes.npy")
    person = load_person(seq_dir, PERSONS[person_idx])

    n_raw = raw_mask.shape[0]
    raw_target = np.full((n_raw, 2, 3), np.nan, dtype=np.float32)
    raw_visual_target = np.full((n_raw, 2, 3), np.nan, dtype=np.float32)
    raw_valid = np.zeros((n_raw, 2), dtype=bool)
    raw_face_id = np.full((n_raw, 2), FACE_TO_ID[""], dtype=np.int16)
    raw_vfrac = np.full((n_raw, 2), np.nan, dtype=np.float32)
    raw_surf_dist = np.full((n_raw, 2), np.nan, dtype=np.float32)
    raw_source = np.full((n_raw, 2), "", dtype=object)
    rows = []

    for raw_f in range(n_raw):
        R_raw = obj_poses[raw_f, :3, :3]
        t_raw = obj_poses[raw_f, :3, 3]
        body_obj_pos = visual_pos @ R_raw.T + t_raw
        body_obj_mat = R_raw @ mesh_rot
        obj_world = surface_points @ R_raw.T + t_raw
        tree = cKDTree(obj_world)
        vertices = person["vertices"][raw_f]

        for hi, hand in enumerate(HANDS):
            if raw_mask[raw_f, hi]:
                hand_verts = vertices[HAND_RANGES[hand]]
                dists, nn_idx = tree.query(hand_verts, k=1)
                close = dists < threshold
                if close.any():
                    target_raw_local = surface_points[nn_idx[close]].mean(axis=0)
                    source = "close_centroid"
                else:
                    tip_verts = vertices[FINGERTIP_IDS[hand]]
                    tip_dists, tip_nn_idx = tree.query(tip_verts, k=1)
                    target_raw_local = surface_points[tip_nn_idx[int(np.argmin(tip_dists))]]
                    source = "tip_fallback"
                visual_target_body = (target_raw_local - visual_pos) @ mesh_rot
                target_body = project_to_box_surface(visual_target_body, half)
                face = face_label(target_body, half)
                vfrac = vertical_fraction(target_body, body_obj_pos, body_obj_mat, half)
                raw_target[raw_f, hi] = target_body.astype(np.float32)
                raw_visual_target[raw_f, hi] = visual_target_body.astype(np.float32)
                raw_valid[raw_f, hi] = True
                raw_face_id[raw_f, hi] = FACE_TO_ID[face]
                raw_vfrac[raw_f, hi] = vfrac
                raw_surf_dist[raw_f, hi] = surface_dist(target_body, half)
                raw_source[raw_f, hi] = source
                rows.append(
                    {
                        "raw_frame": raw_f,
                        "hand": hand,
                        "source": source,
                        "visual_target_x": float(visual_target_body[0]),
                        "visual_target_y": float(visual_target_body[1]),
                        "visual_target_z": float(visual_target_body[2]),
                        "target_x": float(target_body[0]),
                        "target_y": float(target_body[1]),
                        "target_z": float(target_body[2]),
                        "face": face,
                        "vertical_frac": float(vfrac),
                        "surface_dist_m": float(raw_surf_dist[raw_f, hi]),
                        "mask_min_dist_m": float(raw_min_dist[raw_f, hi]),
                        "mask_tip_min_dist_m": float(raw_tip_min_dist[raw_f, hi]),
                        "mask_vertices_lt_thresh": int(raw_vertex_count[raw_f, hi]),
                    }
                )

    raw_target_filled, fill_stats = fill_targets(raw_target, raw_valid)
    raw_visual_target_filled, visual_fill_stats = fill_targets(
        raw_visual_target, raw_valid
    )
    trim_start = int(mask_npz["trim_start"])
    spider_len = int(mask_npz["spider_contact_mask_3cm"].shape[0])
    spider_slice = slice(trim_start, trim_start + spider_len)
    eval_raw_idx = mask_npz["eval_raw_idx"].astype(np.int64)

    spider_target = raw_target_filled[spider_slice]
    spider_visual_target = raw_visual_target_filled[spider_slice]
    spider_valid = raw_valid[spider_slice]
    spider_face_id = raw_face_id[spider_slice]
    spider_vfrac = raw_vfrac[spider_slice]
    spider_surf_dist = raw_surf_dist[spider_slice]

    eval_target = raw_target_filled[eval_raw_idx]
    eval_visual_target = raw_visual_target_filled[eval_raw_idx]
    eval_valid = raw_valid[eval_raw_idx]
    eval_face_id = raw_face_id[eval_raw_idx]
    eval_vfrac = raw_vfrac[eval_raw_idx]
    eval_surf_dist = raw_surf_dist[eval_raw_idx]

    out_dir = out_root / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "raw_contact_targets.npz"
    np.savez(
        npz_path,
        raw_contact_target_object_local=raw_target_filled,
        spider_contact_target_object_local=spider_target,
        eval_contact_target_object_local=eval_target,
        raw_contact_visual_target_object_local=raw_visual_target_filled,
        spider_contact_visual_target_object_local=spider_visual_target,
        eval_contact_visual_target_object_local=eval_visual_target,
        raw_target_valid=raw_valid,
        spider_target_valid=spider_valid,
        eval_target_valid=eval_valid,
        raw_target_face_id=raw_face_id,
        spider_target_face_id=spider_face_id,
        eval_target_face_id=eval_face_id,
        raw_target_vertical_frac=raw_vfrac,
        spider_target_vertical_frac=spider_vfrac,
        eval_target_vertical_frac=eval_vfrac,
        raw_target_surface_dist_m=raw_surf_dist,
        spider_target_surface_dist_m=spider_surf_dist,
        eval_target_surface_dist_m=eval_surf_dist,
        face_names=np.array(FACE_NAMES),
        raw_contact_mask_3cm=raw_mask,
        spider_contact_mask_3cm=mask_npz["spider_contact_mask_3cm"][:, person_idx, :],
        eval_contact_mask_3cm=mask_npz["eval_contact_mask_3cm"][:, person_idx, :],
        eval_raw_idx=eval_raw_idx,
        trim_start=np.array(trim_start),
        threshold_m=np.array(threshold),
        target_kind=np.array("collision_surface_projected_from_raw_visual"),
        scene_collision_half_extents=half,
        object_visual_geom_pos=visual_pos,
        object_visual_geom_quat_wxyz=visual_quat,
    )

    csv_path = out_dir / "raw_contact_targets.csv"
    if rows:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = summarize_case(
        row,
        mask_path,
        seq_dir,
        mesh_path,
        geom,
        raw_target_filled,
        raw_valid,
        raw_face_id,
        raw_vfrac,
        raw_surf_dist,
        fill_stats,
        visual_fill_stats,
        npz_path,
        csv_path,
    )
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=jsonable), encoding="utf-8")
    return summary


def summarize_hand(valid: np.ndarray, face_id: np.ndarray, vfrac: np.ndarray, surf_dist: np.ndarray, hi: int) -> dict:
    idx = np.where(valid[:, hi])[0]
    if idx.size == 0:
        return {"n": 0}
    faces = [FACE_NAMES[int(face_id[i, hi])] for i in idx]
    vf = vfrac[idx, hi]
    sd = surf_dist[idx, hi]
    return {
        "n": int(idx.size),
        "first_last_raw": [int(idx[0]), int(idx[-1])],
        "face_counts": dict(Counter(faces)),
        "vertical_frac_mean_min_max": [
            float(np.nanmean(vf)),
            float(np.nanmin(vf)),
            float(np.nanmax(vf)),
        ],
        "surface_dist_cm_mean_max": [
            float(np.nanmean(sd) * 100.0),
            float(np.nanmax(sd) * 100.0),
        ],
    }


def summarize_case(
    row: dict[str, str],
    mask_path: Path,
    seq_dir: Path,
    mesh_path: Path,
    geom: dict,
    raw_target: np.ndarray,
    raw_valid: np.ndarray,
    raw_face_id: np.ndarray,
    raw_vfrac: np.ndarray,
    raw_surf_dist: np.ndarray,
    fill_stats: dict[str, int],
    visual_fill_stats: dict[str, int],
    npz_path: Path,
    csv_path: Path,
) -> dict:
    summary = {
        "variant": row["variant"],
        "role": row["role"],
        "task": row["task"],
        "override": row["override"],
        "person_idx": int(row["person_idx"]),
        "person": PERSONS[int(row["person_idx"])],
        "mask_path": str(mask_path.relative_to(REPO)),
        "raw_seq_dir": str(seq_dir),
        "raw_mesh": str(mesh_path),
        "scene": str(geom["scene"].relative_to(REPO)),
        "scene_collision_half_extents": geom["half"],
        "object_visual_geom_pos": geom["visual_pos"],
        "object_visual_geom_quat_wxyz": geom["visual_quat"],
        "target_kind": "collision_surface_projected_from_raw_visual",
        "fill_stats": fill_stats,
        "visual_fill_stats": visual_fill_stats,
        "hands": {
            hand: summarize_hand(raw_valid, raw_face_id, raw_vfrac, raw_surf_dist, hi)
            for hi, hand in enumerate(HANDS)
        },
        "outputs": {
            "npz": str(npz_path.relative_to(REPO)),
            "csv": str(csv_path.relative_to(REPO)),
        },
    }
    try:
        summary["old_g1_target_comparison"] = compare_old_g1_target(
            row, npz_path, geom["half"]
        )
    except Exception as exc:
        summary["old_g1_target_comparison_error"] = repr(exc)
    return summary


def load_external_target_for_ref(path: Path, target_len: int) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    if (
        "spider_contact_target_object_local" in data
        and data["spider_contact_target_object_local"].shape[0] == target_len
    ):
        arr = data["spider_contact_target_object_local"]
    elif (
        "eval_contact_target_object_local" in data
        and data["eval_contact_target_object_local"].shape[0] == target_len
    ):
        arr = data["eval_contact_target_object_local"]
    elif "eval_contact_target_object_local" in data:
        arr = resize_time(data["eval_contact_target_object_local"], target_len)
    else:
        arr = resize_time(data["spider_contact_target_object_local"], target_len)
    return arr.astype(np.float64)


def load_mask_for_ref(mask_path: Path, person_idx: int, target_len: int) -> np.ndarray:
    data = np.load(mask_path, allow_pickle=True)
    if (
        "spider_contact_mask_3cm" in data
        and data["spider_contact_mask_3cm"].shape[0] == target_len
    ):
        mask = data["spider_contact_mask_3cm"][:, person_idx, :]
    elif (
        "eval_contact_mask_3cm" in data
        and data["eval_contact_mask_3cm"].shape[0] == target_len
    ):
        mask = data["eval_contact_mask_3cm"][:, person_idx, :]
    elif "eval_contact_mask_3cm" in data:
        mask = resize_time(data["eval_contact_mask_3cm"][:, person_idx, :], target_len)
    else:
        mask = resize_time(data["spider_contact_mask_3cm"][:, person_idx, :], target_len)
    return mask.astype(bool)


def compare_old_g1_target(row: dict[str, str], target_path: Path, half: np.ndarray) -> dict:
    qpos_ref, _ctrl_ref = load_ref(row["override"], row["task"])
    model = mujoco.MjModel.from_xml_path(str(PROC / row["task"] / "scene_act.xml"))
    data = mujoco.MjData(model)
    obj_bid = name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    body_ids = [
        name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link"),
        name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link"),
    ]
    external = load_external_target_for_ref(target_path, len(qpos_ref))
    mask_path = REPO / row["mask_path"]
    mask = load_mask_for_ref(mask_path, int(row["person_idx"]), len(qpos_ref))
    offset = np.array([0.05, 0.0, 0.0], dtype=float)
    old = np.zeros_like(external)
    for t, q in enumerate(qpos_ref):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        obj_pos = data.xpos[obj_bid].copy()
        obj_mat = data.xmat[obj_bid].reshape(3, 3).copy()
        for hi, bid in enumerate(body_ids):
            hand_pos = data.xpos[bid].copy()
            hand_mat = data.xmat[bid].reshape(3, 3).copy()
            old[t, hi] = obj_mat.T @ ((hand_pos + hand_mat @ offset) - obj_pos)

    out: dict[str, object] = {}
    for hi, hand in enumerate(HANDS):
        active = mask[:, hi]
        if not active.any():
            out[hand] = {"n": 0}
            continue
        old_surf = np.clip(old[active, hi], -half, half)
        ext_surf = np.clip(external[active, hi], -half, half)
        delta = np.linalg.norm(old_surf - ext_surf, axis=1)
        out[hand] = {
            "n": int(active.sum()),
            "surface_delta_cm_mean_min_max": [
                float(delta.mean() * 100.0),
                float(delta.min() * 100.0),
                float(delta.max() * 100.0),
            ],
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--threshold", type=float, default=0.03)
    parser.add_argument("--sample-count", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=77)
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    summaries = [
        generate_case(row, args.out_root, args.threshold, args.sample_count, args.seed)
        for row in read_tsv(args.cases)
    ]
    aggregate = {
        "num_cases": len(summaries),
        "cases": summaries,
    }
    path = args.out_root / "aggregate_summary.json"
    path.write_text(json.dumps(aggregate, indent=2, default=jsonable), encoding="utf-8")
    print(json.dumps(aggregate, indent=2, default=jsonable))


if __name__ == "__main__":
    main()
