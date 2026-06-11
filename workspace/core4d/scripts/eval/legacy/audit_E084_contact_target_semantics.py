#!/usr/bin/env python3
"""Audit CORE4D raw contact points vs G1 HDMI dynamic contact targets.

This script is intentionally case-specific by default: it checks the E084
main case where visual inspection suggested the hands chase a suspicious box
edge.  The output separates three layers:

1. raw CORE4D SMPL-X hand vertices against object mesh;
2. the per-hand 3cm binary contact mask used by MJWP;
3. the G1 reference point used by contact_hdmi reward
   (wrist_yaw_link + contact_hdmi_eef_offset).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


DEFAULT_CASE = "d003_box021_20231018_029_p2_upperobj_e083"
DEFAULT_OVERRIDE = "core4d_E084A_d003_box021_20231018_029_p2_safety"
DEFAULT_MASK = (
    REPO
    / "workspace/core4d/results/E084/contact_masks/"
    / "d003_box021_20231018_029_p2/raw_contact_mask_3cm.npz"
)
DEFAULT_OUT = REPO / "workspace/core4d/results/E084/contact_target_audit"

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


def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Counter):
        return dict(obj)
    if isinstance(obj, defaultdict):
        return dict(obj)
    raise TypeError(type(obj).__name__)


def name2id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise KeyError(f"missing {obj_type}: {name}")
    return int(idx)


def load_person(seq_dir: Path, person: str) -> dict:
    path = seq_dir / f"{person}_poses.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=True)["arr_0"].item()


def load_mask(mask_path: Path, person_idx: int, target_len: int) -> dict:
    data = np.load(mask_path, allow_pickle=True)
    if "spider_contact_mask_3cm" in data and data["spider_contact_mask_3cm"].shape[0] == target_len:
        axis = "spider"
    elif "eval_contact_mask_3cm" in data and data["eval_contact_mask_3cm"].shape[0] == target_len:
        axis = "eval"
    else:
        axis = "eval" if "eval_contact_mask_3cm" in data else "spider"

    key = f"{axis}_contact_mask_3cm"
    source = data[key][:, person_idx, :].astype(bool)
    if source.shape[0] == target_len:
        resize_idx = np.arange(target_len, dtype=np.int64)
        mask = source
    else:
        resize_idx = np.round(np.linspace(0, source.shape[0] - 1, target_len)).astype(np.int64)
        mask = source[resize_idx]

    if axis == "eval" and "eval_raw_idx" in data:
        source_raw_idx = data["eval_raw_idx"].astype(np.int64)
        raw_idx_for_qpos = source_raw_idx[resize_idx]
    elif axis == "spider":
        trim_start = int(data["trim_start"])
        source_raw_idx = trim_start + np.arange(source.shape[0], dtype=np.int64)
        raw_idx_for_qpos = source_raw_idx[resize_idx]
    else:
        source_raw_idx = np.arange(source.shape[0], dtype=np.int64)
        raw_idx_for_qpos = source_raw_idx[resize_idx]

    return {
        "npz": data,
        "axis": axis,
        "key": key,
        "source": source,
        "mask": mask,
        "resize_idx": resize_idx,
        "source_raw_idx": source_raw_idx,
        "raw_idx_for_qpos": raw_idx_for_qpos,
    }


def face_label(local: np.ndarray, half: np.ndarray) -> str:
    abs_local = np.abs(local)
    outside = abs_local - half
    if np.any(outside > 0.0):
        axis = int(np.argmax(outside))
    else:
        axis = int(np.argmin(half - abs_local))
    sign = "+" if local[axis] >= 0 else "-"
    return f"{sign}{'xyz'[axis]}"


def face_vertical_labels(obj_mat: np.ndarray) -> tuple[str, str, dict[str, float]]:
    normals = {}
    for axis, name in enumerate("xyz"):
        n = obj_mat[:, axis]
        normals[f"+{name}"] = float(np.dot(n, np.array([0.0, 0.0, 1.0])))
        normals[f"-{name}"] = float(np.dot(-n, np.array([0.0, 0.0, 1.0])))
    bottom = min(normals, key=normals.get)
    top = max(normals, key=normals.get)
    return bottom, top, normals


def vertical_fraction(local: np.ndarray, obj_pos: np.ndarray, obj_mat: np.ndarray, half: np.ndarray) -> float:
    corners = np.array(
        [[sx, sy, sz] for sx in (-half[0], half[0]) for sy in (-half[1], half[1]) for sz in (-half[2], half[2])],
        dtype=float,
    )
    corners_w = corners @ obj_mat.T + obj_pos
    zmin = float(corners_w[:, 2].min())
    zmax = float(corners_w[:, 2].max())
    world = local @ obj_mat.T + obj_pos
    return float((world[2] - zmin) / max(zmax - zmin, 1e-9))


def point_stats(rows: list[dict], prefix: str = "") -> dict:
    out: dict[str, object] = {"n": len(rows)}
    if not rows:
        return out
    for key in ("local_x", "local_y", "local_z", "vertical_frac", "surface_dist_m"):
        vals = np.array([r[prefix + key] for r in rows if np.isfinite(r[prefix + key])], dtype=float)
        if vals.size:
            out[key] = {
                "mean": float(vals.mean()),
                "median": float(np.median(vals)),
                "min": float(vals.min()),
                "max": float(vals.max()),
            }
    out["face_counts"] = dict(Counter(r[prefix + "face"] for r in rows if r.get(prefix + "face")))
    return out


def g1_target_rows(
    model: mujoco.MjModel,
    qpos_ref: np.ndarray,
    mask: np.ndarray,
    raw_idx_for_qpos: np.ndarray,
    half: np.ndarray,
    eef_offset: np.ndarray,
) -> tuple[list[dict], list[dict]]:
    data = mujoco.MjData(model)
    obj_bid = name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    body_ids = {
        "left": name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link"),
        "right": name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link"),
    }
    site_ids = {
        "left": name2id(model, mujoco.mjtObj.mjOBJ_SITE, "contact_left_hand"),
        "right": name2id(model, mujoco.mjtObj.mjOBJ_SITE, "contact_right_hand"),
    }
    geom_ids = {
        "left": name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "lh"),
        "right": name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "rh"),
    }

    rows = []
    frame_rows = []
    for t, q in enumerate(qpos_ref):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        obj_pos = data.xpos[obj_bid].copy()
        obj_mat = data.xmat[obj_bid].reshape(3, 3).copy()
        bottom_face, top_face, up_dots = face_vertical_labels(obj_mat)
        corners = np.array(
            [[sx, sy, sz] for sx in (-half[0], half[0]) for sy in (-half[1], half[1]) for sz in (-half[2], half[2])],
            dtype=float,
        )
        corner_z = (corners @ obj_mat.T + obj_pos)[:, 2]
        frame_rows.append(
            {
                "frame": t,
                "raw_frame": int(raw_idx_for_qpos[t]),
                "bottom_face": bottom_face,
                "top_face": top_face,
                "bottom_z": float(corner_z.min()),
                "top_z": float(corner_z.max()),
                "up_dot_+x": up_dots["+x"],
                "up_dot_+y": up_dots["+y"],
                "up_dot_+z": up_dots["+z"],
            }
        )

        for hi, hand in enumerate(HANDS):
            bid = body_ids[hand]
            wrist_pos = data.xpos[bid].copy()
            wrist_mat = data.xmat[bid].reshape(3, 3).copy()
            points = {
                "wrist": wrist_pos,
                "reward_point": wrist_pos + wrist_mat @ eef_offset,
                "contact_site": data.site_xpos[site_ids[hand]].copy(),
                "hand_geom_center": data.geom_xpos[geom_ids[hand]].copy(),
            }
            row = {
                "frame": t,
                "raw_frame": int(raw_idx_for_qpos[t]),
                "hand": hand,
                "mask": bool(mask[t, hi]),
                "bottom_face": bottom_face,
                "top_face": top_face,
            }
            for pname, world in points.items():
                local = obj_mat.T @ (world - obj_pos)
                clamped = np.clip(local, -half, half)
                surf_dist = float(np.linalg.norm(local - clamped))
                row.update(
                    {
                        f"{pname}_local_x": float(local[0]),
                        f"{pname}_local_y": float(local[1]),
                        f"{pname}_local_z": float(local[2]),
                        f"{pname}_surface_dist_m": surf_dist,
                        f"{pname}_face": face_label(local, half),
                        f"{pname}_vertical_frac": vertical_fraction(local, obj_pos, obj_mat, half),
                        f"{pname}_surface_vertical_frac": vertical_fraction(clamped, obj_pos, obj_mat, half),
                    }
                )
            rows.append(row)
    return rows, frame_rows


def raw_contact_rows(
    seq_dir: Path,
    mesh_path: Path,
    mask_npz: np.lib.npyio.NpzFile,
    person_idx: int,
    threshold: float,
    sample_count: int,
    seed: int,
    mesh_to_body_pos: np.ndarray,
    mesh_to_body_quat_wxyz: np.ndarray,
    body_half: np.ndarray,
) -> tuple[list[dict], dict]:
    mesh = trimesh.load(mesh_path, process=False)
    rng = np.random.default_rng(seed)
    surface_points, _ = trimesh.sample.sample_surface(mesh, sample_count, seed=rng)
    mesh_bounds = np.asarray(mesh.bounds, dtype=float)
    mesh_center = mesh_bounds.mean(axis=0)
    half = (mesh_bounds[1] - mesh_bounds[0]) * 0.5
    # MuJoCo compiles the mesh into a canonical mesh frame and stores a fixed
    # visual geom transform.  Original/raw mesh coordinates are:
    #   raw_local = body_local @ rot.T + pos
    # so body_local = (raw_local - pos) @ rot.
    mesh_rot = Rotation.from_quat(
        [
            mesh_to_body_quat_wxyz[1],
            mesh_to_body_quat_wxyz[2],
            mesh_to_body_quat_wxyz[3],
            mesh_to_body_quat_wxyz[0],
        ]
    ).as_matrix()
    body_surface_points = (surface_points - mesh_to_body_pos) @ mesh_rot

    obj_poses = np.load(seq_dir / "smooth_objposes.npy")
    person = load_person(seq_dir, PERSONS[person_idx])
    raw_mask = mask_npz["raw_contact_mask_3cm"][:, person_idx, :].astype(bool)
    raw_min_dist = mask_npz["raw_min_dist_m"][:, person_idx, :]
    raw_tip_min_dist = mask_npz["raw_tip_min_dist_m"][:, person_idx, :]
    raw_vertex_count = mask_npz["raw_vertex_count_lt_thresh"][:, person_idx, :]

    rows = []
    raw_cache: dict[tuple[int, str], dict] = {}
    for raw_f in range(raw_mask.shape[0]):
        R = obj_poses[raw_f, :3, :3]
        t = obj_poses[raw_f, :3, 3]
        body_obj_pos = mesh_to_body_pos @ R.T + t
        body_obj_mat = R @ mesh_rot
        obj_world = surface_points @ R.T + t
        tree = cKDTree(obj_world)
        vertices = person["vertices"][raw_f]
        for hi, hand in enumerate(HANDS):
            hand_ids = HAND_RANGES[hand]
            tip_ids = FINGERTIP_IDS[hand]
            hand_verts = vertices[hand_ids]
            dists, nn_idx = tree.query(hand_verts, k=1)
            close = dists < threshold
            tip_dists, tip_nn_idx = tree.query(vertices[tip_ids], k=1)
            best_tip = int(np.argmin(tip_dists))

            if close.any():
                hand_local = (hand_verts[close] - t) @ R
                surf_local = surface_points[nn_idx[close]]
                hand_centroid_local = (hand_local.mean(axis=0) - mesh_to_body_pos) @ mesh_rot
                surf_centroid_local = (surf_local.mean(axis=0) - mesh_to_body_pos) @ mesh_rot
                face = face_label(surf_centroid_local, body_half)
                vfrac = vertical_fraction(surf_centroid_local, body_obj_pos, body_obj_mat, body_half)
            else:
                hand_centroid_local = np.full(3, np.nan)
                surf_centroid_local = np.full(3, np.nan)
                face = ""
                vfrac = np.nan

            tip_surf_local = body_surface_points[tip_nn_idx[best_tip]]
            tip_vertex_raw_local = (vertices[tip_ids[best_tip]] - t) @ R
            tip_vertex_local = (tip_vertex_raw_local - mesh_to_body_pos) @ mesh_rot
            row = {
                "raw_frame": raw_f,
                "hand": hand,
                "mask": bool(raw_mask[raw_f, hi]),
                "min_dist_m": float(raw_min_dist[raw_f, hi]),
                "tip_min_dist_m": float(raw_tip_min_dist[raw_f, hi]),
                "n_vertices_lt_thresh": int(raw_vertex_count[raw_f, hi]),
                "close_vertex_count_recomputed": int(close.sum()),
                "surface_local_x": float(surf_centroid_local[0]),
                "surface_local_y": float(surf_centroid_local[1]),
                "surface_local_z": float(surf_centroid_local[2]),
                "surface_face": face,
                "surface_vertical_frac": float(vfrac) if np.isfinite(vfrac) else np.nan,
                "hand_centroid_local_x": float(hand_centroid_local[0]),
                "hand_centroid_local_y": float(hand_centroid_local[1]),
                "hand_centroid_local_z": float(hand_centroid_local[2]),
                "tip_surface_local_x": float(tip_surf_local[0]),
                "tip_surface_local_y": float(tip_surf_local[1]),
                "tip_surface_local_z": float(tip_surf_local[2]),
                "tip_surface_face": face_label(tip_surf_local, body_half),
                "tip_surface_vertical_frac": vertical_fraction(tip_surf_local, body_obj_pos, body_obj_mat, body_half),
                "tip_vertex_local_x": float(tip_vertex_local[0]),
                "tip_vertex_local_y": float(tip_vertex_local[1]),
                "tip_vertex_local_z": float(tip_vertex_local[2]),
            }
            rows.append(row)
            raw_cache[(raw_f, hand)] = row

    meta = {
        "mesh_bounds": mesh_bounds,
        "mesh_center": mesh_center,
        "mesh_half_extents": half,
        "mujoco_body_surface_bounds": np.vstack(
            [body_surface_points.min(axis=0), body_surface_points.max(axis=0)]
        ),
        "mujoco_body_surface_half_extents": (body_surface_points.max(axis=0) - body_surface_points.min(axis=0)) * 0.5,
        "mesh_to_body_pos": mesh_to_body_pos,
        "mesh_to_body_quat_wxyz": mesh_to_body_quat_wxyz,
        "raw_frames": int(raw_mask.shape[0]),
        "raw_cache": raw_cache,
    }
    return rows, meta


def paired_rows(
    g1_rows: list[dict],
    raw_cache: dict[tuple[int, str], dict],
    half: np.ndarray,
) -> list[dict]:
    rows = []
    for g in g1_rows:
        if not g["mask"]:
            continue
        r = raw_cache.get((int(g["raw_frame"]), g["hand"]))
        if not r or not r["mask"]:
            continue
        g_local = np.array(
            [
                g["reward_point_local_x"],
                g["reward_point_local_y"],
                g["reward_point_local_z"],
            ],
            dtype=float,
        )
        g_surf = np.clip(
            g_local,
            -half,
            half,
        )
        raw_surf = np.array(
            [
                r["surface_local_x"],
                r["surface_local_y"],
                r["surface_local_z"],
            ],
            dtype=float,
        )
        if not np.isfinite(raw_surf).all():
            continue
        rows.append(
            {
                "frame": g["frame"],
                "raw_frame": g["raw_frame"],
                "hand": g["hand"],
                "g1_reward_face": g["reward_point_face"],
                "raw_surface_face": r["surface_face"],
                "g1_reward_vertical_frac": g["reward_point_vertical_frac"],
                "g1_reward_surface_vertical_frac": g["reward_point_surface_vertical_frac"],
                "raw_surface_vertical_frac": r["surface_vertical_frac"],
                "local_surface_delta_m": float(np.linalg.norm(g_surf - raw_surf)),
                "local_surface_delta_x": float(g_surf[0] - raw_surf[0]),
                "local_surface_delta_y": float(g_surf[1] - raw_surf[1]),
                "local_surface_delta_z": float(g_surf[2] - raw_surf[2]),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize_delta(rows: list[dict]) -> dict:
    out = {"n": len(rows)}
    if not rows:
        return out
    for key in (
        "local_surface_delta_m",
        "local_surface_delta_x",
        "local_surface_delta_y",
        "local_surface_delta_z",
        "g1_reward_surface_vertical_frac",
        "raw_surface_vertical_frac",
    ):
        vals = np.array([r[key] for r in rows], dtype=float)
        out[key] = {
            "mean": float(vals.mean()),
            "median": float(np.median(vals)),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }
    out["face_pair_counts"] = dict(
        Counter(f"{r['g1_reward_face']}->{r['raw_surface_face']}" for r in rows)
    )
    return out


def make_plots(out_dir: Path, g1_rows: list[dict], raw_rows: list[dict], paired: list[dict], half: np.ndarray) -> None:
    active_g1 = [r for r in g1_rows if r["mask"]]
    active_raw = [r for r in raw_rows if r["mask"]]

    for hand in HANDS:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        g = [r for r in active_g1 if r["hand"] == hand]
        rr = [r for r in active_raw if r["hand"] == hand]
        planes = [
            ("x", "y", half[0], half[1]),
            ("x", "z", half[0], half[2]),
            ("y", "z", half[1], half[2]),
        ]
        for ax, (a, b, ha, hb) in zip(axes, planes):
            if rr:
                ax.scatter(
                    [r[f"surface_local_{a}"] for r in rr],
                    [r[f"surface_local_{b}"] for r in rr],
                    s=12,
                    alpha=0.35,
                    label="raw surface centroid",
                )
            if g:
                ax.scatter(
                    [r[f"reward_point_local_{a}"] for r in g],
                    [r[f"reward_point_local_{b}"] for r in g],
                    s=10,
                    alpha=0.35,
                    label="G1 reward point",
                )
            ax.add_patch(
                plt.Rectangle((-ha, -hb), 2 * ha, 2 * hb, fill=False, color="black", linewidth=1)
            )
            ax.set_xlabel(f"object local {a} (m)")
            ax.set_ylabel(f"object local {b} (m)")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.25)
        axes[0].legend(loc="best", fontsize=8)
        fig.suptitle(f"E084 contact target audit: {hand} hand")
        fig.tight_layout()
        fig.savefig(out_dir / f"{hand}_local_scatter.png", dpi=160)
        plt.close(fig)

    if paired:
        fig, ax = plt.subplots(figsize=(10, 4))
        for hand, color in (("left", "tab:green"), ("right", "tab:blue")):
            rows = [r for r in paired if r["hand"] == hand]
            if not rows:
                continue
            ax.plot(
                [r["frame"] for r in rows],
                [r["g1_reward_surface_vertical_frac"] for r in rows],
                color=color,
                linestyle="-",
                label=f"{hand} G1 surface vfrac",
            )
            ax.plot(
                [r["frame"] for r in rows],
                [r["raw_surface_vertical_frac"] for r in rows],
                color=color,
                linestyle="--",
                label=f"{hand} raw surface vfrac",
            )
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel("G1 ref frame")
        ax.set_ylabel("vertical fraction on box, 0=bottom, 1=top")
        ax.grid(alpha=0.25)
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "vertical_fraction_timeseries.png", dpi=160)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument("--override", default=DEFAULT_OVERRIDE)
    parser.add_argument("--mask", type=Path, default=DEFAULT_MASK)
    parser.add_argument("--person-idx", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.03)
    parser.add_argument("--sample-count", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    audit_json = json.loads((args.mask.parent / "audit_summary_3cm.json").read_text())
    seq_dir = Path(audit_json["seq_dir"])
    mesh_path = Path(audit_json["mesh"])
    scene_path = (
        REPO
        / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
        / args.case
        / "scene_act.xml"
    )

    qpos_ref, _ctrl_ref = load_ref(args.override, args.case)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    obj_gid = name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_collision")
    visual_gid = name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_visual")
    scene_half = model.geom_size[obj_gid].copy()
    visual_pos = model.geom_pos[visual_gid].copy()
    visual_quat = model.geom_quat[visual_gid].copy()
    eef_offset = np.array([0.05, 0.0, 0.0], dtype=float)

    mask_info = load_mask(args.mask, args.person_idx, len(qpos_ref))
    g1_rows, frame_rows = g1_target_rows(
        model,
        qpos_ref,
        mask_info["mask"],
        mask_info["raw_idx_for_qpos"],
        scene_half,
        eef_offset,
    )
    raw_rows, raw_meta = raw_contact_rows(
        seq_dir,
        mesh_path,
        mask_info["npz"],
        args.person_idx,
        args.threshold,
        args.sample_count,
        args.seed,
        visual_pos,
        visual_quat,
        scene_half,
    )
    paired = paired_rows(g1_rows, raw_meta["raw_cache"], scene_half)

    write_csv(args.out_dir / "g1_contact_targets.csv", g1_rows)
    write_csv(args.out_dir / "g1_object_vertical_faces.csv", frame_rows)
    write_csv(args.out_dir / "raw_contact_points.csv", raw_rows)
    write_csv(args.out_dir / "paired_g1_vs_raw.csv", paired)
    make_plots(args.out_dir, g1_rows, raw_rows, paired, scene_half)

    summary = {
        "case": args.case,
        "override": args.override,
        "scene_path": str(scene_path.relative_to(REPO)),
        "mask_path": str(args.mask.relative_to(REPO)),
        "raw_seq_dir": str(seq_dir),
        "raw_mesh": str(mesh_path),
        "person_idx": args.person_idx,
        "person": PERSONS[args.person_idx],
        "qpos_ref_len": int(len(qpos_ref)),
        "mask": {
            "axis_selected_by_run_mjwp_auto": mask_info["axis"],
            "key": mask_info["key"],
            "source_len": int(mask_info["source"].shape[0]),
            "resized_len": int(mask_info["mask"].shape[0]),
            "source_active_pct_left_right": (mask_info["source"].mean(axis=0) * 100.0).tolist(),
            "resized_active_pct_left_right": (mask_info["mask"].mean(axis=0) * 100.0).tolist(),
            "qpos_active_first_last": {
                hand: [
                    int(np.where(mask_info["mask"][:, hi])[0][0]),
                    int(np.where(mask_info["mask"][:, hi])[0][-1]),
                ]
                for hi, hand in enumerate(HANDS)
                if np.where(mask_info["mask"][:, hi])[0].size
            },
            "raw_frames_used_first_last": {
                hand: [
                    int(mask_info["raw_idx_for_qpos"][np.where(mask_info["mask"][:, hi])[0][0]]),
                    int(mask_info["raw_idx_for_qpos"][np.where(mask_info["mask"][:, hi])[0][-1]]),
                ]
                for hi, hand in enumerate(HANDS)
                if np.where(mask_info["mask"][:, hi])[0].size
            },
        },
        "geometry": {
            "scene_collision_half_extents": scene_half,
            "raw_mesh_bounds": raw_meta["mesh_bounds"],
            "raw_mesh_center": raw_meta["mesh_center"],
            "raw_mesh_half_extents": raw_meta["mesh_half_extents"],
            "raw_points_transformed_to_mujoco_body_frame": True,
            "object_visual_geom_pos": visual_pos,
            "object_visual_geom_quat_wxyz": visual_quat,
            "mujoco_body_surface_bounds": raw_meta["mujoco_body_surface_bounds"],
            "mujoco_body_surface_half_extents": raw_meta["mujoco_body_surface_half_extents"],
            "g1_reward_point": "wrist_yaw_link + [0.05, 0, 0]",
            "g1_contact_site": "contact_*_hand site at wrist_yaw_link + [0.08, 0, 0]",
            "g1_hand_geom": "lh/rh sphere center at wrist_yaw_link + [0.10, 0, 0], radius=0.05",
        },
        "object_vertical_face_counts_in_g1_ref": {
            "bottom": dict(Counter(r["bottom_face"] for r in frame_rows)),
            "top": dict(Counter(r["top_face"] for r in frame_rows)),
        },
        "g1_active_reward_point_stats": {
            hand: point_stats(
                [
                    {
                        "local_x": r["reward_point_local_x"],
                        "local_y": r["reward_point_local_y"],
                        "local_z": r["reward_point_local_z"],
                        "vertical_frac": r["reward_point_vertical_frac"],
                        "surface_dist_m": r["reward_point_surface_dist_m"],
                        "face": r["reward_point_face"],
                    }
                    for r in g1_rows
                    if r["hand"] == hand and r["mask"]
                ]
            )
            for hand in HANDS
        },
        "g1_active_site_and_geom_stats": {
            hand: {
                "contact_site": point_stats(
                    [
                        {
                            "local_x": r["contact_site_local_x"],
                            "local_y": r["contact_site_local_y"],
                            "local_z": r["contact_site_local_z"],
                            "vertical_frac": r["contact_site_vertical_frac"],
                            "surface_dist_m": r["contact_site_surface_dist_m"],
                            "face": r["contact_site_face"],
                        }
                        for r in g1_rows
                        if r["hand"] == hand and r["mask"]
                    ]
                ),
                "hand_geom_center": point_stats(
                    [
                        {
                            "local_x": r["hand_geom_center_local_x"],
                            "local_y": r["hand_geom_center_local_y"],
                            "local_z": r["hand_geom_center_local_z"],
                            "vertical_frac": r["hand_geom_center_vertical_frac"],
                            "surface_dist_m": r["hand_geom_center_surface_dist_m"],
                            "face": r["hand_geom_center_face"],
                        }
                        for r in g1_rows
                        if r["hand"] == hand and r["mask"]
                    ]
                ),
            }
            for hand in HANDS
        },
        "raw_active_surface_stats": {
            hand: point_stats(
                [
                    {
                        "local_x": r["surface_local_x"],
                        "local_y": r["surface_local_y"],
                        "local_z": r["surface_local_z"],
                        "vertical_frac": r["surface_vertical_frac"],
                        "surface_dist_m": r["min_dist_m"],
                        "face": r["surface_face"],
                    }
                    for r in raw_rows
                    if r["hand"] == hand and r["mask"]
                ]
            )
            for hand in HANDS
        },
        "raw_active_fingertip_surface_stats": {
            hand: point_stats(
                [
                    {
                        "local_x": r["tip_surface_local_x"],
                        "local_y": r["tip_surface_local_y"],
                        "local_z": r["tip_surface_local_z"],
                        "vertical_frac": r["tip_surface_vertical_frac"],
                        "surface_dist_m": r["tip_min_dist_m"],
                        "face": r["tip_surface_face"],
                    }
                    for r in raw_rows
                    if r["hand"] == hand and r["mask"]
                ]
            )
            for hand in HANDS
        },
        "paired_g1_reward_surface_vs_raw_surface": {
            hand: summarize_delta([r for r in paired if r["hand"] == hand])
            for hand in HANDS
        },
        "outputs": {
            "summary_json": str((args.out_dir / "summary.json").relative_to(REPO)),
            "g1_csv": str((args.out_dir / "g1_contact_targets.csv").relative_to(REPO)),
            "raw_csv": str((args.out_dir / "raw_contact_points.csv").relative_to(REPO)),
            "paired_csv": str((args.out_dir / "paired_g1_vs_raw.csv").relative_to(REPO)),
            "plots": [
                str((args.out_dir / "left_local_scatter.png").relative_to(REPO)),
                str((args.out_dir / "right_local_scatter.png").relative_to(REPO)),
                str((args.out_dir / "vertical_fraction_timeseries.png").relative_to(REPO)),
            ],
        },
    }

    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=_jsonable),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, default=_jsonable))


if __name__ == "__main__":
    main()
