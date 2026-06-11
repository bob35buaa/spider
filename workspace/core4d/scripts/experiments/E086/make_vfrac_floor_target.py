#!/usr/bin/env python3
"""Create E086 target variants by clamping target world-vertical fraction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[4]
E085_DIR = REPO / "workspace/core4d/scripts/E085"
if str(E085_DIR) not in sys.path:
    sys.path.insert(0, str(E085_DIR))

from generate_raw_contact_targets import (  # noqa: E402
    FACE_NAMES,
    face_label,
    load_scene_geometry,
    project_to_box_surface,
    quat_wxyz_to_mat,
    surface_dist,
    vertical_fraction,
)


SRC = REPO / "workspace/core4d/results/E085/raw_targets/E085A_rawtarget_main/raw_contact_targets.npz"
MASK = REPO / "workspace/core4d/results/E084/contact_masks/d003_box021_20231018_029_p2/raw_contact_mask_3cm.npz"
OUT_DIR = REPO / "workspace/core4d/results/E086/vfrac_floor_targets/E086B_vfrac_floor_main"
TASK = "d003_box021_20231018_029_p2_upperobj_e083"
HAND_TO_INDEX = {"left": 0, "right": 1}
FACE_TO_ID = {name: i for i, name in enumerate(FACE_NAMES)}


def corner_z_range(obj_pos: np.ndarray, obj_mat: np.ndarray, half: np.ndarray) -> tuple[float, float]:
    corners = np.array(
        [
            [sx, sy, sz]
            for sx in (-half[0], half[0])
            for sy in (-half[1], half[1])
            for sz in (-half[2], half[2])
        ],
        dtype=np.float64,
    )
    z = (corners @ obj_mat.T + obj_pos)[:, 2]
    return float(z.min()), float(z.max())


def clamp_one(
    local: np.ndarray,
    raw_idx: int,
    obj_poses: np.ndarray,
    visual_pos: np.ndarray,
    mesh_rot: np.ndarray,
    half: np.ndarray,
    min_vfrac: float,
) -> tuple[np.ndarray, bool]:
    R_raw = obj_poses[raw_idx, :3, :3]
    t_raw = obj_poses[raw_idx, :3, 3]
    obj_pos = visual_pos @ R_raw.T + t_raw
    obj_mat = R_raw @ mesh_rot
    vf = vertical_fraction(local, obj_pos, obj_mat, half)
    if vf >= min_vfrac:
        return local.copy(), False
    z_min, z_max = corner_z_range(obj_pos, obj_mat, half)
    target_world = local @ obj_mat.T + obj_pos
    target_world[2] = z_min + min_vfrac * max(z_max - z_min, 1e-9)
    raised_local = (target_world - obj_pos) @ obj_mat
    return project_to_box_surface(raised_local, half), True


def clamp_array(
    target: np.ndarray,
    valid: np.ndarray,
    raw_indices: np.ndarray,
    obj_poses: np.ndarray,
    visual_pos: np.ndarray,
    mesh_rot: np.ndarray,
    half: np.ndarray,
    hand_idx: int,
    min_vfrac: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    out = target.copy()
    face_id = np.full(valid.shape, FACE_TO_ID[""], dtype=np.int16)
    vfrac = np.full(valid.shape, np.nan, dtype=np.float32)
    surf_dist = np.full(valid.shape, np.nan, dtype=np.float32)
    changed = 0
    for i, raw_idx in enumerate(raw_indices):
        R_raw = obj_poses[int(raw_idx), :3, :3]
        t_raw = obj_poses[int(raw_idx), :3, 3]
        obj_pos = visual_pos @ R_raw.T + t_raw
        obj_mat = R_raw @ mesh_rot
        for hi in range(target.shape[1]):
            local = out[i, hi].astype(np.float64)
            if hi == hand_idx and bool(valid[i, hi]):
                local, did_change = clamp_one(
                    local,
                    int(raw_idx),
                    obj_poses,
                    visual_pos,
                    mesh_rot,
                    half,
                    min_vfrac,
                )
                out[i, hi] = local.astype(out.dtype)
                changed += int(did_change)
            face = face_label(local, half)
            face_id[i, hi] = FACE_TO_ID[face]
            vfrac[i, hi] = vertical_fraction(local, obj_pos, obj_mat, half)
            surf_dist[i, hi] = surface_dist(local, half)
    return out, face_id, vfrac, surf_dist, changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=SRC)
    parser.add_argument("--mask", type=Path, default=MASK)
    parser.add_argument("--task", default=TASK)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--hand", choices=["left", "right"], default="left")
    parser.add_argument("--min-vfrac", type=float, default=0.20)
    args = parser.parse_args()

    src = args.src if args.src.is_absolute() else REPO / args.src
    mask_path = args.mask if args.mask.is_absolute() else REPO / args.mask
    out_dir = args.out_dir if args.out_dir.is_absolute() else REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(src, allow_pickle=True)
    mask = np.load(mask_path, allow_pickle=True)
    audit = json.loads((mask_path.parent / "audit_summary_3cm.json").read_text())
    obj_poses = np.load(Path(audit["seq_dir"]) / "smooth_objposes.npy")
    geom = load_scene_geometry(args.task)
    half = geom["half"]
    visual_pos = geom["visual_pos"]
    mesh_rot = quat_wxyz_to_mat(geom["visual_quat"])
    hand_idx = HAND_TO_INDEX[args.hand]

    raw_len = data["raw_contact_target_object_local"].shape[0]
    spider_len = data["spider_contact_target_object_local"].shape[0]
    eval_raw_idx = data["eval_raw_idx"].astype(np.int64)
    trim_start = int(data["trim_start"])

    raw_idx = np.arange(raw_len, dtype=np.int64)
    spider_idx = trim_start + np.arange(spider_len, dtype=np.int64)

    raw_target, raw_face, raw_vfrac, raw_dist, raw_changed = clamp_array(
        data["raw_contact_target_object_local"],
        data["raw_target_valid"],
        raw_idx,
        obj_poses,
        visual_pos,
        mesh_rot,
        half,
        hand_idx,
        args.min_vfrac,
    )
    spider_target, spider_face, spider_vfrac, spider_dist, spider_changed = clamp_array(
        data["spider_contact_target_object_local"],
        data["spider_target_valid"],
        spider_idx,
        obj_poses,
        visual_pos,
        mesh_rot,
        half,
        hand_idx,
        args.min_vfrac,
    )
    eval_target, eval_face, eval_vfrac, eval_dist, eval_changed = clamp_array(
        data["eval_contact_target_object_local"],
        data["eval_target_valid"],
        eval_raw_idx,
        obj_poses,
        visual_pos,
        mesh_rot,
        half,
        hand_idx,
        args.min_vfrac,
    )

    out_path = out_dir / "raw_contact_targets.npz"
    payload = {key: data[key] for key in data.files}
    payload.update(
        raw_contact_target_object_local=raw_target,
        spider_contact_target_object_local=spider_target,
        eval_contact_target_object_local=eval_target,
        raw_target_face_id=raw_face,
        spider_target_face_id=spider_face,
        eval_target_face_id=eval_face,
        raw_target_vertical_frac=raw_vfrac,
        spider_target_vertical_frac=spider_vfrac,
        eval_target_vertical_frac=eval_vfrac,
        raw_target_surface_dist_m=raw_dist,
        spider_target_surface_dist_m=spider_dist,
        eval_target_surface_dist_m=eval_dist,
        target_kind=np.array(f"nearest_face_{args.hand}_vfrac_floor_{args.min_vfrac:.2f}"),
    )
    np.savez(out_path, **payload)

    summary = {
        "source": str(src.relative_to(REPO)),
        "output": str(out_path.relative_to(REPO)),
        "task": args.task,
        "hand": args.hand,
        "min_vfrac": args.min_vfrac,
        "changed": {
            "raw": raw_changed,
            "spider": spider_changed,
            "eval": eval_changed,
        },
        "eval_hand_vfrac_mean_min": [
            float(np.nanmean(eval_vfrac[:, hand_idx])),
            float(np.nanmin(eval_vfrac[:, hand_idx])),
        ],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
