#!/usr/bin/env python3
"""Build E028b manifest for D003 Box021 candidate anchor refit.

E028b keeps the E028 candidate denominator fixed but replaces the canonical
side-face center anchor with a projected robust centroid of the selected-face
contact cloud.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[4]
BASE = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
E028_RESULTS = REPO / "workspace/core4d_collab_retarget/results/E028"
E028_MANIFEST = E028_RESULTS / "manifest.tsv"
E028_CANDIDATES = E028_RESULTS / "candidates.json"
RESULTS = REPO / "workspace/core4d_collab_retarget/results/E028b_anchor_refit"
MANIFEST = RESULTS / "manifest.tsv"

FACES = ["+x", "-x", "+y", "-y", "+z", "-z"]
SIDE_FACES = ["+x", "-x", "+y", "-y"]
FREE_AXIS_CLAMP_FRAC = 0.90
MIN_SELECTED_FACE_POINTS = 12


def _read_manifest(path: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
        return list(reader.fieldnames or []), {row["variant"]: row for row in rows}


def _quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q / np.clip(np.linalg.norm(q), 1e-8, None)
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


def _object_half(task: str) -> np.ndarray:
    model = mujoco.MjModel.from_xml_path(str(BASE / task / "scene.xml"))
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_collision")
    if geom < 0:
        raise ValueError(f"{task} has no object_collision geom")
    return model.geom_size[geom, :3].astype(np.float64)


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


def _active_contact_cloud(row: dict[str, str], half: np.ndarray) -> tuple[np.ndarray, list[str]]:
    task = row["source_task"]
    model = mujoco.MjModel.from_xml_path(str(BASE / task / "scene.xml"))
    qadr = _object_qadr(model)
    traj = np.load(BASE / task / "0/trajectory_kinematic.npz")
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

    points: list[np.ndarray] = []
    labels: list[str] = []
    for i, q in enumerate(qpos):
        obj_pos = q[qadr : qadr + 3].astype(np.float64)
        obj_quat = q[qadr + 3 : qadr + 7].astype(np.float64)
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
            labels.append(_face_label(local, half))
    if not points:
        return np.zeros((0, 3), dtype=np.float64), []
    return np.asarray(points, dtype=np.float64), labels


def _point_from_row(row: dict[str, str]) -> np.ndarray:
    return np.array(
        [
            float(row["support_proxy_point_local_x"]),
            float(row["support_proxy_point_local_y"]),
            float(row["support_proxy_point_local_z"]),
        ],
        dtype=np.float64,
    )


def _robust_centroid(points: np.ndarray) -> np.ndarray:
    if len(points) <= 4:
        return points.mean(axis=0)
    median = np.median(points, axis=0)
    dist = np.linalg.norm(points - median[None, :], axis=1)
    keep = dist <= np.percentile(dist, 80.0)
    if int(keep.sum()) < max(3, len(points) // 3):
        return median
    return points[keep].mean(axis=0)


def _project_to_face(face: str, centroid: np.ndarray, half: np.ndarray) -> np.ndarray:
    point = np.asarray(centroid, dtype=np.float64).copy()
    axis = 0 if face.endswith("x") else 1
    sign = 1.0 if face.startswith("+") else -1.0
    point[axis] = sign * float(half[axis])
    for idx in range(3):
        if idx == axis:
            continue
        limit = FREE_AXIS_CLAMP_FRAC * float(half[idx])
        point[idx] = float(np.clip(point[idx], -limit, limit))
    return point


def _face_stats(labels: list[str]) -> dict[str, Any]:
    counts = {face: labels.count(face) for face in FACES}
    total = max(1, len(labels))
    top_face = max(FACES, key=lambda face: counts[face])
    side_rank = sorted(SIDE_FACES, key=lambda face: counts[face], reverse=True)
    z_rank = sorted(["+z", "-z"], key=lambda face: counts[face], reverse=True)
    side_face = side_rank[0]
    second_side = side_rank[1]
    z_face = z_rank[0]
    return {
        "counts": counts,
        "top_face": top_face,
        "top_face_frac": counts[top_face] / total,
        "side_face": side_face,
        "side_face_frac": counts[side_face] / total,
        "z_face": z_face,
        "z_face_frac": counts[z_face] / total,
        "side_margin": (counts[side_face] - counts[second_side]) / total,
    }


def _review_reasons(row: dict[str, str], stats: dict[str, Any], selected_count: int, point: np.ndarray, half: np.ndarray) -> str:
    reasons: list[str] = []
    old = str(row.get("anchor_face_review_reason", "")).strip()
    if old:
        reasons.extend([item for item in old.split(",") if item])
    if selected_count < MIN_SELECTED_FACE_POINTS:
        reasons.append("selected_face_too_few_points")
    if float(stats["side_face_frac"]) < 0.35:
        reasons.append("weak_side_face_support")
    if float(stats["side_margin"]) < 0.10:
        reasons.append("ambiguous_side_face_margin")
    face = row["anchor_face"]
    axis = 0 if face.endswith("x") else 1
    for idx, name in enumerate("xyz"):
        if idx == axis:
            continue
        if abs(point[idx]) >= 0.85 * float(half[idx]):
            reasons.append(f"projected_centroid_near_{name}_edge")
    out: list[str] = []
    for reason in reasons:
        if reason not in out:
            out.append(reason)
    return ",".join(out)


def _variant_name(e028_variant: str) -> str:
    if not e028_variant.startswith("E028_"):
        raise ValueError(f"Unexpected E028 variant name: {e028_variant}")
    return "E028b_" + e028_variant[len("E028_") :].replace("_canonical_t02", "_contact_centroid_t02")


def _as_str(value: float) -> str:
    return f"{float(value):.8g}"


def build_manifest(
    *,
    e028_manifest: Path,
    candidates_path: Path,
    manifest_path: Path,
) -> list[dict[str, str]]:
    old_fields, old_rows = _read_manifest(e028_manifest)
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    rows: list[dict[str, str]] = []
    new_candidates: list[str] = []
    for old_variant in candidates:
        if old_variant not in old_rows:
            raise KeyError(f"{old_variant} missing from {e028_manifest}")
        old = dict(old_rows[old_variant])
        half = _object_half(old["source_task"])
        points, labels = _active_contact_cloud(old, half)
        stats = _face_stats(labels)
        face = old["anchor_face"]
        selected = np.asarray([label == face for label in labels], dtype=bool)
        selected_points = points[selected] if len(points) else np.zeros((0, 3), dtype=np.float64)
        old_point = _point_from_row(old)
        if len(selected_points) >= MIN_SELECTED_FACE_POINTS:
            selected_centroid = selected_points.mean(axis=0)
            robust = _robust_centroid(selected_points)
            point = _project_to_face(face, robust, half)
            fallback = False
        else:
            selected_centroid = np.full(3, np.nan, dtype=np.float64)
            robust = np.full(3, np.nan, dtype=np.float64)
            point = old_point.copy()
            fallback = True
        old_dist = float(np.linalg.norm(old_point - selected_centroid)) if np.all(np.isfinite(selected_centroid)) else float("nan")
        new_dist = float(np.linalg.norm(point - selected_centroid)) if np.all(np.isfinite(selected_centroid)) else float("nan")
        ratio = float(new_dist / old_dist) if np.isfinite(old_dist) and old_dist > 1e-8 else float("nan")
        variant = _variant_name(old_variant)
        new_candidates.append(variant)

        row = {key: old.get(key, "") for key in old_fields}
        row.update(
            {
                "variant": variant,
                "derived_task": f"{old['source_task']}_freejoint_legobj_e028b",
                "queue": "local",
                "role": "d003_box021_candidates_anchor_refit",
                "wave": "B",
                "scene_name": f"scene_e028b_jointB_{old['source_task']}_contact_centroid_t02",
                "support_proxy_point_local_x": _as_str(point[0]),
                "support_proxy_point_local_y": _as_str(point[1]),
                "support_proxy_point_local_z": _as_str(point[2]),
                "support_point_method": "d003_selected_face_contact_centroid_projected",
                "anchor_policy": "e028b_contact_centroid_projected",
                "anchor_face_source": "d003_contact_pos_object_local_selected_face",
                "anchor_face_review_reason": _review_reasons(old, stats, len(selected_points), point, half),
                "anchor_top_face": str(stats["top_face"]),
                "anchor_top_face_frac": _as_str(float(stats["top_face_frac"])),
                "anchor_side_face": str(stats["side_face"]),
                "anchor_side_face_frac": _as_str(float(stats["side_face_frac"])),
                "anchor_z_face": str(stats["z_face"]),
                "anchor_z_face_frac": _as_str(float(stats["z_face_frac"])),
                "anchor_side_margin": _as_str(float(stats["side_margin"])),
                "contact_points_used": str(len(points)),
                "canonical_z_frac": _as_str(point[2] / half[2] if half[2] > 1e-8 else float("nan")),
                "anchor_audit_class": "e028b_contact_centroid_projected",
                "anchor_current_face": str(stats["top_face"]),
                "anchor_selected_face": face,
                "online_video_path": f"workspace/core4d_collab_retarget/results/E028b_anchor_refit/online_video/{variant}.mp4",
            }
        )
        review_reason = row["anchor_face_review_reason"]
        row["anchor_face_review"] = str(bool(review_reason)).lower()

        row.update(
            {
                "source_e028_variant": old_variant,
                "old_support_proxy_point_local_x": _as_str(old_point[0]),
                "old_support_proxy_point_local_y": _as_str(old_point[1]),
                "old_support_proxy_point_local_z": _as_str(old_point[2]),
                "selected_face_points_used": str(len(selected_points)),
                "selected_face_centroid_x": _as_str(selected_centroid[0]),
                "selected_face_centroid_y": _as_str(selected_centroid[1]),
                "selected_face_centroid_z": _as_str(selected_centroid[2]),
                "selected_face_robust_centroid_x": _as_str(robust[0]),
                "selected_face_robust_centroid_y": _as_str(robust[1]),
                "selected_face_robust_centroid_z": _as_str(robust[2]),
                "old_anchor_to_selected_face_centroid_m": _as_str(old_dist),
                "anchor_to_selected_face_centroid_m": _as_str(new_dist),
                "anchor_distance_ratio_vs_e028": _as_str(ratio),
                "anchor_refit_fallback": str(fallback).lower(),
                "face_count_pos_x": str(stats["counts"]["+x"]),
                "face_count_neg_x": str(stats["counts"]["-x"]),
                "face_count_pos_y": str(stats["counts"]["+y"]),
                "face_count_neg_y": str(stats["counts"]["-y"]),
                "face_count_pos_z": str(stats["counts"]["+z"]),
                "face_count_neg_z": str(stats["counts"]["-z"]),
            }
        )
        rows.append(row)

    extra_fields = [
        "source_e028_variant",
        "old_support_proxy_point_local_x",
        "old_support_proxy_point_local_y",
        "old_support_proxy_point_local_z",
        "selected_face_points_used",
        "selected_face_centroid_x",
        "selected_face_centroid_y",
        "selected_face_centroid_z",
        "selected_face_robust_centroid_x",
        "selected_face_robust_centroid_y",
        "selected_face_robust_centroid_z",
        "old_anchor_to_selected_face_centroid_m",
        "anchor_to_selected_face_centroid_m",
        "anchor_distance_ratio_vs_e028",
        "anchor_refit_fallback",
        "face_count_pos_x",
        "face_count_neg_x",
        "face_count_pos_y",
        "face_count_neg_y",
        "face_count_pos_z",
        "face_count_neg_z",
    ]
    fieldnames = old_fields + [field for field in extra_fields if field not in old_fields]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    (manifest_path.parent / "candidates.json").write_text(
        json.dumps(new_candidates, indent=2), encoding="utf-8"
    )
    (manifest_path.parent / "source_candidates.json").write_text(
        json.dumps(candidates, indent=2), encoding="utf-8"
    )
    summary = {
        "input_candidates": len(candidates),
        "manifest_rows": len(rows),
        "anchor_policy": "e028b_contact_centroid_projected",
        "num_anchor_refit_fallback": sum(row["anchor_refit_fallback"] == "true" for row in rows),
        "mean_anchor_distance_ratio_vs_e028": float(
            np.nanmean([float(row["anchor_distance_ratio_vs_e028"]) for row in rows])
        ),
        "variants": new_candidates,
    }
    (manifest_path.parent / "preflight_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e028-manifest", type=Path, default=E028_MANIFEST)
    parser.add_argument("--candidates", type=Path, default=E028_CANDIDATES)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    rows = build_manifest(
        e028_manifest=args.e028_manifest,
        candidates_path=args.candidates,
        manifest_path=args.manifest,
    )
    print(f"Wrote {args.manifest.resolve().relative_to(REPO)} rows={len(rows)}")
    for row in rows:
        print(
            f"{row['variant']}: source={row['source_e028_variant']} "
            f"face={row['anchor_face']} point=[{row['support_proxy_point_local_x']},"
            f"{row['support_proxy_point_local_y']},{row['support_proxy_point_local_z']}] "
            f"old_dist={row['old_anchor_to_selected_face_centroid_m']} "
            f"new_dist={row['anchor_to_selected_face_centroid_m']} "
            f"ratio={row['anchor_distance_ratio_vs_e028']} review={row['anchor_face_review_reason']}"
        )


if __name__ == "__main__":
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    main()
