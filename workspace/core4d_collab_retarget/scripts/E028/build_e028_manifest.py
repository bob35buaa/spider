#!/usr/bin/env python3
"""Build the E028 D003 Box021 13-case manifest.

The input is the Holosoma D003 OmniRetarget + SPIDER preprocess summary.  This
script only selects D003 pass rows and estimates one canonical support-proxy
side face per new D003 task from the task's kinematic contact points.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[4]
BASE = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
RESULTS = REPO / "workspace/core4d_collab_retarget/results/E028"
MANIFEST = RESULTS / "manifest.tsv"
HOLOSOMA_ROOT = Path("/home/ubuntu/Workspace/holosoma")
D003_SUMMARY = (
    HOLOSOMA_ROOT
    / "workspace/v3/data_constructon/d003_omniretarget_spider_production/d003_omniretarget_spider_summary.tsv"
)
D003_CONTACT_MASKS = (
    HOLOSOMA_ROOT
    / "workspace/v3/data_constructon/d003_omniretarget_spider_production/results/contact_masks"
)

CANONICAL_Z_FRAC = 0.62
ORDER = [
    "d003_box021_20231011_034_p1",
    "d003_box021_20231011_035_p1",
    "d003_box021_20231011_035_p2",
    "d003_box021_20231018_028_p2",
    "d003_box021_20231018_029_p1",
    "d003_box021_20231018_029_p2",
    "d003_box021_20231018_030_p1",
    "d003_box021_20231018_030_p2",
    "d003_box021_20231018_031_p2",
    "d003_box021_20231020_019_p1",
    "d003_box021_20231020_019_p2",
    "d003_box021_20231020_020_p1",
    "d003_box021_20231020_020_p2",
]

FIELDNAMES = [
    "variant",
    "source_task",
    "derived_task",
    "mask_source_exp",
    "mask_slug",
    "person_idx",
    "queue",
    "role",
    "wave",
]

MANIFEST_FIELDS = FIELDNAMES + [
    "scene_name",
    "support_proxy_point_local_x",
    "support_proxy_point_local_y",
    "support_proxy_point_local_z",
    "weld_solref_timeconst",
    "weld_solimp_1",
    "weld_solimp_2",
    "weld_solimp_width",
    "support_proxy_gravity_scale",
    "hold_contact_rew_scale",
    "hold_contact_sigma",
    "hold_contact_start_eval_time",
    "hold_contact_end_eval_time",
    "hold_contact_require_ref_contact",
    "support_point_method",
    "source_variant",
    "anchor_policy",
    "anchor_face",
    "anchor_face_source",
    "anchor_face_review",
    "anchor_face_review_reason",
    "anchor_top_face",
    "anchor_top_face_frac",
    "anchor_side_face",
    "anchor_side_face_frac",
    "anchor_z_face",
    "anchor_z_face_frac",
    "anchor_side_margin",
    "contact_points_used",
    "canonical_z_frac",
    "object_half_x",
    "object_half_y",
    "object_half_z",
    "gt_anchor_available",
    "gt_anchor_x",
    "gt_anchor_y",
    "gt_anchor_z",
    "gt_anchor_face",
    "gt_anchor_dist_m",
    "anchor_audit_class",
    "anchor_current_face",
    "anchor_selected_face",
    "anchor_selected_partner_top_relation",
    "anchor_centroid_cancellation",
    "anchor_low_support",
    "reference_frames",
    "d003_sequence",
    "d003_person",
    "d003_object",
    "d003_trimmed_frames",
    "d003_trim_start",
    "d003_contact_mask_raw_shape",
    "d003_contact_mask_ref_shape",
    "d003_contact_mask_eval_shape",
    "trajectory_path",
    "mask_path_source",
    "online_video_path",
]


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = q.astype(np.float64)
    q = q / np.clip(np.linalg.norm(q), 1e-8, None)
    qvec = q[1:]
    t = 2.0 * np.cross(qvec, v)
    return v + q[0] * t + np.cross(qvec, t)


def _quat_apply_inv(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qi = q.astype(np.float64).copy()
    qi = qi / np.clip(np.linalg.norm(qi), 1e-8, None)
    qi[1:] *= -1.0
    return _quat_apply(qi, v)


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


def _canonical_anchor(face: str, half: np.ndarray) -> np.ndarray:
    point = np.zeros(3, dtype=np.float64)
    axis = 0 if face.endswith("x") else 1
    sign = 1.0 if face.startswith("+") else -1.0
    point[axis] = sign * float(half[axis])
    point[2] = float(CANONICAL_Z_FRAC * half[2])
    return point


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


def _estimate_anchor(task: str, mask_npz: Path, person_idx: int) -> dict[str, object]:
    traj_path = BASE / task / "0/trajectory_kinematic.npz"
    traj = np.load(traj_path)
    qpos = traj["qpos"].reshape(-1, traj["qpos"].shape[-1]).astype(np.float64)
    contact_pos = traj["contact_pos"].reshape(qpos.shape[0], 2, 3).astype(np.float64)
    contact = traj["contact"].reshape(qpos.shape[0], 2).astype(bool) if "contact" in traj else None
    mask = _mask_for_person(mask_npz, person_idx, qpos.shape[0])
    half = _object_half(task)

    points: list[np.ndarray] = []
    for i, q in enumerate(qpos):
        obj_pos = q[-7:-4].astype(np.float64)
        obj_quat = q[-4:].astype(np.float64)
        active = mask[i] if mask is not None else (contact[i] if contact is not None else np.ones(2, dtype=bool))
        if contact is not None:
            active = np.logical_or(active, contact[i])
        for hand_idx in range(2):
            if not bool(active[hand_idx]):
                continue
            p_world = contact_pos[i, hand_idx]
            if not np.all(np.isfinite(p_world)):
                continue
            local = _quat_apply_inv(obj_quat, p_world - obj_pos)
            if np.all(np.isfinite(local)):
                points.append(local)

    if not points:
        face = "+y"
        point = _canonical_anchor(face, half)
        return {
            "face": face,
            "point": point,
            "method": "fallback_no_active_contact",
            "review": True,
            "review_reason": "no_active_contact_points",
            "contact_points_used": 0,
            "top_face": "",
            "top_face_frac": 0.0,
            "side_face": face,
            "side_face_frac": 0.0,
            "z_face": "",
            "z_face_frac": 0.0,
            "side_margin": 0.0,
            "half": half,
        }

    labels = [_face_label(p, half) for p in points]
    counts = {face: labels.count(face) for face in ["+x", "-x", "+y", "-y", "+z", "-z"]}
    total = float(len(points))
    top_face = max(counts, key=lambda k: counts[k])
    side_faces = ["+x", "-x", "+y", "-y"]
    z_faces = ["+z", "-z"]
    side_rank = sorted(side_faces, key=lambda k: counts[k], reverse=True)
    z_rank = sorted(z_faces, key=lambda k: counts[k], reverse=True)
    side_face = side_rank[0]
    z_face = z_rank[0]
    side_frac = counts[side_face] / total
    second_side_frac = counts[side_rank[1]] / total
    z_frac = counts[z_face] / total
    top_frac = counts[top_face] / total
    side_margin = side_frac - second_side_frac
    review_reasons: list[str] = []
    if top_face in {"+z", "-z"}:
        review_reasons.append(f"top_face_is_{top_face}")
    if side_frac < 0.35:
        review_reasons.append("weak_side_face_support")
    if side_margin < 0.10:
        review_reasons.append("ambiguous_side_face_margin")
    if z_frac > side_frac:
        review_reasons.append("z_face_exceeds_side_face")
    face = side_face
    point = _canonical_anchor(face, half)
    return {
        "face": face,
        "point": point,
        "method": "d003_contact_pos_side_face_canonical_z",
        "review": bool(review_reasons),
        "review_reason": ",".join(review_reasons),
        "contact_points_used": len(points),
        "top_face": top_face,
        "top_face_frac": top_frac,
        "side_face": side_face,
        "side_face_frac": side_frac,
        "z_face": z_face,
        "z_face_frac": z_frac,
        "side_margin": side_margin,
        "half": half,
    }


def _check_source(row: dict[str, str], mask_npz: Path) -> list[str]:
    task = row["target_task"]
    task_dir = BASE / task
    missing: list[str] = []
    for rel in ["scene.xml", "scene_act.xml", "0/trajectory_kinematic.npz", "task_info.json"]:
        if not (task_dir / rel).is_file():
            missing.append(rel)
    if not mask_npz.is_file():
        missing.append(str(mask_npz))
    if missing:
        return missing
    model = mujoco.MjModel.from_xml_path(str(task_dir / "scene.xml"))
    if (model.nq, model.nv, model.nu) != (43, 41, 29):
        missing.append(f"unexpected_scene_dims={model.nq}/{model.nv}/{model.nu}")
    traj = np.load(task_dir / "0/trajectory_kinematic.npz")
    for key in ["qpos", "qvel", "ctrl", "contact", "contact_pos"]:
        if key not in traj:
            missing.append(f"missing_npz_key={key}")
            continue
        if np.isnan(traj[key]).any():
            missing.append(f"nan_in_{key}")
    return missing


def _selected_rows(summary_path: Path) -> list[dict[str, str]]:
    rows = [r for r in _read_tsv(summary_path) if r.get("d003_decision_group") == "进入下一轮"]
    by_task = {r["target_task"]: r for r in rows}
    ordered = [by_task[name] for name in ORDER if name in by_task]
    extra = sorted([r for r in rows if r["target_task"] not in ORDER], key=lambda r: r["target_task"])
    return ordered + extra


def build_manifest(summary_path: Path, manifest_path: Path) -> list[dict[str, str]]:
    selected = _selected_rows(summary_path)
    if len(selected) != 13:
        raise RuntimeError(f"Expected 13 D003 pass rows, got {len(selected)}")
    rows: list[dict[str, str]] = []
    rejected: list[dict[str, object]] = []
    for src in selected:
        task = src["target_task"]
        person_idx = 0 if src["person"] == "person1" else 1
        mask_npz = D003_CONTACT_MASKS / task / "raw_contact_mask_3cm.npz"
        problems = _check_source(src, mask_npz)
        if problems:
            rejected.append({"target_task": task, "problems": problems})
            continue
        anchor = _estimate_anchor(task, mask_npz, person_idx)
        point = np.asarray(anchor["point"], dtype=np.float64)
        half = np.asarray(anchor["half"], dtype=np.float64)
        variant = f"E028_{task}_canonical_t02"
        rows.append(
            {
                "variant": variant,
                "source_task": task,
                "derived_task": f"{task}_freejoint_legobj_e028",
                "mask_source_exp": "holosoma_d003",
                "mask_slug": task,
                "person_idx": str(person_idx),
                "queue": "local",
                "role": "d003_box021_13case",
                "wave": "A",
                "scene_name": f"scene_e028_jointB_{task}_canonical_t02",
                "support_proxy_point_local_x": f"{point[0]:.8g}",
                "support_proxy_point_local_y": f"{point[1]:.8g}",
                "support_proxy_point_local_z": f"{point[2]:.8g}",
                "weld_solref_timeconst": "0.02",
                "weld_solimp_1": "0.9",
                "weld_solimp_2": "0.95",
                "weld_solimp_width": "0.001",
                "support_proxy_gravity_scale": "0.5",
                "hold_contact_rew_scale": "0.0",
                "hold_contact_sigma": "0.05",
                "hold_contact_start_eval_time": "0.64",
                "hold_contact_end_eval_time": "4.08",
                "hold_contact_require_ref_contact": "true",
                "support_point_method": str(anchor["method"]),
                "source_variant": task,
                "anchor_policy": "d003_estimated_side_face_center_upper",
                "anchor_face": str(anchor["face"]),
                "anchor_face_source": "d003_contact_pos_object_local",
                "anchor_face_review": str(bool(anchor["review"])).lower(),
                "anchor_face_review_reason": str(anchor["review_reason"]),
                "anchor_top_face": str(anchor["top_face"]),
                "anchor_top_face_frac": f"{float(anchor['top_face_frac']):.8g}",
                "anchor_side_face": str(anchor["side_face"]),
                "anchor_side_face_frac": f"{float(anchor['side_face_frac']):.8g}",
                "anchor_z_face": str(anchor["z_face"]),
                "anchor_z_face_frac": f"{float(anchor['z_face_frac']):.8g}",
                "anchor_side_margin": f"{float(anchor['side_margin']):.8g}",
                "contact_points_used": str(int(anchor["contact_points_used"])),
                "canonical_z_frac": f"{CANONICAL_Z_FRAC:.8g}",
                "object_half_x": f"{half[0]:.8g}",
                "object_half_y": f"{half[1]:.8g}",
                "object_half_z": f"{half[2]:.8g}",
                "gt_anchor_available": "false",
                "gt_anchor_x": "",
                "gt_anchor_y": "",
                "gt_anchor_z": "",
                "gt_anchor_face": "",
                "gt_anchor_dist_m": "nan",
                "anchor_audit_class": "d003_estimated",
                "anchor_current_face": str(anchor["top_face"]),
                "anchor_selected_face": str(anchor["face"]),
                "anchor_selected_partner_top_relation": "",
                "anchor_centroid_cancellation": "false",
                "anchor_low_support": "false",
                "reference_frames": src["trimmed_frames"],
                "d003_sequence": src["sequence"],
                "d003_person": src["person"],
                "d003_object": src["object_name"],
                "d003_trimmed_frames": src["trimmed_frames"],
                "d003_trim_start": src["trim_start"],
                "d003_contact_mask_raw_shape": src["contact_mask_raw_shape"],
                "d003_contact_mask_ref_shape": src["contact_mask_ref_shape"],
                "d003_contact_mask_eval_shape": src["contact_mask_eval_shape"],
                "trajectory_path": str(BASE / task / "0/trajectory_kinematic.npz"),
                "mask_path_source": str(mask_npz),
                "online_video_path": f"workspace/core4d_collab_retarget/results/E028/online_video/{variant}.mp4",
            }
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "input_rows": len(selected),
        "manifest_rows": len(rows),
        "boundary_anchor_review": sum(r["anchor_face_review"] == "true" for r in rows),
        "discarded_or_skipped": len(rejected),
        "rejected": rejected,
        "variants": [r["variant"] for r in rows],
    }
    (manifest_path.parent / "preflight_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=D003_SUMMARY)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    rows = build_manifest(args.summary, args.manifest)
    print(f"Wrote {args.manifest.relative_to(REPO)} rows={len(rows)}")
    for row in rows:
        review = " review=" + row["anchor_face_review_reason"] if row["anchor_face_review"] == "true" else ""
        print(
            f"{row['variant']}: task={row['source_task']} frames={row['reference_frames']} "
            f"face={row['anchor_face']} points={row['contact_points_used']}{review}"
        )


if __name__ == "__main__":
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    main()
