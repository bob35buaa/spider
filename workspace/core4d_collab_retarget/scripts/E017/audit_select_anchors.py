#!/usr/bin/env python3
"""E017 anchor audit and candidate scene generation.

This keeps E014's soft-weld structure, but replaces E016's single centroid
anchor with a pre-retarget audit plus a face-cluster fallback for centroid
cancellation cases.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[4]
BASE = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
E016_DIR = REPO / "workspace/core4d_collab_retarget/scripts/E016"
if str(E016_DIR) not in sys.path:
    sys.path.insert(0, str(E016_DIR))

import generate_e016_assets as e016_assets  # noqa: E402


E016_VARIANTS = REPO / "workspace/core4d_collab_retarget/scripts/E016/variants.tsv"
E016_MANIFEST = REPO / "workspace/core4d_collab_retarget/results/E016/manifest.tsv"
E016_COMPARISON = REPO / "workspace/core4d_collab_retarget/results/E016/comparison.csv"
RESULTS = REPO / "workspace/core4d_collab_retarget/results/E017"
MANIFEST = RESULTS / "manifest.tsv"
AUDIT_CSV = RESULTS / "anchor_audit.csv"
AUDIT_JSON = RESULTS / "anchor_audit_summary.json"
METHOD_AUDIT_CSV = RESULTS / "anchor_method_audit.csv"
ATTRIBUTION_CSV = RESULTS / "e016_anchor_failure_attribution.csv"

ANCHOR_BODY = "support_weld_anchor"
ANCHOR_GEOM = "support_weld_anchor_geom"
WELD_NAME = "e017_support_weld"
# B1 修复 (exp_diagnostic_v2 §2)：FACE_ORDER 从 xy-only 扩展到全 6 面，
# 老 face_label 永远不返回 ±z 是 wrong-target，导致 box021 D003 9/13 case
# anchor_face_review=true。统一改用 face_utils.FACE_ORDER。
FACE_ORDER = ["+x", "-x", "+y", "-y", "+z", "-z"]
OPPOSITE_FACE = {
    "+x": "-x", "-x": "+x",
    "+y": "-y", "-y": "+y",
    "+z": "-z", "-z": "+z",
}
PERSON_RE = re.compile(r"_person([12])(?=$|_)")

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
    "support_point_method",
    "source_variant",
    "anchor_policy",
    "anchor_audit_class",
    "anchor_current_face",
    "anchor_selected_face",
    "anchor_current_support_frac",
    "anchor_top_face",
    "anchor_top_face_frac",
    "anchor_opposed_top2",
]

E014_MANUAL_SEEDS = {
    "box023_person2": np.array([0.16, 0.0, 0.10], dtype=np.float64),
    "box025_person2": np.array([0.0, 0.38, 0.30], dtype=np.float64),
}


def _read_tsv_rows(path: Path, fieldnames: list[str] | None = None) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = [line for line in f if line.strip() and not line.startswith("#")]
    if fieldnames is None:
        return list(csv.DictReader(rows, delimiter="\t"))
    return list(csv.DictReader(rows, delimiter="\t", fieldnames=fieldnames))


def _read_csv_by_key(path: Path, key: str) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        return {row[key]: row for row in csv.DictReader(f)}


def _mask_path(row: dict[str, str]) -> Path:
    return (
        REPO
        / "workspace/core4d/results"
        / row["mask_source_exp"]
        / "contact_masks"
        / row["mask_slug"]
        / "raw_contact_mask_3cm.npz"
    )


def _load_mask_for_person(row: dict[str, str], person_idx: int, target_len: int) -> np.ndarray | None:
    mask_path = _mask_path(row)
    if not mask_path.is_file():
        return None
    data = np.load(mask_path, allow_pickle=True)
    key = "eval_contact_mask_3cm" if "eval_contact_mask_3cm" in data else "spider_contact_mask_3cm"
    raw = data[key]
    person_idx = min(max(int(person_idx), 0), raw.shape[1] - 1)
    mask = raw[:, person_idx, :2].astype(bool)
    if len(mask) == target_len:
        return mask
    idx = np.round(np.linspace(0, len(mask) - 1, target_len)).astype(int)
    return mask[idx]


def _source_person_num(source_task: str) -> int | None:
    match = PERSON_RE.search(source_task)
    if match is None:
        return None
    return int(match.group(1))


def _counterpart_source_task(source_task: str) -> tuple[str, int | None]:
    match = PERSON_RE.search(source_task)
    if match is None:
        return "", None
    current = int(match.group(1))
    other = 1 if current == 2 else 2
    task = f"{source_task[:match.start(1)]}{other}{source_task[match.end(1):]}"
    return task, other - 1


def _point_from_manifest(row: dict[str, str]) -> np.ndarray:
    return np.array(
        [
            float(row["support_proxy_point_local_x"]),
            float(row["support_proxy_point_local_y"]),
            float(row["support_proxy_point_local_z"]),
        ],
        dtype=np.float64,
    )


def _format_point(point: np.ndarray) -> tuple[str, str, str]:
    return (f"{point[0]:.8g}", f"{point[1]:.8g}", f"{point[2]:.8g}")


def _quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return e016_assets._quat_apply(q, v)


def _quat_apply_inv(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return e016_assets._quat_apply_inv(q, v)


def _remove_named_children(parent: ET.Element, tag: str, names: set[str]) -> None:
    for child in list(parent.findall(tag)):
        if child.get("name") in names:
            parent.remove(child)


def _indent(elem: ET.Element, level: int = 0) -> None:
    spacer = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = spacer + "  "
        for child in elem:
            _indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = spacer
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = spacer


def face_label(point: np.ndarray, half: np.ndarray) -> str:
    """B1 修复：全 3D argmax，可返回 ±z；老版仅 xy。"""
    norm = np.abs(point) / np.clip(half, 1e-6, None)
    axis = int(np.argmax(norm))
    sign = "+" if point[axis] >= 0.0 else "-"
    return f"{sign}{'xyz'[axis]}"


def _face_axis_sign(face: str) -> tuple[int, float]:
    """B1 修复：支持全 6 面；老版 ±z 静默映射到 y。"""
    axis = "xyz".index(face[1])
    sign = 1.0 if face[0] == "+" else -1.0
    return axis, sign


def _face_sort_key(item: tuple[str, int]) -> tuple[int, int]:
    face, count = item
    return (-count, FACE_ORDER.index(face) if face in FACE_ORDER else 99)


def _clip_anchor(point: np.ndarray, half: np.ndarray, *, z_upper_frac: float = 0.65) -> np.ndarray:
    """B1 修复：3 个轴对称裁剪。

    旧版 z 强压到 [-0.25·hz, 0.65·hz]，意味着 anchor 永远不能在底面或顶面，
    与"接触点投票主面 +z 主导"的下游需求互斥。新版默认对称 0.90·half；
    z_upper_frac 保留参数但不再非对称裁剪，仅作为输入兼容。
    """
    out = np.asarray(point, dtype=np.float64).copy()
    out = np.clip(out, -0.90 * half, 0.90 * half)
    return out


def _snap_to_face(point: np.ndarray, half: np.ndarray, face: str) -> np.ndarray:
    """B1 修复：用 3-轴循环代替 `other = 1 - axis`，支持 ±z 面。"""
    out = _clip_anchor(point, half)
    axis, sign = _face_axis_sign(face)
    out[axis] = sign * half[axis]
    others = [i for i in (0, 1, 2) if i != axis]
    for j in others:
        out[j] = float(np.clip(out[j], -0.65 * half[j], 0.65 * half[j]))
    return out


def _load_support_points(
    row: dict[str, str],
    task_dir: Path,
    *,
    person_idx: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = mujoco.MjModel.from_xml_path(str(task_dir / "scene.xml"))
    data = mujoco.MjData(model)
    qpos = np.load(task_dir / "0/trajectory_kinematic.npz")["qpos"]
    qpos = qpos.reshape(-1, qpos.shape[-1])
    if person_idx is None:
        person_idx = int(row["person_idx"])
    mask = _load_mask_for_person(row, person_idx, len(qpos))

    left_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_palm")
    right_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_palm")
    obj_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    obj_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_collision")
    half = model.geom_size[obj_geom, :3].astype(np.float64)

    points: list[np.ndarray] = []
    times: list[int] = []
    hands: list[int] = []
    for i, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        obj_pos = data.xpos[obj_body].copy()
        obj_quat = q[-4:].astype(np.float64)
        active_left = mask is None or bool(mask[i, 0])
        active_right = mask is None or bool(mask[i, 1])
        if active_left:
            points.append(_quat_apply_inv(obj_quat, data.site_xpos[left_site] - obj_pos))
            times.append(i)
            hands.append(0)
        if active_right:
            points.append(_quat_apply_inv(obj_quat, data.site_xpos[right_site] - obj_pos))
            times.append(i)
            hands.append(1)
    if not points:
        return np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=np.int64), half
    return np.stack(points).astype(np.float64), np.asarray(times, dtype=np.int64), half


def _face_stats(points: np.ndarray, half: np.ndarray) -> tuple[list[str], Counter[str], str, float, str, float]:
    labels = [face_label(p, half) for p in points]
    counts = Counter(labels)
    for face in FACE_ORDER:
        counts.setdefault(face, 0)
    n = max(len(labels), 1)
    sorted_faces = sorted(counts.items(), key=_face_sort_key)
    top_face, top_count = sorted_faces[0]
    second_face, second_count = sorted_faces[1]
    return labels, counts, top_face, top_count / n, second_face, second_count / n


def _empty_face_stats() -> tuple[list[str], Counter[str], str, float, str, float]:
    counts: Counter[str] = Counter()
    for face in FACE_ORDER:
        counts.setdefault(face, 0)
    return [], counts, "", 0.0, "", 0.0


def _entropy(counts: Counter[str], n: int) -> float:
    if n <= 0:
        return 0.0
    ent = 0.0
    for face in FACE_ORDER:
        p = counts[face] / n
        if p > 0:
            ent -= p * math.log(p, 2)
    return ent


def _cluster_anchor(points: np.ndarray, labels: list[str], half: np.ndarray, face: str) -> np.ndarray:
    if len(points) == 0:
        fallback = np.array([0.0, half[1], max(0.08, 0.60 * half[2])], dtype=np.float64)
        return _snap_to_face(fallback, half, "+y")
    cluster = points[np.asarray(labels) == face]
    if len(cluster) == 0:
        cluster = points
    center = np.median(cluster, axis=0)
    return _snap_to_face(center, half, face)


def _centroid_v2_anchor(current: np.ndarray, half: np.ndarray) -> np.ndarray:
    face = face_label(current, half)
    return _snap_to_face(current, half, face)


def _dist_stats(anchor: np.ndarray, points: np.ndarray) -> tuple[float, float, float]:
    if len(points) == 0:
        return float("nan"), float("nan"), float("nan")
    dist = np.linalg.norm(points - anchor[None, :], axis=1)
    return float(np.min(dist)), float(np.median(dist)), float(np.mean(dist))


def _float_or_nan(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _manual_seed_for(row: dict[str, str], half: np.ndarray) -> np.ndarray | None:
    seed = E014_MANUAL_SEEDS.get(row["source_task"])
    if seed is None:
        return None
    return seed.astype(np.float64).copy()


def _support_frac_from_audit(audit: dict[str, Any], face: str) -> float:
    n = int(audit["n_active_points"])
    if n <= 0:
        return 0.0
    key = {
        "+x": "face_count_pos_x",
        "-x": "face_count_neg_x",
        "+y": "face_count_pos_y",
        "-y": "face_count_neg_y",
    }[face]
    return float(audit[key]) / n


def _support_frac_from_prefixed_counts(audit: dict[str, Any], prefix: str, face: str) -> float:
    n = int(audit.get(f"{prefix}_n_active_points", 0) or 0)
    if n <= 0 or face not in FACE_ORDER:
        return 0.0
    suffix = {
        "+x": "face_count_pos_x",
        "-x": "face_count_neg_x",
        "+y": "face_count_pos_y",
        "-y": "face_count_neg_y",
    }[face]
    return float(audit.get(f"{prefix}_{suffix}", 0) or 0) / n


def _partner_side_status(audit: dict[str, Any], anchor_face: str) -> str:
    if not bool(audit.get("partner_task_available", False)):
        return "partner_task_missing"
    if int(audit.get("partner_n_active_points", 0) or 0) <= 0:
        return "partner_no_active_points"
    partner_top = str(audit.get("partner_top_face", ""))
    partner_top_frac = _float_or_nan(audit.get("partner_top_face_frac", ""))
    if not partner_top or partner_top_frac < 0.25:
        return "partner_face_ambiguous"
    return "partner_face_match" if anchor_face == partner_top else "partner_face_mismatch"


def _method_variant(source_variant: str, policy: str) -> str:
    slug = _slug_from_e016_variant(source_variant)
    if policy == "face_cluster":
        return f"E017_{slug}_face_cluster"
    return f"E017_{slug}_centroid_v2"


def _method_anchor_rows(audit: dict[str, Any]) -> list[dict[str, Any]]:
    half = np.array(
        [float(audit["object_half_x"]), float(audit["object_half_y"]), float(audit["object_half_z"])],
        dtype=np.float64,
    )
    gt = None
    if audit["manual_seed_available"]:
        gt = np.array(
            [float(audit["manual_seed_x"]), float(audit["manual_seed_y"]), float(audit["manual_seed_z"])],
            dtype=np.float64,
        )
    gt_face = face_label(gt, half) if gt is not None else ""
    methods = [
        (
            "E016_centroid",
            str(audit["source_variant"]),
            np.array(
                [float(audit["current_anchor_x"]), float(audit["current_anchor_y"]), float(audit["current_anchor_z"])],
                dtype=np.float64,
            ),
        ),
        (
            "E017_auto",
            _method_variant(str(audit["source_variant"]), str(audit["selected_policy"])),
            np.array(
                [float(audit["selected_anchor_x"]), float(audit["selected_anchor_y"]), float(audit["selected_anchor_z"])],
                dtype=np.float64,
            ),
        ),
    ]
    rows: list[dict[str, Any]] = []
    for method, variant, anchor in methods:
        anchor_face = face_label(anchor, half)
        support_frac = _support_frac_from_audit(audit, anchor_face)
        partner_support_frac = _support_frac_from_prefixed_counts(audit, "partner", anchor_face)
        partner_status = _partner_side_status(audit, anchor_face)
        gt_dist = float(np.linalg.norm(anchor - gt)) if gt is not None else float("nan")
        gt_xy_dist = float(np.linalg.norm(anchor[:2] - gt[:2])) if gt is not None else float("nan")
        gt_z_abs = float(abs(anchor[2] - gt[2])) if gt is not None else float("nan")
        if gt is None:
            gt_status = "no_gt"
        elif anchor_face != gt_face:
            gt_status = "gt_face_mismatch"
        elif gt_dist <= 0.08:
            gt_status = "gt_pass"
        elif gt_dist <= 0.12 and gt_z_abs <= 0.11:
            gt_status = "gt_near_with_offset"
        else:
            gt_status = "gt_far"
        weak_status = "support_ok" if support_frac >= 0.10 else "unsupported_face"
        if bool(audit["opposed_top2"]) and anchor_face not in {str(audit["top_face"]), str(audit["second_face"])}:
            weak_status = "centroid_cancellation_unsupported"
        rows.append(
            {
                "source_variant": audit["source_variant"],
                "source_task": audit["source_task"],
                "method": method,
                "variant": variant,
                "anchor_source_side": audit["anchor_algorithm_contact_side"],
                "anchor_x": float(anchor[0]),
                "anchor_y": float(anchor[1]),
                "anchor_z": float(anchor[2]),
                "anchor_face": anchor_face,
                "anchor_face_support_frac": support_frac,
                "top_face": audit["top_face"],
                "top_face_frac": audit["top_face_frac"],
                "second_face": audit["second_face"],
                "second_face_frac": audit["second_face_frac"],
                "opposed_top2": audit["opposed_top2"],
                "selected_person_idx": audit["selected_person_idx"],
                "selected_person_num": audit["selected_person_num"],
                "partner_source_task": audit["partner_source_task"],
                "partner_person_idx": audit["partner_person_idx"],
                "partner_task_available": audit["partner_task_available"],
                "partner_n_active_points": audit["partner_n_active_points"],
                "partner_top_face": audit["partner_top_face"],
                "partner_top_face_frac": audit["partner_top_face_frac"],
                "partner_second_face": audit["partner_second_face"],
                "partner_second_face_frac": audit["partner_second_face_frac"],
                "partner_anchor_face_support_frac": partner_support_frac,
                "partner_side_status": partner_status,
                "selected_partner_top_relation": audit["selected_partner_top_relation"],
                "gt_available": gt is not None,
                "gt_anchor_x": "" if gt is None else float(gt[0]),
                "gt_anchor_y": "" if gt is None else float(gt[1]),
                "gt_anchor_z": "" if gt is None else float(gt[2]),
                "gt_face": gt_face,
                "gt_dist_m": gt_dist,
                "gt_xy_dist_m": gt_xy_dist,
                "gt_z_abs_m": gt_z_abs,
                "gt_status": gt_status,
                "weak_status": weak_status,
                "e016_diagnostic_class": audit["e016_diagnostic_class"],
                "e016_contact_preservation_5cm_pct": audit["e016_contact_preservation_5cm_pct"],
                "e016_deep_penetration_duration_pct": audit["e016_deep_penetration_duration_pct"],
            }
        )
    return rows


def _classify_e016_failure(audit: dict[str, Any], method_rows: list[dict[str, Any]]) -> dict[str, Any]:
    e016 = next(row for row in method_rows if row["method"] == "E016_centroid")
    e017 = next(row for row in method_rows if row["method"] == "E017_auto")
    diag = str(audit["e016_diagnostic_class"])
    contact = _float_or_nan(audit["e016_contact_preservation_5cm_pct"])
    gt_available = bool(e016["gt_available"])
    e016_gt_status = str(e016["gt_status"])
    e017_gt_status = str(e017["gt_status"])
    e016_weak = str(e016["weak_status"])
    e017_weak = str(e017["weak_status"])
    e016_partner_status = str(e016["partner_side_status"])
    e017_partner_status = str(e017["partner_side_status"])

    level = "non_anchor_or_unknown"
    reason = (
        "E016/E017 auto anchors are selected-person-contact based; no GT or counterpart-person evidence "
        "makes support-side anchor the primary failure source."
    )
    run_validation = False

    if gt_available and e016_gt_status == "gt_face_mismatch":
        level = "clear_anchor_error_gt_mismatch"
        reason = "E014 GT support anchor exists and E016 selected-person anchor selects a different face."
        run_validation = True
    elif gt_available and e016_gt_status in {"gt_near_with_offset", "gt_far"}:
        level = "possible_anchor_error_gt_offset"
        reason = "E014 GT face matches but E016 anchor has a nontrivial position/height offset."
        run_validation = True
    elif (
        not gt_available
        and e016_partner_status == "partner_face_mismatch"
        and diag in {"contact_preservation_gap", "artifact_failed"}
    ):
        level = "possible_anchor_error_not_partner_side"
        reason = (
            "No E014 GT; E016 is selected-person-contact based and its face disagrees with the "
            "counterpart-person dominant contact face."
        )
        run_validation = e017_partner_status == "partner_face_match"
    elif e016_weak == "centroid_cancellation_unsupported" and diag in {"contact_preservation_gap", "artifact_failed"}:
        level = "possible_anchor_error_centroid_cancellation"
        reason = "No GT, but E016 anchor lands on an unsupported face after opposed-face contact cancellation."
        run_validation = True
    elif e016_weak == "centroid_cancellation_unsupported":
        level = "weak_anchor_suspicious_not_primary"
        reason = "Anchor is unsupported, but E016 failure/contact pattern does not make anchor the primary suspect."
    elif diag == "push_or_leg_shortcut" and not math.isnan(contact) and contact >= 50.0:
        level = "likely_non_anchor_robot_artifact"
        reason = "E016 contact preservation is already high; failure is more likely leg/floor shortcut."
    elif e016_weak == "unsupported_face":
        level = "weak_anchor_suspicious"
        reason = "E016 anchor face has low contact support, but evidence is weaker than centroid cancellation or GT mismatch."

    recommended: list[str] = []
    if run_validation:
        recommended.append(str(e017["variant"]))
        if gt_available:
            recommended.append(f"E017_{_slug_from_e016_variant(str(audit['source_variant']))}_e014_seed")
    return {
        "source_variant": audit["source_variant"],
        "source_task": audit["source_task"],
        "e016_diagnostic_class": diag,
        "e016_contact_preservation_5cm_pct": audit["e016_contact_preservation_5cm_pct"],
        "e016_anchor_face": e016["anchor_face"],
        "e016_anchor_face_support_frac": e016["anchor_face_support_frac"],
        "e016_gt_status": e016_gt_status,
        "e016_partner_side_status": e016_partner_status,
        "e016_partner_anchor_face_support_frac": e016["partner_anchor_face_support_frac"],
        "e017_auto_variant": e017["variant"],
        "e017_anchor_face": e017["anchor_face"],
        "e017_anchor_face_support_frac": e017["anchor_face_support_frac"],
        "e017_gt_status": e017_gt_status,
        "e017_partner_side_status": e017_partner_status,
        "e017_partner_anchor_face_support_frac": e017["partner_anchor_face_support_frac"],
        "anchor_algorithm_contact_side": e016["anchor_source_side"],
        "partner_source_task": e016["partner_source_task"],
        "partner_top_face": e016["partner_top_face"],
        "partner_top_face_frac": e016["partner_top_face_frac"],
        "selected_partner_top_relation": e016["selected_partner_top_relation"],
        "anchor_failure_level": level,
        "anchor_failure_reason": reason,
        "run_algorithm_validation": run_validation,
        "recommended_validation_variants": ",".join(recommended),
    }


def write_method_audit_and_attribution(audits: list[dict[str, Any]], result_root: Path) -> None:
    method_rows: list[dict[str, Any]] = []
    by_variant_rows: dict[str, list[dict[str, Any]]] = {}
    for audit in audits:
        rows = _method_anchor_rows(audit)
        method_rows.extend(rows)
        by_variant_rows[str(audit["source_variant"])] = rows

    method_keys = sorted({k for row in method_rows for k in row.keys()})
    with (result_root / "anchor_method_audit.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=method_keys)
        writer.writeheader()
        writer.writerows(method_rows)

    attribution = [
        _classify_e016_failure(audit, by_variant_rows[str(audit["source_variant"])])
        for audit in audits
    ]
    attr_keys = sorted({k for row in attribution for k in row.keys()})
    with (result_root / "e016_anchor_failure_attribution.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=attr_keys)
        writer.writeheader()
        writer.writerows(attribution)

    summary = {
        "num_cases": len(audits),
        "auto_anchor_algorithm_contact_side": "selected_person_contact_mask",
        "anchor_failure_levels": {
            name: sum(row["anchor_failure_level"] == name for row in attribution)
            for name in sorted({str(row["anchor_failure_level"]) for row in attribution})
        },
        "partner_side_status_counts": {
            name: sum(
                row["method"] == "E016_centroid" and row["partner_side_status"] == name
                for row in method_rows
            )
            for name in sorted(
                {str(row["partner_side_status"]) for row in method_rows if row["method"] == "E016_centroid"}
            )
        },
        "selected_partner_top_relations": {
            name: sum(str(audit["selected_partner_top_relation"]) == name for audit in audits)
            for name in sorted({str(audit["selected_partner_top_relation"]) for audit in audits})
        },
        "validation_variants": [
            variant
            for row in attribution
            if row["run_algorithm_validation"]
            for variant in str(row["recommended_validation_variants"]).split(",")
            if variant
        ],
        "gt_cases": [
            row["source_variant"]
            for row in method_rows
            if row["method"] == "E016_centroid" and row["gt_available"]
        ],
    }
    (result_root / "anchor_method_audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )


def audit_one(
    base_row: dict[str, str],
    e016_manifest: dict[str, dict[str, str]],
    e016_eval: dict[str, dict[str, str]],
) -> dict[str, Any]:
    task_dir = BASE / base_row["source_task"]
    selected_person_idx = int(base_row["person_idx"])
    selected_person_num = _source_person_num(base_row["source_task"])
    points, times, half = _load_support_points(base_row, task_dir, person_idx=selected_person_idx)
    labels, counts, top_face, top_frac, second_face, second_frac = _face_stats(points, half)
    n = len(points)

    partner_task, partner_person_idx = _counterpart_source_task(base_row["source_task"])
    partner_task_available = bool(partner_task and (BASE / partner_task).is_dir() and partner_person_idx is not None)
    if partner_task_available:
        partner_points, _partner_times, partner_half = _load_support_points(
            base_row,
            BASE / partner_task,
            person_idx=int(partner_person_idx),
        )
        (
            partner_labels,
            partner_counts,
            partner_top_face,
            partner_top_frac,
            partner_second_face,
            partner_second_frac,
        ) = _face_stats(partner_points, partner_half) if len(partner_points) else _empty_face_stats()
    else:
        partner_points = np.zeros((0, 3), dtype=np.float64)
        (
            partner_labels,
            partner_counts,
            partner_top_face,
            partner_top_frac,
            partner_second_face,
            partner_second_frac,
        ) = _empty_face_stats()
    partner_n = len(partner_points)
    if not partner_task_available:
        selected_partner_relation = "partner_task_missing"
    elif partner_n <= 0 or not partner_top_face:
        selected_partner_relation = "partner_no_active_points"
    elif top_face == partner_top_face:
        selected_partner_relation = "same_top_face"
    elif OPPOSITE_FACE.get(top_face) == partner_top_face:
        selected_partner_relation = "opposed_top_face"
    else:
        selected_partner_relation = "different_top_face"

    e016_row = e016_manifest.get(base_row["variant"])
    if e016_row is None:
        current_anchor, current_method = e016_assets.infer_support_point(base_row, task_dir)
    else:
        current_anchor = _point_from_manifest(e016_row)
        current_method = e016_row.get("support_point_method", "")
    current_face = face_label(current_anchor, half)
    current_support_frac = counts[current_face] / max(n, 1)
    opposed_top2 = OPPOSITE_FACE.get(top_face) == second_face and second_frac >= 0.20

    manual_seed = _manual_seed_for(base_row, half)
    manual_face = face_label(manual_seed, half) if manual_seed is not None else ""

    centroid_anchor = _centroid_v2_anchor(current_anchor, half)
    face_anchor = _cluster_anchor(points, labels, half, top_face)
    selected_anchor = centroid_anchor
    selected_face = face_label(centroid_anchor, half)
    selected_policy = "centroid_z65"

    e016_diag = e016_eval.get(base_row["variant"], {}).get("E016_diagnostic_class", "")
    e016_contact = e016_eval.get(base_row["variant"], {}).get(
        "paper_omniretarget_contact_preservation_5cm_pct", ""
    )
    e016_deep = e016_eval.get(base_row["variant"], {}).get(
        "paper_omniretarget_robot_object_deep_penetration_duration_pct", ""
    )

    manual_mismatch = bool(manual_seed is not None and manual_face != current_face)
    manual_match = bool(manual_seed is not None and manual_face == current_face)
    centroid_cancellation = bool(opposed_top2 and current_face not in {top_face, second_face})
    low_support = bool(current_support_frac < 0.10 and top_frac >= 0.25)
    contact_gap = e016_diag in {"contact_preservation_gap", "artifact_failed"}

    if manual_mismatch:
        audit_class = "likely_anchor_wrong_manual_mismatch"
        selected_anchor = face_anchor
        selected_face = top_face
        selected_policy = "face_cluster"
    elif manual_match:
        audit_class = "manual_seed_face_matches_current"
    elif centroid_cancellation and contact_gap:
        audit_class = "possible_anchor_wrong_centroid_cancellation"
        selected_anchor = face_anchor
        selected_face = top_face
        selected_policy = "face_cluster"
    elif centroid_cancellation or low_support:
        audit_class = "ambiguous_low_confidence_centroid"
        selected_anchor = face_anchor
        selected_face = top_face
        selected_policy = "face_cluster"
    elif "push_or_leg_shortcut" in e016_diag:
        audit_class = "likely_non_anchor_robot_artifact"
    else:
        audit_class = "anchor_face_plausible"

    cur_min, cur_med, cur_mean = _dist_stats(current_anchor, points)
    cand_min, cand_med, cand_mean = _dist_stats(selected_anchor, points)

    def partner_frac(face: str) -> float:
        if partner_n <= 0 or face not in FACE_ORDER:
            return 0.0
        return float(partner_counts[face]) / partner_n

    current_partner_support_frac = partner_frac(current_face)
    selected_partner_support_frac = partner_frac(selected_face)
    manual_seed_partner_support_frac = partner_frac(manual_face)
    if manual_seed is None:
        e014_gt_vs_partner = ""
    elif not partner_task_available:
        e014_gt_vs_partner = "partner_task_missing"
    elif partner_n <= 0 or not partner_top_face:
        e014_gt_vs_partner = "partner_no_active_points"
    elif manual_face == partner_top_face:
        e014_gt_vs_partner = "gt_matches_partner_top_face"
    else:
        e014_gt_vs_partner = "gt_differs_from_partner_top_face"

    row = {
        "source_variant": base_row["variant"],
        "source_task": base_row["source_task"],
        "mask_slug": base_row["mask_slug"],
        "person_idx": base_row["person_idx"],
        "selected_person_idx": selected_person_idx,
        "selected_person_num": "" if selected_person_num is None else selected_person_num,
        "anchor_algorithm_contact_side": "selected_person_contact_mask",
        "partner_source_task": partner_task,
        "partner_person_idx": "" if partner_person_idx is None else partner_person_idx,
        "partner_task_available": partner_task_available,
        "n_active_points": n,
        "selected_n_active_points": n,
        "partner_n_active_points": partner_n,
        "object_half_x": float(half[0]),
        "object_half_y": float(half[1]),
        "object_half_z": float(half[2]),
        "current_anchor_x": float(current_anchor[0]),
        "current_anchor_y": float(current_anchor[1]),
        "current_anchor_z": float(current_anchor[2]),
        "current_method": current_method,
        "current_face": current_face,
        "current_face_support_frac": float(current_support_frac),
        "current_partner_face_support_frac": current_partner_support_frac,
        "top_face": top_face,
        "top_face_frac": float(top_frac),
        "selected_top_face": top_face,
        "selected_top_face_frac": float(top_frac),
        "second_face": second_face,
        "second_face_frac": float(second_frac),
        "face_count_pos_x": counts["+x"],
        "face_count_neg_x": counts["-x"],
        "face_count_pos_y": counts["+y"],
        "face_count_neg_y": counts["-y"],
        "face_entropy_bits": float(_entropy(counts, n)),
        "partner_top_face": partner_top_face,
        "partner_top_face_frac": float(partner_top_frac),
        "partner_second_face": partner_second_face,
        "partner_second_face_frac": float(partner_second_frac),
        "partner_face_count_pos_x": partner_counts["+x"],
        "partner_face_count_neg_x": partner_counts["-x"],
        "partner_face_count_pos_y": partner_counts["+y"],
        "partner_face_count_neg_y": partner_counts["-y"],
        "partner_face_entropy_bits": float(_entropy(partner_counts, partner_n)),
        "selected_partner_top_relation": selected_partner_relation,
        "opposed_top2": opposed_top2,
        "centroid_cancellation": centroid_cancellation,
        "low_support": low_support,
        "manual_seed_available": manual_seed is not None,
        "manual_seed_face": manual_face,
        "manual_seed_x": "" if manual_seed is None else float(manual_seed[0]),
        "manual_seed_y": "" if manual_seed is None else float(manual_seed[1]),
        "manual_seed_z": "" if manual_seed is None else float(manual_seed[2]),
        "manual_seed_partner_face_support_frac": manual_seed_partner_support_frac,
        "e014_gt_vs_partner_mask_status": e014_gt_vs_partner,
        "selected_policy": selected_policy,
        "selected_face": selected_face,
        "selected_partner_face_support_frac": selected_partner_support_frac,
        "selected_anchor_x": float(selected_anchor[0]),
        "selected_anchor_y": float(selected_anchor[1]),
        "selected_anchor_z": float(selected_anchor[2]),
        "current_anchor_min_palm_dist_m": cur_min,
        "current_anchor_median_palm_dist_m": cur_med,
        "current_anchor_mean_palm_dist_m": cur_mean,
        "selected_anchor_min_palm_dist_m": cand_min,
        "selected_anchor_median_palm_dist_m": cand_med,
        "selected_anchor_mean_palm_dist_m": cand_mean,
        "e016_diagnostic_class": e016_diag,
        "e016_contact_preservation_5cm_pct": e016_contact,
        "e016_deep_penetration_duration_pct": e016_deep,
        "audit_class": audit_class,
    }
    return row


def generate_support_scene(task_dir: Path, scene_name: str, point_local: np.ndarray) -> Path:
    src = task_dir / "scene.xml"
    qpos0 = np.load(task_dir / "0/trajectory_kinematic.npz")["qpos"][0]
    obj_pos = qpos0[-7:-4].astype(np.float64)
    obj_quat = qpos0[-4:].astype(np.float64)
    support_pos = obj_pos + _quat_apply(obj_quat, point_local)
    support_quat = obj_quat / np.clip(np.linalg.norm(obj_quat), 1e-8, None)

    tree = ET.parse(src)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"{src} missing worldbody")
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")

    _remove_named_children(worldbody, "body", {ANCHOR_BODY, "object_target"})
    _remove_named_children(equality, "weld", {WELD_NAME, "e016_support_weld", "e014_support_weld", "object_weld"})

    anchor = ET.Element(
        "body",
        {
            "name": ANCHOR_BODY,
            "mocap": "true",
            "pos": " ".join(f"{v:.8g}" for v in support_pos),
            "quat": " ".join(f"{v:.8g}" for v in support_quat),
        },
    )
    ET.SubElement(
        anchor,
        "geom",
        {
            "name": ANCHOR_GEOM,
            "type": "sphere",
            "size": "0.035",
            "rgba": "0.0 0.65 0.25 0.35",
            "group": "4",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    ET.SubElement(anchor, "site", {"name": "trace_support_weld_anchor", "size": "0.025"})
    worldbody.append(anchor)
    ET.SubElement(
        equality,
        "weld",
        {
            "name": WELD_NAME,
            "body1": "object",
            "body2": ANCHOR_BODY,
            "relpose": f"{point_local[0]:.8g} {point_local[1]:.8g} {point_local[2]:.8g} 1 0 0 0",
            "solref": "0.02 1",
            "solimp": "0.9 0.95 0.001",
        },
    )
    _indent(root)
    out = task_dir / f"{scene_name}.xml"
    tree.write(out, encoding="unicode", xml_declaration=False)
    text = out.read_text(encoding="utf-8")
    if not text.endswith("\n"):
        out.write_text(text + "\n", encoding="utf-8")

    model = mujoco.MjModel.from_xml_path(str(out))
    if model.nq != 43 or model.nv != 41 or model.nu != 29:
        raise ValueError(f"{out} changed dims: nq/nv/nu={model.nq}/{model.nv}/{model.nu}")
    return out


def _slug_from_e016_variant(name: str) -> str:
    return name[5:] if name.startswith("E016_") else name


def _build_manifest_row(
    base_row: dict[str, str],
    audit: dict[str, Any],
    point: np.ndarray,
    *,
    variant: str,
    role: str,
    policy: str,
    selected_face: str,
) -> dict[str, str]:
    x, y, z = _format_point(point)
    slug = variant[5:] if variant.startswith("E017_") else variant
    derived_task = f"{base_row['source_task']}_freejoint_legobj_e017"
    return {
        "variant": variant,
        "source_task": base_row["source_task"],
        "derived_task": derived_task,
        "mask_source_exp": base_row["mask_source_exp"],
        "mask_slug": base_row["mask_slug"],
        "person_idx": base_row["person_idx"],
        "queue": base_row["queue"],
        "role": role,
        "wave": "A",
        "scene_name": f"scene_e017_jointB_{slug}",
        "support_proxy_point_local_x": x,
        "support_proxy_point_local_y": y,
        "support_proxy_point_local_z": z,
        "weld_solref_timeconst": "0.02",
        "weld_solimp_1": "0.9",
        "weld_solimp_2": "0.95",
        "weld_solimp_width": "0.001",
        "support_proxy_gravity_scale": "0.5",
        "hold_contact_rew_scale": "0.0",
        "support_point_method": policy,
        "source_variant": base_row["variant"],
        "anchor_policy": policy,
        "anchor_audit_class": str(audit["audit_class"]),
        "anchor_current_face": str(audit["current_face"]),
        "anchor_selected_face": selected_face,
        "anchor_current_support_frac": f"{float(audit['current_face_support_frac']):.6g}",
        "anchor_top_face": str(audit["top_face"]),
        "anchor_top_face_frac": f"{float(audit['top_face_frac']):.6g}",
        "anchor_opposed_top2": str(bool(audit["opposed_top2"])),
    }


def build_manifest_rows(
    base_rows: list[dict[str, str]],
    audits: dict[str, dict[str, Any]],
    *,
    include_manual_seeds: bool,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for base_row in base_rows:
        audit = audits[base_row["variant"]]
        slug = _slug_from_e016_variant(base_row["variant"])
        selected = np.array(
            [
                float(audit["selected_anchor_x"]),
                float(audit["selected_anchor_y"]),
                float(audit["selected_anchor_z"]),
            ],
            dtype=np.float64,
        )
        policy = str(audit["selected_policy"])
        suffix = "face_cluster" if policy == "face_cluster" else "centroid_v2"
        role = (
            "anchor_debug"
            if (
                "wrong" in str(audit["audit_class"])
                or "ambiguous" in str(audit["audit_class"])
                or bool(audit["manual_seed_available"])
            )
            else "general"
        )
        rows.append(
            _build_manifest_row(
                base_row,
                audit,
                selected,
                variant=f"E017_{slug}_{suffix}",
                role=role,
                policy=policy,
                selected_face=str(audit["selected_face"]),
            )
        )
        manual_seed = E014_MANUAL_SEEDS.get(base_row["source_task"])
        if include_manual_seeds and manual_seed is not None:
            half = np.array(
                [float(audit["object_half_x"]), float(audit["object_half_y"]), float(audit["object_half_z"])],
                dtype=np.float64,
            )
            # Preserve the exact E014 support points. E014 intentionally allows
            # points slightly outside the collision half-size to encode a
            # partner-side support location rather than a strict surface clamp.
            seed = manual_seed.astype(np.float64).copy()
            rows.append(
                _build_manifest_row(
                    base_row,
                    audit,
                    seed,
                    variant=f"E017_{slug}_e014_seed",
                    role="manual_seed",
                    policy="e014_manual_seed",
                    selected_face=face_label(seed, half),
                )
            )
    return rows


def write_audit(audits: list[dict[str, Any]], result_root: Path) -> None:
    result_root.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in audits for k in row.keys()})
    with (result_root / "anchor_audit.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(audits)
    summary = {
        "num_cases": len(audits),
        "auto_anchor_algorithm_contact_side": "selected_person_contact_mask",
        "audit_classes": {
            name: sum(str(row["audit_class"]) == name for row in audits)
            for name in sorted({str(row["audit_class"]) for row in audits})
        },
        "selected_partner_top_relations": {
            name: sum(str(row["selected_partner_top_relation"]) == name for row in audits)
            for name in sorted({str(row["selected_partner_top_relation"]) for row in audits})
        },
        "e014_gt_vs_partner_mask_status": {
            name: sum(str(row["e014_gt_vs_partner_mask_status"]) == name for row in audits)
            for name in sorted(
                {
                    str(row["e014_gt_vs_partner_mask_status"])
                    for row in audits
                    if str(row["e014_gt_vs_partner_mask_status"])
                }
            )
        },
        "selected_policies": {
            name: sum(str(row["selected_policy"]) == name for row in audits)
            for name in sorted({str(row["selected_policy"]) for row in audits})
        },
        "likely_anchor_related": [
            row["source_variant"]
            for row in audits
            if "wrong" in str(row["audit_class"]) or "ambiguous" in str(row["audit_class"])
        ],
    }
    (result_root / "anchor_audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )


def write_manifest_and_scenes(rows: list[dict[str, str]], *, force: bool) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    prepared_tasks: set[str] = set()
    for row in rows:
        task = row["derived_task"]
        if task not in prepared_tasks:
            task_dir = e016_assets.copy_freejoint_case(row["source_task"], task, force=force)
            e016_assets.patch_leg_object_pairs(task_dir / "scene.xml", row["source_task"], task)
            prepared_tasks.add(task)
        else:
            task_dir = BASE / task
        point = _point_from_manifest(row)
        out = generate_support_scene(task_dir, row["scene_name"], point)
        print(
            f"{row['variant']}: task={task} scene={out.relative_to(REPO)} "
            f"policy={row['anchor_policy']} point={point.tolist()}"
        )

    with MANIFEST.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {MANIFEST.relative_to(REPO)}")

    summary_path = RESULTS / "anchor_method_audit_summary.json"
    if summary_path.is_file():
        validation_names = json.loads(summary_path.read_text(encoding="utf-8")).get("validation_variants", [])
        validation_set = {str(name) for name in validation_names}
        validation_rows = [dict(row) for row in rows if row["variant"] in validation_set]
        remote_idx = 0
        for row in validation_rows:
            if row["queue"] != "local":
                row["queue"] = f"remote_gpu{remote_idx % 2}"
                remote_idx += 1
        validation_manifest = RESULTS / "manifest_validation.tsv"
        with validation_manifest.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, delimiter="\t")
            writer.writeheader()
            writer.writerows(validation_rows)
        print(f"Wrote {validation_manifest.relative_to(REPO)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", type=Path, default=E016_VARIANTS)
    parser.add_argument("--e016-manifest", type=Path, default=E016_MANIFEST)
    parser.add_argument("--e016-comparison", type=Path, default=E016_COMPARISON)
    parser.add_argument("--result-root", type=Path, default=RESULTS)
    parser.add_argument("--write", action="store_true", help="also write E017 manifest and scene XML files")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-manual-seeds", action="store_true")
    args = parser.parse_args()

    base_rows = _read_tsv_rows(args.variants, FIELDNAMES)
    e016_manifest = {
        row["variant"]: row
        for row in _read_tsv_rows(args.e016_manifest)
    } if args.e016_manifest.is_file() else {}
    e016_eval = _read_csv_by_key(args.e016_comparison, "variant")

    audits = [audit_one(row, e016_manifest, e016_eval) for row in base_rows]
    write_audit(audits, args.result_root)
    write_method_audit_and_attribution(audits, args.result_root)
    print(f"Wrote {(args.result_root / 'anchor_audit.csv').relative_to(REPO)}")
    print(f"Wrote {(args.result_root / 'anchor_method_audit.csv').relative_to(REPO)}")
    print(f"Wrote {(args.result_root / 'e016_anchor_failure_attribution.csv').relative_to(REPO)}")
    print(json.dumps(json.loads((args.result_root / "anchor_audit_summary.json").read_text()), indent=2, sort_keys=True))

    if args.write:
        audit_by_variant = {str(row["source_variant"]): row for row in audits}
        manifest_rows = build_manifest_rows(
            base_rows,
            audit_by_variant,
            include_manual_seeds=not args.no_manual_seeds,
        )
        write_manifest_and_scenes(manifest_rows, force=args.force)


if __name__ == "__main__":
    main()
