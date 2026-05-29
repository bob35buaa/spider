#!/usr/bin/env python3
"""Shared helpers for E020 failure-attribution audit."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[4]
WS = REPO / "workspace/core4d_collab_retarget"
E018B = WS / "results/E018b"
E017 = WS / "results/E017"
OUT = WS / "results/E020_audit"
PER_CASE = OUT / "per_case"
MANIFEST = E018B / "manifest.tsv"
COMPARISON = E018B / "comparison.csv"

HANDS = ("left", "right")
PERSONS = ("person1", "person2")
HAND_RANGES = {
    "left": np.arange(4700, 5500, dtype=np.int64),
    "right": np.arange(7500, 8150, dtype=np.int64),
}


def _ensure_repo_path() -> None:
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))


def _read_rows(path: Path, *, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _to_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _fmt(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def case_label(variant: str) -> str:
    label = variant
    if label.startswith("E018b_"):
        label = label[len("E018b_") :]
    if label.endswith("_canonical_t02"):
        label = label[: -len("_canonical_t02")]
    return label


def case_dir(variant: str) -> Path:
    path = PER_CASE / variant
    path.mkdir(parents=True, exist_ok=True)
    return path


def all_contexts() -> list[dict[str, Any]]:
    manifest = _read_rows(MANIFEST, delimiter="\t")
    comp = {row["variant"]: row for row in _read_rows(COMPARISON)}
    e017_audit = {
        row["source_variant"]: row
        for row in _read_rows(E017 / "anchor_audit.csv")
        if row.get("source_variant")
    }
    e017_attr = {
        row["source_variant"]: row
        for row in _read_rows(E017 / "e016_anchor_failure_attribution.csv")
        if row.get("source_variant")
    }
    out: list[dict[str, Any]] = []
    for row in manifest:
        variant = row["variant"]
        out.append(
            {
                "variant": variant,
                "label": case_label(variant),
                "manifest": row,
                "comparison": comp[variant],
                "e017_audit": e017_audit.get(row["source_variant"], {}),
                "e017_attr": e017_attr.get(row["source_variant"], {}),
            }
        )
    return out


def _align_array(arr: np.ndarray, target_len: int) -> np.ndarray:
    arr = np.asarray(arr)
    if len(arr) == target_len:
        return arr
    if len(arr) == 0:
        return np.zeros((target_len,) + arr.shape[1:], dtype=arr.dtype)
    idx = np.round(np.linspace(0, len(arr) - 1, target_len)).astype(int)
    return arr[idx]


def _interp_array(arr: np.ndarray, target_len: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if len(arr) == target_len:
        return arr
    if len(arr) == 0:
        return np.zeros((target_len,) + arr.shape[1:], dtype=np.float64)
    src = np.linspace(0.0, 1.0, len(arr))
    dst = np.linspace(0.0, 1.0, target_len)
    flat = arr.reshape(len(arr), -1)
    out = np.empty((target_len, flat.shape[1]), dtype=np.float64)
    for i in range(flat.shape[1]):
        out[:, i] = np.interp(dst, src, flat[:, i])
    return out.reshape((target_len,) + arr.shape[1:])


def load_sim_qpos(variant: str) -> np.ndarray:
    data = np.load(E018B / f"{variant}.npz", allow_pickle=True)
    qpos = data["qpos"]
    return qpos.reshape(-1, qpos.shape[-1]).astype(np.float64)


def load_ref_qpos(ctx: dict[str, Any], *, target_len: int | None = None) -> np.ndarray:
    task = ctx["manifest"]["derived_task"]
    path = E018B / "scene_snapshot" / task / "0/trajectory_kinematic.npz"
    qpos = np.load(path, allow_pickle=True)["qpos"].astype(np.float64)
    if target_len is not None:
        qpos = _interp_array(qpos, target_len)
        qpos[:, -4:] /= np.clip(np.linalg.norm(qpos[:, -4:], axis=1, keepdims=True), 1e-8, None)
    return qpos


def load_ref_contact(ctx: dict[str, Any], *, target_len: int | None = None) -> np.ndarray:
    task = ctx["manifest"]["derived_task"]
    path = E018B / "scene_snapshot" / task / "0/trajectory_kinematic.npz"
    data = np.load(path, allow_pickle=True)
    contact = data["contact"][:, :2].astype(bool)
    if target_len is not None:
        contact = _align_array(contact, target_len)
    return contact


def load_mask_npz(ctx: dict[str, Any]) -> np.lib.npyio.NpzFile:
    slug = ctx["manifest"]["mask_slug"]
    return np.load(E018B / "contact_masks" / slug / "raw_contact_mask_3cm.npz", allow_pickle=True)


def load_mask_summary(ctx: dict[str, Any]) -> dict[str, Any]:
    slug = ctx["manifest"]["mask_slug"]
    path = E018B / "contact_masks" / slug / "audit_summary_3cm.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_timeseries(variant: str) -> list[dict[str, str]]:
    return _read_rows(E018B / f"timeseries_{variant}.csv")


def load_legobj_timeseries(variant: str, kind: str) -> list[dict[str, str]]:
    return [
        row
        for row in _read_rows(E018B / f"legobj_timeseries_{variant}.csv")
        if row.get("kind") == kind
    ]


def _plot_setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_simple_plot(path: Path, title: str, series: list[tuple[str, np.ndarray]], ylabel: str = "") -> None:
    plt = _plot_setup()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3.4), dpi=140)
    for name, values in series:
        values = np.asarray(values, dtype=np.float64)
        ax.plot(np.arange(len(values)), values, linewidth=1.3, label=name)
    ax.set_title(title)
    ax.set_xlabel("frame")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


_ALL_FACES = ["+x", "-x", "+y", "-y", "+z", "-z"]  # B1 修复：含 ±z


def _face_label(point: np.ndarray, center: np.ndarray, half: np.ndarray) -> str:
    """B1 修复：全 3D argmax，可返回 ±z。"""
    rel = (np.asarray(point, dtype=np.float64) - center) / np.clip(half, 1e-8, None)
    axis = int(np.argmax(np.abs(rel)))
    sign = "+" if rel[axis] >= 0.0 else "-"
    return f"{sign}{'xyz'[axis]}"


def _face_counts(points: np.ndarray, center: np.ndarray, half: np.ndarray) -> tuple[str, float, dict[str, int]]:
    """B1 修复：counts 字典覆盖 6 个面而非仅 4 个侧面。"""
    if len(points) == 0:
        return "", 0.0, {face: 0 for face in _ALL_FACES}
    labels = [_face_label(point, center, half) for point in points]
    counts = Counter(labels)
    for face in _ALL_FACES:
        counts.setdefault(face, 0)
    face, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    return face, float(count / max(len(points), 1)), dict(counts)


def _load_raw_person(seq_dir: Path, person: str) -> dict[str, Any]:
    path = seq_dir / f"{person}_poses.npz"
    return np.load(path, allow_pickle=True)["arr_0"].item()


def _raw_contact_cache(ctx: dict[str, Any], sample_count: int = 6000) -> dict[str, Any]:
    slug = ctx["manifest"]["mask_slug"]
    cache = OUT / "raw_contact_cache" / f"{slug}_sample{sample_count}.npz"
    if cache.is_file():
        data = np.load(cache, allow_pickle=True)
        return {key: data[key] for key in data.files}

    try:
        import trimesh
        from scipy.spatial import cKDTree
    except Exception as exc:  # pragma: no cover - environment guard
        raise RuntimeError("E020 raw contact centroid audit requires trimesh and scipy") from exc

    summary = load_mask_summary(ctx)
    seq_dir = Path(summary["seq_dir"])
    mesh_path = Path(summary["mesh"])
    threshold = float(summary.get("threshold_m", 0.03))
    trim_start = int(summary["trim_start"])
    spider_frames = int(summary["spider_frames"])
    raw_end = trim_start + spider_frames
    obj_poses = np.load(seq_dir / "smooth_objposes.npy")
    people = {person: _load_raw_person(seq_dir, person) for person in PERSONS}
    mesh = trimesh.load(mesh_path, process=False)
    rng = np.random.default_rng(20020)
    surface_local, _ = trimesh.sample.sample_surface(mesh, sample_count, seed=rng)

    points: dict[str, list[np.ndarray]] = {
        "person0": [],
        "person1": [],
        "person0_left": [],
        "person0_right": [],
        "person1_left": [],
        "person1_right": [],
    }
    mask_data = load_mask_npz(ctx)
    raw_mask = mask_data["raw_contact_mask_3cm"]
    for raw_f in range(trim_start, raw_end):
        obj_world = surface_local @ obj_poses[raw_f, :3, :3].T + obj_poses[raw_f, :3, 3]
        tree = cKDTree(obj_world)
        for pi, person in enumerate(PERSONS):
            vertices = people[person]["vertices"][raw_f]
            for hand in HANDS:
                hi = 0 if hand == "left" else 1
                if not bool(raw_mask[raw_f, pi, hi]):
                    continue
                ids = HAND_RANGES[hand]
                dist, idx = tree.query(vertices[ids], k=1)
                active = dist < threshold
                if not np.any(active):
                    continue
                local_hits = surface_local[idx[active]]
                points[f"person{pi}"].append(local_hits)
                points[f"person{pi}_{hand}"].append(local_hits)

    out: dict[str, np.ndarray] = {}
    for key, parts in points.items():
        arr = np.concatenate(parts, axis=0) if parts else np.zeros((0, 3), dtype=np.float64)
        if len(arr) > 20000:
            keep = rng.choice(len(arr), size=20000, replace=False)
            arr = arr[keep]
        out[key] = arr.astype(np.float32)
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    out["raw_center"] = ((bounds[0] + bounds[1]) * 0.5).astype(np.float32)
    out["raw_half"] = ((bounds[1] - bounds[0]) * 0.5).astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    return out


def run_anchor_vs_raw() -> Path:
    rows: list[dict[str, Any]] = []
    for ctx in all_contexts():
        m = ctx["manifest"]
        variant = ctx["variant"]
        raw = _raw_contact_cache(ctx)
        person_idx = int(m["person_idx"])
        partner_idx = 1 - person_idx
        selected = raw[f"person{person_idx}"]
        partner = raw[f"person{partner_idx}"]
        center = raw["raw_center"].astype(np.float64)
        raw_half = raw["raw_half"].astype(np.float64)
        mj_half = np.array(
            [_to_float(m["object_half_x"]), _to_float(m["object_half_y"]), _to_float(m["object_half_z"])],
            dtype=np.float64,
        )
        anchor = np.array(
            [
                _to_float(m["support_proxy_point_local_x"]),
                _to_float(m["support_proxy_point_local_y"]),
                _to_float(m["support_proxy_point_local_z"]),
            ],
            dtype=np.float64,
        )
        anchor_raw = center + (anchor / np.clip(mj_half, 1e-8, None)) * raw_half
        selected_centroid = selected.mean(axis=0) if len(selected) else np.full(3, np.nan)
        partner_centroid = partner.mean(axis=0) if len(partner) else np.full(3, np.nan)
        selected_dist = float(np.linalg.norm(selected_centroid - anchor_raw)) if len(selected) else float("nan")
        partner_dist = float(np.linalg.norm(partner_centroid - anchor_raw)) if len(partner) else float("nan")
        selected_face, selected_face_frac, _ = _face_counts(selected, center, raw_half)
        partner_face, partner_face_frac, _ = _face_counts(partner, center, raw_half)
        anchor_face = _face_label(anchor_raw, center, raw_half)
        pass_gate = bool(
            np.isfinite(partner_dist)
            and (
                partner_dist <= 0.08
                or (partner_face == anchor_face and partner_face_frac >= 0.45 and partner_dist <= 0.14)
            )
        )
        row = {
            "variant": variant,
            "case": ctx["label"],
            "S1_anchor_vs_raw_pass": pass_gate,
            "anchor_face": anchor_face,
            "raw_selected_top_face": selected_face,
            "raw_selected_top_face_frac": selected_face_frac,
            "raw_partner_top_face": partner_face,
            "raw_partner_top_face_frac": partner_face_frac,
            "selected_contact_points": int(len(selected)),
            "partner_contact_points": int(len(partner)),
            "selected_anchor_dist_m": selected_dist,
            "partner_anchor_dist_m": partner_dist,
            "evidence": (
                f"partner_dist={_fmt(partner_dist)}m, partner_face={partner_face}"
                f"({partner_face_frac:.0%}), anchor_face={anchor_face}"
            ),
        }
        rows.append(row)
        _plot_anchor_vs_raw(case_dir(variant) / "anchor_vs_raw.png", ctx, selected, partner, anchor_raw, center, raw_half, row)
    out = OUT / "anchor_vs_raw.csv"
    _write_rows(out, rows)
    return out


def _plot_anchor_vs_raw(
    path: Path,
    ctx: dict[str, Any],
    selected: np.ndarray,
    partner: np.ndarray,
    anchor_raw: np.ndarray,
    center: np.ndarray,
    half: np.ndarray,
    row: dict[str, Any],
) -> None:
    plt = _plot_setup()
    fig, ax = plt.subplots(figsize=(5.4, 5.0), dpi=140)
    rng = np.random.default_rng(20)
    for name, points, color in [
        ("selected raw contact", selected, "#2f6fbb"),
        ("partner raw contact", partner, "#c75b12"),
    ]:
        if len(points):
            idx = rng.choice(len(points), size=min(1200, len(points)), replace=False)
            pts = points[idx]
            ax.scatter(pts[:, 0], pts[:, 1], s=3, alpha=0.25, c=color, label=name)
    ax.scatter([anchor_raw[0]], [anchor_raw[1]], marker="x", s=90, c="#111111", label="E018b anchor")
    rect = plt.Rectangle((center[0] - half[0], center[1] - half[1]), 2 * half[0], 2 * half[1], fill=False, color="black", linewidth=1)
    ax.add_patch(rect)
    ax.set_title(f"S1 anchor vs raw: {ctx['label']}\n{row['evidence']}")
    ax.set_xlabel("object-local x (raw mesh frame, m)")
    ax.set_ylabel("object-local y (raw mesh frame, m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run_ref_physics() -> Path:
    rows: list[dict[str, Any]] = []
    for ctx in all_contexts():
        variant = ctx["variant"]
        comp = ctx["comparison"]
        m = ctx["manifest"]
        ref = load_ref_qpos(ctx)
        half_z = _to_float(m["object_half_z"])
        ref_pelvis_z_min = float(np.min(ref[:, 2]))
        ref_obj_bottom_min = float(np.min(ref[:, -5] - half_z))
        ref_leg_pct = _to_float(comp.get("full_ref_leg_box_interference_frames_pct"))
        ref_hand_pct = _to_float(comp.get("full_ref_hand_object_contact_frames_pct"))
        ref_floor_pct = _to_float(comp.get("full_ref_object_floor_contact_frames_pct"))
        pass_gate = bool(
            ref_pelvis_z_min >= 0.55
            and ref_obj_bottom_min >= -0.20
            and ref_leg_pct <= 50.0
            and ref_hand_pct >= 30.0
        )
        rows.append(
            {
                "variant": variant,
                "case": ctx["label"],
                "S2_ref_physics_pass": pass_gate,
                "ref_pelvis_z_min_m": ref_pelvis_z_min,
                "ref_object_bottom_min_m": ref_obj_bottom_min,
                "ref_hand_contact_pct": ref_hand_pct,
                "ref_leg_interference_pct": ref_leg_pct,
                "ref_object_floor_contact_pct": ref_floor_pct,
                "evidence": (
                    f"pelvis_min={_fmt(ref_pelvis_z_min)}m, "
                    f"ref_leg={_fmt(ref_leg_pct, 1)}%, ref_hand={_fmt(ref_hand_pct, 1)}%"
                ),
            }
        )
        ts = load_timeseries(variant)
        _save_simple_plot(
            case_dir(variant) / "ref_physics_timeline.png",
            f"S2 ref physics: {ctx['label']}",
            [
                ("ref pelvis z", np.array([_to_float(r["ref_pelvis_z_m"]) for r in ts])),
                ("ref object z", np.array([_to_float(r["ref_object_z_m"]) for r in ts])),
                ("ref min hand sdf", np.array([_to_float(r["ref_min_hand_sdf_m"]) for r in ts])),
            ],
            "m",
        )
    out = OUT / "ref_physics.csv"
    _write_rows(out, rows)
    return out


def run_mask_vs_raw() -> Path:
    rows: list[dict[str, Any]] = []
    for ctx in all_contexts():
        variant = ctx["variant"]
        m = ctx["manifest"]
        data = load_mask_npz(ctx)
        person_idx = int(m["person_idx"])
        raw = data["spider_contact_mask_3cm"][:, person_idx, :2].astype(bool)
        current = load_ref_contact(ctx, target_len=len(raw))
        mismatch = float(np.mean(current != raw) * 100.0)
        overclaim = float(np.mean(np.logical_and(current, ~raw)) * 100.0)
        underclaim = float(np.mean(np.logical_and(~current, raw)) * 100.0)
        raw_any_pct = float(np.mean(raw.any(axis=1)) * 100.0)
        current_any_pct = float(np.mean(current.any(axis=1)) * 100.0)
        pass_gate = mismatch <= 15.0
        rows.append(
            {
                "variant": variant,
                "case": ctx["label"],
                "S3_mask_vs_raw_pass": pass_gate,
                "raw_selected_any_contact_pct": raw_any_pct,
                "current_ref_any_contact_pct": current_any_pct,
                "mask_mismatch_pct": mismatch,
                "mask_overclaim_pct": overclaim,
                "mask_underclaim_pct": underclaim,
                "raw_left_pct": float(np.mean(raw[:, 0]) * 100.0),
                "raw_right_pct": float(np.mean(raw[:, 1]) * 100.0),
                "current_left_pct": float(np.mean(current[:, 0]) * 100.0),
                "current_right_pct": float(np.mean(current[:, 1]) * 100.0),
                "evidence": (
                    f"current_contact={_fmt(current_any_pct, 1)}%, "
                    f"raw_any={_fmt(raw_any_pct, 1)}%, mismatch={_fmt(mismatch, 1)}%"
                ),
            }
        )
        _plot_mask_timeline(case_dir(variant) / "mask_vs_raw_timeline.png", ctx, raw, current)
    out = OUT / "mask_vs_raw.csv"
    _write_rows(out, rows)
    return out


def _plot_mask_timeline(path: Path, ctx: dict[str, Any], raw: np.ndarray, current: np.ndarray) -> None:
    plt = _plot_setup()
    fig, ax = plt.subplots(figsize=(8, 2.8), dpi=140)
    series = [
        ("raw L", raw[:, 0].astype(float), 0.0),
        ("raw R", raw[:, 1].astype(float), 1.2),
        ("current L", current[:, 0].astype(float), 2.4),
        ("current R", current[:, 1].astype(float), 3.6),
    ]
    for name, values, offset in series:
        ax.step(np.arange(len(values)), values + offset, where="post", label=name)
    ax.set_title(f"S3 mask vs raw: {ctx['label']}")
    ax.set_yticks([0.5, 1.7, 2.9, 4.1], ["raw L", "raw R", "current L", "current R"])
    ax.set_xlabel("30Hz reference frame")
    ax.set_ylim(-0.2, 4.9)
    ax.grid(True, axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run_sim_ref_overlay() -> Path:
    rows: list[dict[str, Any]] = []
    for ctx in all_contexts():
        variant = ctx["variant"]
        comp = ctx["comparison"]
        contact_pct = _to_float(comp.get("paper_omniretarget_contact_preservation_5cm_pct"))
        deep_pen_pct = _to_float(comp.get("paper_omniretarget_robot_object_deep_penetration_duration_pct"))
        leg_pct = _to_float(comp.get("case_window_sim_leg_box_interference_frames_pct"))
        fall = _to_bool(comp.get("E018b_robot_fall_detected"))
        contact_ok = _to_bool(comp.get("paper_omniretarget_contact_preservation_ok"))
        deep_pen_ok = _to_bool(comp.get("paper_omniretarget_robot_object_deep_penetration_ok"))
        floor_leg_ok = _to_bool(comp.get("E018b_floor_leg_ok"))
        pass_gate = bool(
            _to_bool(comp.get("paper_dynaretarget_object_success"))
            and _to_bool(comp.get("paper_transport_success"))
            and not fall
            and contact_ok
            and deep_pen_ok
            and floor_leg_ok
        )
        rows.append(
            {
                "variant": variant,
                "case": ctx["label"],
                "S4_sim_ref_alignment_pass": pass_gate,
                "object_Epos_m": _to_float(comp.get("paper_object_Epos_case_m")),
                "object_Erot_deg": _to_float(comp.get("paper_object_Erot_case_deg")),
                "transport_success": _to_bool(comp.get("paper_transport_success")),
                "robot_fall_detected": fall,
                "contact_preservation_5cm_pct": contact_pct,
                "contact_preservation_ok": contact_ok,
                "deep_penetration_pct": deep_pen_pct,
                "deep_penetration_ok": deep_pen_ok,
                "leg_interference_pct": leg_pct,
                "floor_leg_ok": floor_leg_ok,
                "evidence": (
                    f"Epos={_fmt(_to_float(comp.get('paper_object_Epos_case_m')))}m, "
                    f"contact={_fmt(contact_pct, 1)}%(ok={contact_ok}), "
                    f"deep_pen={_fmt(deep_pen_pct, 1)}%(ok={deep_pen_ok}), fall={fall}"
                ),
            }
        )
        _plot_sim_ref_overlay(case_dir(variant) / "sim_ref_overlay.png", ctx)
        _plot_penetration_heatmap(case_dir(variant) / "penetration_heatmap.png", ctx)
        _plot_joint_err_heatmap(case_dir(variant) / "joint_err_heatmap.png", ctx)
    out = OUT / "sim_ref_overlay.csv"
    _write_rows(out, rows)
    return out


def _plot_sim_ref_overlay(path: Path, ctx: dict[str, Any]) -> None:
    ts = load_timeseries(ctx["variant"])
    _save_simple_plot(
        path,
        f"S4 sim/ref overlay: {ctx['label']}",
        [
            ("obj err", np.array([_to_float(r["obj_err_m"]) for r in ts])),
            ("sim pelvis z", np.array([_to_float(r["sim_pelvis_z_m"]) for r in ts])),
            ("ref pelvis z", np.array([_to_float(r["ref_pelvis_z_m"]) for r in ts])),
            ("sim min hand sdf", np.array([_to_float(r["sim_min_hand_sdf_m"]) for r in ts])),
            ("ref min hand sdf", np.array([_to_float(r["ref_min_hand_sdf_m"]) for r in ts])),
        ],
        "m",
    )


def _plot_penetration_heatmap(path: Path, ctx: dict[str, Any]) -> None:
    plt = _plot_setup()
    variant = ctx["variant"]
    rows = []
    labels = []
    for kind in ["sim", "ref"]:
        leg = load_legobj_timeseries(variant, kind)
        labels.extend([f"{kind} leg", f"{kind} hand"])
        rows.append(np.maximum(0.0, -np.array([_to_float(r["leg_box_sdf_min_m"]) for r in leg])))
        rows.append(np.maximum(0.0, -np.array([_to_float(r["hand_box_sdf_min_m"]) for r in leg])))
    max_len = max(len(row) for row in rows)
    data = np.vstack([_align_array(row[:, None], max_len).ravel() for row in rows])
    fig, ax = plt.subplots(figsize=(8, 2.8), dpi=140)
    im = ax.imshow(data * 100.0, aspect="auto", interpolation="nearest", cmap="magma")
    ax.set_title(f"Penetration heatmap proxy: {ctx['label']}")
    ax.set_xlabel("frame")
    ax.set_yticks(range(len(labels)), labels)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("proxy depth cm")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_joint_err_heatmap(path: Path, ctx: dict[str, Any]) -> None:
    plt = _plot_setup()
    sim = load_sim_qpos(ctx["variant"])
    ref = load_ref_qpos(ctx, target_len=len(sim))
    err = np.abs(sim[:, 7:36] - ref[:, 7:36]).T
    fig, ax = plt.subplots(figsize=(8, 4.0), dpi=140)
    im = ax.imshow(err, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0.0)
    ax.set_title(f"Joint Err heatmap: {ctx['label']}")
    ax.set_xlabel("60Hz sim frame")
    ax.set_ylabel("G1 joint index 0..28")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("|q_sim - q_ref| rad")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _csv_by_variant(path: Path) -> dict[str, dict[str, str]]:
    return {row["variant"]: row for row in _read_rows(path)}


def run_decide_root_cause() -> tuple[Path, Path]:
    required = [
        OUT / "anchor_vs_raw.csv",
        OUT / "ref_physics.csv",
        OUT / "mask_vs_raw.csv",
        OUT / "sim_ref_overlay.csv",
    ]
    if not all(path.is_file() for path in required):
        run_anchor_vs_raw()
        run_ref_physics()
        run_mask_vs_raw()
        run_sim_ref_overlay()
    s1 = _csv_by_variant(OUT / "anchor_vs_raw.csv")
    s2 = _csv_by_variant(OUT / "ref_physics.csv")
    s3 = _csv_by_variant(OUT / "mask_vs_raw.csv")
    s4 = _csv_by_variant(OUT / "sim_ref_overlay.csv")

    rows: list[dict[str, Any]] = []
    for ctx in all_contexts():
        variant = ctx["variant"]
        comp = ctx["comparison"]
        diag = comp["E018b_diagnostic_class"]
        root, owner, experiment = _decide_case(ctx, s1[variant], s2[variant], s3[variant], s4[variant])
        evidence = "; ".join(
            [
                s1[variant]["evidence"],
                s2[variant]["evidence"],
                s3[variant]["evidence"],
                s4[variant]["evidence"],
            ]
        )
        rows.append(
            {
                "variant": variant,
                "case": ctx["label"],
                "diagnostic_class_E018b": diag,
                "failure_present": diag != "paper_generalization_pass",
                "S1_anchor_vs_raw_pass": s1[variant]["S1_anchor_vs_raw_pass"],
                "S2_ref_physics_pass": s2[variant]["S2_ref_physics_pass"],
                "S3_mask_vs_raw_pass": s3[variant]["S3_mask_vs_raw_pass"],
                "S4_sim_ref_alignment_pass": s4[variant]["S4_sim_ref_alignment_pass"],
                "root_cause": root,
                "recommended_owner": owner,
                "next_experiment": experiment,
                "object_Epos_m": comp["paper_object_Epos_case_m"],
                "object_Erot_deg": comp["paper_object_Erot_case_deg"],
                "contact_preservation_5cm_pct": comp["paper_omniretarget_contact_preservation_5cm_pct"],
                "deep_penetration_pct": comp["paper_omniretarget_robot_object_deep_penetration_duration_pct"],
                "robot_fall_detected": comp["E018b_robot_fall_detected"],
                "mask_overclaim_pct": s3[variant]["mask_overclaim_pct"],
                "partner_anchor_dist_m": s1[variant]["partner_anchor_dist_m"],
                "ref_leg_interference_pct": s2[variant]["ref_leg_interference_pct"],
                "evidence": evidence,
            }
        )
    root_csv = OUT / "root_cause_attribution.csv"
    _write_rows(root_csv, rows)
    summary = _write_summary_md(rows)
    return root_csv, summary


def _decide_case(
    ctx: dict[str, Any],
    s1: dict[str, str],
    s2: dict[str, str],
    s3: dict[str, str],
    s4: dict[str, str],
) -> tuple[str, str, str]:
    diag = ctx["comparison"]["E018b_diagnostic_class"]
    label = ctx["label"]
    gt_ok = _to_bool(ctx["comparison"].get("E018b_gt_gate_pass"))
    ref_leg = _to_float(s2["ref_leg_interference_pct"])
    overclaim = _to_float(s3["mask_overclaim_pct"])
    partner_dist = _to_float(s1["partner_anchor_dist_m"])
    partner_face_frac = _to_float(s1["raw_partner_top_face_frac"])

    if diag == "paper_generalization_pass":
        return "pass", "none", "none"
    if diag == "robot_fall_visual_fail":
        return "algo_stability", "SPIDER physics/control", "E022_stability_leg_collision"
    if diag == "contact_preservation_gap" and gt_ok:
        return "algo_contact", "SPIDER contact reward/control", "E023_robot_side_contact_closure"
    if diag == "contact_preservation_gap" and label == "box023_p1":
        return "contact_mask", "OmniRetarget/contact-mask pipeline", "E021_per_eef_mask_repair"
    if diag == "contact_preservation_gap" and label.startswith("desk"):
        return "raw_data", "OmniRetarget data triage", "E024_multi_agent_or_data_filter"
    if diag == "contact_preservation_gap" and ref_leg > 50.0:
        return "retarget_kinematic", "OmniRetarget kinematic pipeline", "E021_ref_geometry_repair"
    if diag == "contact_preservation_gap" and overclaim >= 50.0:
        return "contact_mask", "OmniRetarget/contact-mask pipeline", "E021_per_eef_mask_repair"
    if diag == "contact_preservation_gap" and partner_dist > 0.14 and partner_face_frac >= 0.45:
        return "raw_data", "OmniRetarget data triage", "E024_multi_agent_or_data_filter"
    if diag in {"artifact_failed", "push_or_leg_shortcut"}:
        return "algo_contact", "SPIDER collision/contact control", "E023_robot_side_contact_closure"
    if not _to_bool(ctx["comparison"].get("paper_dynaretarget_object_success")):
        return "algo_tracking", "SPIDER object tracking", "E025_tracking_regression"
    return "algo_contact", "SPIDER contact reward/control", "E023_robot_side_contact_closure"


def _write_summary_md(rows: list[dict[str, Any]]) -> Path:
    counts = Counter(str(row["root_cause"]) for row in rows)
    lines = [
        "# E020 Attribution Summary",
        "",
        "## Root-Cause Counts",
        "",
        "| root_cause | count |",
        "|---|---:|",
    ]
    for key, count in sorted(counts.items()):
        lines.append(f"| `{key}` | {count} |")
    lines.extend(
        [
            "",
            "## Per-Case Attribution",
            "",
            "| Case | E018b diag | S1 | S2 | S3 | S4 | root_cause | next |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| `{row['case']}` | `{row['diagnostic_class_E018b']}` | "
            f"{row['S1_anchor_vs_raw_pass']} | {row['S2_ref_physics_pass']} | "
            f"{row['S3_mask_vs_raw_pass']} | {row['S4_sim_ref_alignment_pass']} | "
            f"`{row['root_cause']}` | `{row['next_experiment']}` |"
        )
    lines.extend(
        [
            "",
            "## Actionable Next Experiments",
            "",
            "- `E021_per_eef_mask_repair` / `E021_ref_geometry_repair`: repair current all-on `trajectory_kinematic.contact` and high-leg-interference kinematic refs before another 13-case sweep.",
            "- `E022_stability_leg_collision`: add robot-side fall, pelvis, and leg/object collision gates to the CEM objective for `box021*` and `bucket001*` fall cases.",
            "- `E023_robot_side_contact_closure`: target cases with good object tracking but bad contact/artifact metrics using per-hand contact shaping and collision penalties.",
            "- `E024_multi_agent_or_data_filter`: separate cases where raw partner support dominates or single-G1 physics is under-specified (`desk021`, partner-heavy buckets).",
            "",
            "## Protocol Notes",
            "",
            "S1 recomputes raw SMPL-X hand/object contact centroids from CORE4D raw files and compares them to the E018b canonical support anchor after axis-wise mesh-frame scaling.",
            "S2 uses the saved kinematic reference and E018b ref geometry metrics; S3 compares the processed SPIDER contact field with raw 3cm per-EEF masks; S4 overlays saved sim/ref rollout metrics.",
        ]
    )
    path = OUT / "attribution_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_keyframes() -> Path:
    try:
        import cv2
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("E020 keyframe rendering requires opencv-python and Pillow") from exc

    index_rows = []
    root_rows = _csv_by_variant(OUT / "root_cause_attribution.csv") if (OUT / "root_cause_attribution.csv").is_file() else {}
    for ctx in all_contexts():
        variant = ctx["variant"]
        video = REPO / ctx["manifest"]["online_video_path"]
        out = case_dir(variant) / "keyframe_triplet.jpg"
        if not video.is_file():
            sheet = E018B / "online_video" / f"{variant}_sheet.jpg"
            if sheet.is_file():
                Image.open(sheet).save(out)
            index_rows.append({"variant": variant, "keyframe_triplet": str(out.relative_to(REPO)), "source": "sheet_fallback"})
            continue
        cap = cv2.VideoCapture(str(video))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        frames = []
        for frac in [0.25, 0.50, 0.75]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(total - 1, int(total * frac)))
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame).resize((480, 160)))
        cap.release()
        if not frames:
            continue
        canvas = Image.new("RGB", (480 * len(frames), 200), "white")
        for i, frame in enumerate(frames):
            canvas.paste(frame, (480 * i, 40))
        draw = ImageDraw.Draw(canvas)
        text = f"{ctx['label']} | {root_rows.get(variant, {}).get('root_cause', ctx['comparison']['E018b_diagnostic_class'])}"
        draw.text((10, 10), text, fill=(0, 0, 0), font=ImageFont.load_default())
        canvas.save(out, quality=92)
        index_rows.append({"variant": variant, "keyframe_triplet": str(out.relative_to(REPO)), "source": str(video.relative_to(REPO))})
    path = OUT / "keyframe_index.csv"
    _write_rows(path, index_rows)
    return path


def build_attribution_panels() -> Path:
    from PIL import Image, ImageDraw, ImageFont

    root_rows = _csv_by_variant(OUT / "root_cause_attribution.csv") if (OUT / "root_cause_attribution.csv").is_file() else {}
    panel_rows = []
    for ctx in all_contexts():
        variant = ctx["variant"]
        cdir = case_dir(variant)
        images = [
            cdir / "anchor_vs_raw.png",
            cdir / "ref_physics_timeline.png",
            cdir / "mask_vs_raw_timeline.png",
            cdir / "sim_ref_overlay.png",
            cdir / "penetration_heatmap.png",
            cdir / "joint_err_heatmap.png",
        ]
        thumbs = []
        for path in images:
            img = Image.open(path).convert("RGB")
            img.thumbnail((520, 300))
            tile = Image.new("RGB", (520, 300), "white")
            tile.paste(img, ((520 - img.width) // 2, (300 - img.height) // 2))
            thumbs.append(tile)
        panel = Image.new("RGB", (1040, 960), "white")
        draw = ImageDraw.Draw(panel)
        root = root_rows.get(variant, {}).get("root_cause", "pending")
        draw.text((16, 12), f"E020 attribution panel: {ctx['label']} | root_cause={root}", fill=(0, 0, 0), font=ImageFont.load_default())
        for i, tile in enumerate(thumbs):
            x = (i % 2) * 520
            y = 60 + (i // 2) * 300
            panel.paste(tile, (x, y))
        out = cdir / "attribution_panel.png"
        panel.save(out)
        panel_rows.append({"variant": variant, "attribution_panel": str(out.relative_to(REPO))})
    path = OUT / "panel_index.csv"
    _write_rows(path, panel_rows)
    return path
