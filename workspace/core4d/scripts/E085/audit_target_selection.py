#!/usr/bin/env python3
"""Audit E085 raw target choices: broad-hand centroid vs fingertip alternatives."""

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
import numpy as np
import trimesh
from scipy.spatial import cKDTree


REPO = Path(__file__).resolve().parents[4]
SCRIPT_DIR = REPO / "workspace/core4d/scripts/E085"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_raw_contact_targets import (  # noqa: E402
    DEFAULT_CASES,
    FACE_NAMES,
    FINGERTIP_IDS,
    HAND_RANGES,
    HANDS,
    PERSONS,
    face_label,
    load_person,
    load_scene_geometry,
    project_to_box_surface,
    quat_wxyz_to_mat,
    read_tsv,
    vertical_fraction,
)


DEFAULT_OUT = REPO / "workspace/core4d/results/E085/target_selection_audit"


def jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(type(obj).__name__)


def candidate_stats(rows: list[dict[str, object]], hand: str, candidate: str) -> dict:
    vals = [
        row
        for row in rows
        if row["hand"] == hand
        and row["candidate"] == candidate
        and np.isfinite(float(row["vertical_frac"]))
    ]
    if not vals:
        return {"n": 0}
    vf = np.asarray([float(row["vertical_frac"]) for row in vals], dtype=np.float64)
    faces = [str(row["face"]) for row in vals]
    return {
        "n": len(vals),
        "vertical_frac_mean_median_min_max": [
            float(vf.mean()),
            float(np.median(vf)),
            float(vf.min()),
            float(vf.max()),
        ],
        "face_counts": dict(Counter(faces)),
    }


def add_row(
    rows: list[dict[str, object]],
    raw_frame: int,
    hand: str,
    candidate: str,
    local: np.ndarray,
    body_obj_pos: np.ndarray,
    body_obj_mat: np.ndarray,
    half: np.ndarray,
    meta: dict[str, object],
) -> None:
    if not np.isfinite(local).all():
        return
    rows.append(
        {
            "raw_frame": raw_frame,
            "hand": hand,
            "candidate": candidate,
            "local_x": float(local[0]),
            "local_y": float(local[1]),
            "local_z": float(local[2]),
            "face": face_label(local, half),
            "vertical_frac": vertical_fraction(local, body_obj_pos, body_obj_mat, half),
            **meta,
        }
    )


def summarize_deltas(rows: list[dict[str, object]], hand: str) -> dict[str, object]:
    by_frame: dict[int, dict[str, np.ndarray]] = defaultdict(dict)
    for row in rows:
        if row["hand"] != hand:
            continue
        frame = int(row["raw_frame"])
        by_frame[frame][str(row["candidate"])] = np.asarray(
            [row["local_x"], row["local_y"], row["local_z"]], dtype=np.float64
        )
    out: dict[str, object] = {}
    for cand in ["tip_best_projected", "tip_mean_projected", "high_close_projected"]:
        deltas = []
        for data in by_frame.values():
            if "broad_projected" in data and cand in data:
                deltas.append(np.linalg.norm(data[cand] - data["broad_projected"]))
        if deltas:
            arr = np.asarray(deltas)
            out[f"{cand}_vs_broad_projected_cm_mean_min_max"] = [
                float(arr.mean() * 100.0),
                float(arr.min() * 100.0),
                float(arr.max() * 100.0),
            ]
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_vfrac(path: Path, rows: list[dict[str, object]], variant: str) -> None:
    candidates = [
        "broad_visual",
        "broad_projected",
        "tip_best_projected",
        "tip_mean_projected",
        "high_close_projected",
    ]
    colors = {
        "broad_visual": "#4c78a8",
        "broad_projected": "#1f77b4",
        "tip_best_projected": "#f58518",
        "tip_mean_projected": "#e45756",
        "high_close_projected": "#54a24b",
    }
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, sharey=True)
    for ax, hand in zip(axes, HANDS):
        for cand in candidates:
            vals = [
                row
                for row in rows
                if row["hand"] == hand and row["candidate"] == cand
            ]
            vals.sort(key=lambda r: int(r["raw_frame"]))
            if not vals:
                continue
            ax.plot(
                [int(r["raw_frame"]) for r in vals],
                [float(r["vertical_frac"]) for r in vals],
                label=cand,
                color=colors[cand],
                linewidth=1.5,
            )
        ax.axhline(0.3, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(f"{variant} {hand}")
        ax.set_ylabel("vertical fraction")
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("raw frame")
    axes[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_scatter(path: Path, rows: list[dict[str, object]], variant: str, hand: str, half: np.ndarray) -> None:
    candidates = [
        "broad_projected",
        "tip_best_projected",
        "tip_mean_projected",
        "high_close_projected",
    ]
    colors = {
        "broad_projected": "#1f77b4",
        "tip_best_projected": "#f58518",
        "tip_mean_projected": "#e45756",
        "high_close_projected": "#54a24b",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, dims, labels in [
        (axes[0], (1, 2), ("local y", "local z")),
        (axes[1], (0, 2), ("local x", "local z")),
    ]:
        for cand in candidates:
            vals = [
                row
                for row in rows
                if row["hand"] == hand and row["candidate"] == cand
            ]
            if not vals:
                continue
            xs = [float(row[f"local_{'xyz'[dims[0]]}"]) for row in vals]
            ys = [float(row[f"local_{'xyz'[dims[1]]}"]) for row in vals]
            ax.scatter(xs, ys, s=14, alpha=0.7, label=cand, color=colors[cand])
        ax.add_patch(
            plt.Rectangle(
                (-half[dims[0]], -half[dims[1]]),
                2 * half[dims[0]],
                2 * half[dims[1]],
                fill=False,
                color="black",
                linewidth=1.2,
            )
        )
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(f"{variant} {hand} target candidates")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def audit_case(row: dict[str, str], out_root: Path, threshold: float, sample_count: int, seed: int) -> dict:
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
    mesh_rot = quat_wxyz_to_mat(geom["visual_quat"])

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
    rows: list[dict[str, object]] = []

    for raw_f in range(raw_mask.shape[0]):
        R_raw = obj_poses[raw_f, :3, :3]
        t_raw = obj_poses[raw_f, :3, 3]
        body_obj_pos = visual_pos @ R_raw.T + t_raw
        body_obj_mat = R_raw @ mesh_rot
        obj_world = surface_points @ R_raw.T + t_raw
        tree = cKDTree(obj_world)
        vertices = person["vertices"][raw_f]

        for hi, hand in enumerate(HANDS):
            if not raw_mask[raw_f, hi]:
                continue
            hand_verts = vertices[HAND_RANGES[hand]]
            dists, nn_idx = tree.query(hand_verts, k=1)
            close = dists < threshold
            tip_verts = vertices[FINGERTIP_IDS[hand]]
            tip_dists, tip_nn_idx = tree.query(tip_verts, k=1)
            close_tip = tip_dists < threshold
            meta = {
                "min_dist_m": float(raw_min_dist[raw_f, hi]),
                "tip_min_dist_m": float(raw_tip_min_dist[raw_f, hi]),
                "mask_vertices_lt_thresh": int(raw_vertex_count[raw_f, hi]),
                "tip_vertices_lt_thresh": int(close_tip.sum()),
            }

            if close.any():
                close_body = body_surface_points[nn_idx[close]]
                broad_visual = close_body.mean(axis=0)
                broad_projected = project_to_box_surface(broad_visual, half)
                add_row(rows, raw_f, hand, "broad_visual", broad_visual, body_obj_pos, body_obj_mat, half, meta)
                add_row(rows, raw_f, hand, "broad_projected", broad_projected, body_obj_pos, body_obj_mat, half, meta)
                close_vfrac = np.asarray(
                    [vertical_fraction(p, body_obj_pos, body_obj_mat, half) for p in close_body],
                    dtype=np.float64,
                )
                cutoff = np.quantile(close_vfrac, 0.75)
                high_body = close_body[close_vfrac >= cutoff]
                high_visual = high_body.mean(axis=0)
                high_projected = project_to_box_surface(high_visual, half)
                add_row(rows, raw_f, hand, "high_close_visual", high_visual, body_obj_pos, body_obj_mat, half, meta)
                add_row(rows, raw_f, hand, "high_close_projected", high_projected, body_obj_pos, body_obj_mat, half, meta)

            tip_body = body_surface_points[tip_nn_idx]
            best = int(np.argmin(tip_dists))
            tip_best_visual = tip_body[best]
            tip_best_projected = project_to_box_surface(tip_best_visual, half)
            tip_mean_visual = tip_body.mean(axis=0)
            tip_mean_projected = project_to_box_surface(tip_mean_visual, half)
            add_row(rows, raw_f, hand, "tip_best_visual", tip_best_visual, body_obj_pos, body_obj_mat, half, meta)
            add_row(rows, raw_f, hand, "tip_best_projected", tip_best_projected, body_obj_pos, body_obj_mat, half, meta)
            add_row(rows, raw_f, hand, "tip_mean_visual", tip_mean_visual, body_obj_pos, body_obj_mat, half, meta)
            add_row(rows, raw_f, hand, "tip_mean_projected", tip_mean_projected, body_obj_pos, body_obj_mat, half, meta)
            if close_tip.any():
                tip_close_visual = tip_body[close_tip].mean(axis=0)
                tip_close_projected = project_to_box_surface(tip_close_visual, half)
                add_row(rows, raw_f, hand, "tip_close_visual", tip_close_visual, body_obj_pos, body_obj_mat, half, meta)
                add_row(rows, raw_f, hand, "tip_close_projected", tip_close_projected, body_obj_pos, body_obj_mat, half, meta)

    out_dir = out_root / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "target_selection_candidates.csv"
    write_csv(csv_path, rows)
    plot_vfrac(out_dir / "vertical_fraction_candidates.png", rows, variant)
    for hand in HANDS:
        plot_scatter(out_dir / f"{hand}_candidate_scatter.png", rows, variant, hand, half)

    candidates = sorted({str(row["candidate"]) for row in rows})
    summary = {
        "variant": variant,
        "task": task,
        "person_idx": person_idx,
        "threshold_m": threshold,
        "sample_count": sample_count,
        "hands": {
            hand: {
                cand: candidate_stats(rows, hand, cand)
                for cand in candidates
            }
            for hand in HANDS
        },
        "deltas": {hand: summarize_deltas(rows, hand) for hand in HANDS},
        "outputs": {
            "csv": str(csv_path.relative_to(REPO)),
            "vertical_fraction_plot": str((out_dir / "vertical_fraction_candidates.png").relative_to(REPO)),
            "left_scatter": str((out_dir / "left_candidate_scatter.png").relative_to(REPO)),
            "right_scatter": str((out_dir / "right_candidate_scatter.png").relative_to(REPO)),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=jsonable), encoding="utf-8"
    )
    return summary


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
        audit_case(row, args.out_root, args.threshold, args.sample_count, args.seed)
        for row in read_tsv(args.cases)
    ]
    aggregate = {"num_cases": len(summaries), "cases": summaries}
    path = args.out_root / "aggregate_summary.json"
    path.write_text(json.dumps(aggregate, indent=2, default=jsonable), encoding="utf-8")
    print(json.dumps(aggregate, indent=2, default=jsonable))


if __name__ == "__main__":
    main()
