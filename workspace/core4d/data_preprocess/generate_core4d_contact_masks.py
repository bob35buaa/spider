#!/usr/bin/env python3
"""Generate CORE4D raw 3cm contact proxy masks.

This script uses the same class of signal as CORE4D's official visualization:
SMPL-X vertices against object surface geometry with a distance threshold.
It is a geometric proxy, not an annotated physical-contact label.

Scope:
  - Generic across CORE4D two-person human-object sequences with files named
    person1_poses.npz, person2_poses.npz, and smooth_objposes.npy.
  - Assumes SMPL-X vertex indexing and the hand proxy vertex ranges below.
  - Output file/key names keep the E077 "3cm" convention; use threshold=0.03
    unless you also rename downstream artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


REPO = Path(__file__).resolve().parents[3]
RAW_ROOT_ENV = os.environ.get("CORE4D_REAL_ROOT")
RAW_ROOT = Path(RAW_ROOT_ENV) if RAW_ROOT_ENV else None
DEFAULT_SEQ = RAW_ROOT / "human_object_motions/20231008/045" if RAW_ROOT else None
DEFAULT_MESH = RAW_ROOT / "object_models/box/box023_m.obj" if RAW_ROOT else None
DEFAULT_OUT = REPO / "workspace/core4d/results/E077/contact_masks/box023"

# Proxy hand ranges used in E076. These are broad SMPL-X hand-region ranges,
# not an official segmentation file.
HAND_RANGES = {
    "left": np.arange(4700, 5500, dtype=np.int64),
    "right": np.arange(7500, 8150, dtype=np.int64),
}
FINGERTIP_IDS = {
    "left": np.array([5361, 4933, 5058, 5169, 5286], dtype=np.int64),
    "right": np.array([8079, 7669, 7794, 7905, 8022], dtype=np.int64),
}
PERSONS = ("person1", "person2")
HANDS = ("left", "right")


def load_person(seq_dir: Path, person: str) -> dict:
    path = seq_dir / f"{person}_poses.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing CORE4D pose file: {path}")
    return np.load(path, allow_pickle=True)["arr_0"].item()


def build_rows(
    raw_mask: np.ndarray,
    raw_min_dist: np.ndarray,
    raw_tip_min_dist: np.ndarray,
    raw_vertex_count: np.ndarray,
    trim_start: int,
    spider_frames: int,
    eval_fps: float,
    ref_fps: float,
) -> list[dict]:
    rows = []
    eval_frames = int(np.ceil(spider_frames / ref_fps * eval_fps))
    for eval_f in range(eval_frames):
        ref_f = int(round((eval_f / eval_fps) * ref_fps))
        ref_f = min(max(ref_f, 0), spider_frames - 1)
        raw_f = trim_start + ref_f
        row = {
            "eval_frame": eval_f,
            "eval_time_s": eval_f / eval_fps,
            "ref_frame_30hz": ref_f,
            "raw_frame": raw_f,
        }
        for pi, person in enumerate(PERSONS):
            for hi, hand in enumerate(HANDS):
                key = f"{person}_{hand}"
                row[f"{key}_contact"] = int(raw_mask[raw_f, pi, hi])
                row[f"{key}_min_dist_m"] = float(raw_min_dist[raw_f, pi, hi])
                row[f"{key}_tip_min_dist_m"] = float(raw_tip_min_dist[raw_f, pi, hi])
                row[f"{key}_n_vertices_lt_thresh"] = int(raw_vertex_count[raw_f, pi, hi])
        rows.append(row)
    return rows


def summarize_window(rows: list[dict], start: int, end: int) -> dict:
    window = [r for r in rows if start <= r["eval_frame"] <= end]
    summary = {
        "eval_window": [start, end],
        "raw_window": [window[0]["raw_frame"], window[-1]["raw_frame"]] if window else None,
        "n": len(window),
    }
    for person in PERSONS:
        for hand in HANDS:
            prefix = f"{person}_{hand}"
            contacts = [r[f"{prefix}_contact"] for r in window]
            dists = [r[f"{prefix}_min_dist_m"] for r in window]
            counts = [r[f"{prefix}_n_vertices_lt_thresh"] for r in window]
            summary[prefix] = {
                "contact_frames": int(sum(contacts)),
                "contact_ratio": float(np.mean(contacts)) if contacts else 0.0,
                "min_dist_cm_mean": float(np.mean(dists) * 100.0) if dists else None,
                "min_dist_cm_min": float(np.min(dists) * 100.0) if dists else None,
                "min_dist_cm_max": float(np.max(dists) * 100.0) if dists else None,
                "vertices_lt_thresh_mean": float(np.mean(counts)) if counts else None,
            }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-dir", type=Path, default=DEFAULT_SEQ, required=DEFAULT_SEQ is None)
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH, required=DEFAULT_MESH is None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--threshold", type=float, default=0.03)
    parser.add_argument("--sample-count", type=int, default=20000)
    parser.add_argument("--trim-start", type=int, default=42)
    parser.add_argument("--spider-frames", type=int, default=136)
    parser.add_argument("--ref-fps", type=float, default=30.0)
    parser.add_argument("--eval-fps", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=77)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    mesh = trimesh.load(args.mesh, process=False)
    surface_points, _ = trimesh.sample.sample_surface(
        mesh, args.sample_count, seed=rng
    )
    obj_pose_path = args.seq_dir / "smooth_objposes.npy"
    if not obj_pose_path.exists():
        raise FileNotFoundError(f"Missing CORE4D object pose file: {obj_pose_path}")
    obj_poses = np.load(obj_pose_path)
    people = {person: load_person(args.seq_dir, person) for person in PERSONS}
    n_raw = int(obj_poses.shape[0])
    trim_end = args.trim_start + args.spider_frames
    if args.trim_start < 0 or args.spider_frames <= 0 or trim_end > n_raw:
        raise ValueError(
            f"Invalid trim window [{args.trim_start}:{trim_end}] for raw frames={n_raw}"
        )

    raw_mask = np.zeros((n_raw, len(PERSONS), len(HANDS)), dtype=bool)
    raw_min_dist = np.full((n_raw, len(PERSONS), len(HANDS)), np.nan, dtype=np.float32)
    raw_tip_min_dist = np.full_like(raw_min_dist, np.nan)
    raw_vertex_count = np.zeros((n_raw, len(PERSONS), len(HANDS)), dtype=np.int32)

    for raw_f in range(n_raw):
        obj_world = surface_points @ obj_poses[raw_f, :3, :3].T + obj_poses[raw_f, :3, 3]
        tree = cKDTree(obj_world)
        for pi, person in enumerate(PERSONS):
            vertices = people[person]["vertices"][raw_f]
            for hi, hand in enumerate(HANDS):
                hand_ids = HAND_RANGES[hand]
                tip_ids = FINGERTIP_IDS[hand]
                hand_dists, _ = tree.query(vertices[hand_ids], k=1)
                tip_dists, _ = tree.query(vertices[tip_ids], k=1)
                raw_min_dist[raw_f, pi, hi] = float(hand_dists.min())
                raw_tip_min_dist[raw_f, pi, hi] = float(tip_dists.min())
                raw_vertex_count[raw_f, pi, hi] = int((hand_dists < args.threshold).sum())
                raw_mask[raw_f, pi, hi] = raw_vertex_count[raw_f, pi, hi] > 0

    spider_slice = slice(args.trim_start, trim_end)
    spider_mask = raw_mask[spider_slice]
    spider_min_dist = raw_min_dist[spider_slice]
    spider_vertex_count = raw_vertex_count[spider_slice]

    eval_frames = int(np.ceil(args.spider_frames / args.ref_fps * args.eval_fps))
    eval_ref_idx = np.array(
        [
            min(max(int(round((f / args.eval_fps) * args.ref_fps)), 0), args.spider_frames - 1)
            for f in range(eval_frames)
        ],
        dtype=np.int32,
    )
    eval_raw_idx = eval_ref_idx + args.trim_start
    eval_mask = spider_mask[eval_ref_idx]
    eval_min_dist = spider_min_dist[eval_ref_idx]
    eval_vertex_count = spider_vertex_count[eval_ref_idx]

    rows = build_rows(
        raw_mask,
        raw_min_dist,
        raw_tip_min_dist,
        raw_vertex_count,
        args.trim_start,
        args.spider_frames,
        args.eval_fps,
        args.ref_fps,
    )
    csv_path = args.out_dir / "raw_contact_mask_3cm.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    npz_path = args.out_dir / "raw_contact_mask_3cm.npz"
    np.savez(
        npz_path,
        raw_contact_mask_3cm=raw_mask,
        raw_min_dist_m=raw_min_dist,
        raw_tip_min_dist_m=raw_tip_min_dist,
        raw_vertex_count_lt_thresh=raw_vertex_count,
        spider_contact_mask_3cm=spider_mask,
        spider_min_dist_m=spider_min_dist,
        spider_vertex_count_lt_thresh=spider_vertex_count,
        eval_contact_mask_3cm=eval_mask,
        eval_min_dist_m=eval_min_dist,
        eval_vertex_count_lt_thresh=eval_vertex_count,
        eval_ref_idx=eval_ref_idx,
        eval_raw_idx=eval_raw_idx,
        persons=np.array(PERSONS),
        hands=np.array(HANDS),
        threshold_m=np.array(args.threshold),
        trim_start=np.array(args.trim_start),
        ref_fps=np.array(args.ref_fps),
        eval_fps=np.array(args.eval_fps),
        sample_count=np.array(args.sample_count),
        hand_left_range=HAND_RANGES["left"],
        hand_right_range=HAND_RANGES["right"],
        fingertip_left_ids=FINGERTIP_IDS["left"],
        fingertip_right_ids=FINGERTIP_IDS["right"],
    )

    summary = {
        "seq_dir": str(args.seq_dir),
        "mesh": str(args.mesh),
        "threshold_m": args.threshold,
        "sample_count": args.sample_count,
        "seed": args.seed,
        "raw_frames": n_raw,
        "trim_start": args.trim_start,
        "spider_frames": args.spider_frames,
        "eval_frames": eval_frames,
        "axis": {
            "person": list(PERSONS),
            "hand": list(HANDS),
        },
        "windows": [
            summarize_window(rows, 100, 114),
            summarize_window(rows, 115, 130),
            summarize_window(rows, 131, 145),
        ],
        "note": (
            "3cm mask is a CORE4D visualization-style geometric proxy, "
            "not an annotated physical-contact label."
        ),
    }
    json_path = args.out_dir / "audit_summary_3cm.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {npz_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    for window in summary["windows"]:
        print(f"window {window['eval_window']} raw={window['raw_window']}")
        if window["n"] == 0:
            print("  skipped: empty eval window")
            continue
        for person in PERSONS:
            for hand in HANDS:
                s = window[f"{person}_{hand}"]
                print(
                    f"  {person}_{hand}: "
                    f"{s['contact_frames']}/{window['n']} "
                    f"min_cm={s['min_dist_cm_mean']:.2f}/"
                    f"{s['min_dist_cm_min']:.2f}/"
                    f"{s['min_dist_cm_max']:.2f}"
                )


if __name__ == "__main__":
    main()
