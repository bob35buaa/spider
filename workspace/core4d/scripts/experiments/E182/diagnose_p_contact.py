#!/usr/bin/env python3
"""Render task-pose P phantom-contact diagnostics for failed E182 screen groups."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from build_task_query_fixture import DEFAULT_OUTPUT_ROOT
from e182_common import atomic_json, relative_to_repo, repo_path, sha256_file
from evaluate_task_queries import (
    DEFAULT_FIXTURE,
    _load_candidate_and_oracle,
    _load_frozen_fixture,
    _raycasting_scene,
    _resolve_case_and_candidate,
    _scene_signed_distance,
    _tape_chunk_path,
    materialize_selected_query_points,
)
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

SCREEN_ROOT = DEFAULT_OUTPUT_ROOT / "screen"
OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT / "p_contact_diagnostic"
OBJECT_KEY = "bucket003"
CASE_ID = "bucket003_20231018_001_p1"
ORACLE_BLUE = "#1f77b4"
CANDIDATE_ORANGE = "#ff6b00"
PHANTOM_MAGENTA = "#d000d0"
MISSED_CYAN = "#00a6c7"
BOTH_GOLD = "#d88c00"
NEITHER_GRAY = "#9a9a9a"


def _screen_rows(path: Path) -> list[dict[str, str]]:
    """Read only the frozen formal screen table."""
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def select_failed_diagnostic_candidates(
    table_path: Path,
) -> list[dict[str, str]]:
    """Pick the minimum frozen score in each failed bucket003 K group."""
    rows = [row for row in _screen_rows(table_path) if row["object_key"] == OBJECT_KEY]
    selected = []
    for max_hulls in (8, 16, 32):
        group = [row for row in rows if int(row["max_hulls"]) == max_hulls]
        if len(group) != 6 or any(
            row["launch_floor_status"] != "FAIL" for row in group
        ):
            raise RuntimeError(f"bucket003/K{max_hulls} is not a complete failed group")
        selected.append(
            min(
                group,
                key=lambda row: (
                    float(row["score_worst"]),
                    int(row["actual_hulls"]),
                    float(row["wall_seconds"]),
                    row["candidate_id"],
                ),
            )
        )
    return selected


def _load_parts(candidate: dict[str, Any]) -> list[trimesh.Trimesh]:
    """Load ordered convex parts after the evaluator has verified their SHA set."""
    return [
        trimesh.load(
            repo_path(part["path"]),
            force="mesh",
            process=False,
            maintain_order=True,
        )
        for part in candidate["parts"]
    ]


def _add_parts(axis: Any, parts: list[trimesh.Trimesh], alpha: float) -> None:
    """Add categorical ordered hull parts to one 3D axis."""
    palette = matplotlib.colormaps["tab20"]
    for index, part in enumerate(parts):
        color = list(palette(index % 20))
        color[3] = alpha
        axis.add_collection3d(
            Poly3DCollection(
                np.asarray(part.triangles),
                facecolors=[color],
                edgecolors=[(0.40, 0.12, 0.02, min(0.8, alpha + 0.25))],
                linewidths=0.18,
            )
        )


def _point_classes(
    candidate_clearance: np.ndarray, oracle_clearance: np.ndarray
) -> dict[str, np.ndarray]:
    """Partition one pose's P points into exact contact-confusion classes."""
    candidate = candidate_clearance <= 0.0
    oracle = oracle_clearance <= 0.0
    return {
        "both": candidate & oracle,
        "phantom": candidate & ~oracle,
        "missed": ~candidate & oracle,
        "neither": ~candidate & ~oracle,
    }


def _set_equal_3d_bounds(axis: Any, points: np.ndarray) -> None:
    """Use one orthographic object-local cube around all displayed evidence."""
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    center = 0.5 * (lower + upper)
    radius = max(float((upper - lower).max()) * 0.56, 0.05)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))
    axis.set_proj_type("ortho")
    axis.set_axis_off()


def _render_3d(
    *,
    candidate_id: str,
    max_hulls: int,
    parts: list[trimesh.Trimesh],
    oracle: trimesh.Trimesh,
    points: np.ndarray,
    classes: dict[str, np.ndarray],
    pose_label: str,
    output: Path,
) -> None:
    """Render four E178-style object-local task-pose views."""
    oracle_points, _ = trimesh.sample.sample_surface(oracle, 12_000, seed=182)
    all_points = np.concatenate(
        [oracle_points, np.concatenate([part.vertices for part in parts]), points],
        axis=0,
    )
    views = ((-55, 23), (35, 20), (125, 18), (-145, 30))
    figure = plt.figure(figsize=(11.2, 9.2), dpi=175)
    for index, (azimuth, elevation) in enumerate(views, start=1):
        axis = figure.add_subplot(2, 2, index, projection="3d")
        axis.scatter(
            oracle_points[:, 0],
            oracle_points[:, 1],
            oracle_points[:, 2],
            s=0.18,
            color=ORACLE_BLUE,
            alpha=0.20,
            depthshade=False,
            rasterized=True,
        )
        _add_parts(axis, parts, 0.10)
        for name, color, size, alpha in (
            ("neither", NEITHER_GRAY, 2.0, 0.12),
            ("both", BOTH_GOLD, 8.0, 0.82),
            ("missed", MISSED_CYAN, 12.0, 0.92),
            ("phantom", PHANTOM_MAGENTA, 14.0, 0.95),
        ):
            mask = classes[name]
            if mask.any():
                axis.scatter(
                    points[mask, 0],
                    points[mask, 1],
                    points[mask, 2],
                    s=size,
                    color=color,
                    alpha=alpha,
                    depthshade=False,
                )
        _set_equal_3d_bounds(axis, all_points)
        axis.view_init(elev=elevation, azim=azimuth)
        axis.set_title(f"az={azimuth}°, el={elevation}°", fontsize=9)
    figure.suptitle(
        f"E182 bucket003 · {candidate_id} · K{max_hulls} task-pose P contact\n"
        f"{pose_label} · magenta=CoACD-only phantom contact",
        fontsize=13,
    )
    figure.legend(
        handles=(
            Patch(color=ORACLE_BLUE, label="original mesh M (surface)"),
            Patch(color=CANDIDATE_ORANGE, label="CoACD hull union C"),
            Patch(color=PHANTOM_MAGENTA, label="C contact / M free (phantom)"),
            Patch(color=MISSED_CYAN, label="M contact / C free (missed)"),
            Patch(color=BOTH_GOLD, label="both contact"),
        ),
        loc="lower center",
        ncol=3,
        fontsize=8,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, facecolor="white")
    plt.close(figure)


def _render_2d(
    *,
    candidate_id: str,
    max_hulls: int,
    union: trimesh.Trimesh,
    oracle: trimesh.Trimesh,
    points: np.ndarray,
    classes: dict[str, np.ndarray],
    timelines: list[dict[str, Any]],
    pose_label: str,
    output: Path,
) -> None:
    """Render three task-point projections plus contact timeline."""
    oracle_points, _ = trimesh.sample.sample_surface(oracle, 18_000, seed=182)
    union_points, _ = trimesh.sample.sample_surface(union, 18_000, seed=183)
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 9.2), dpi=175)
    projections = ((0, 1, "XY / top"), (0, 2, "XZ / side"), (1, 2, "YZ / end"))
    for axis, (horizontal, vertical, title) in zip(
        axes.flat[:3], projections, strict=True
    ):
        axis.scatter(
            oracle_points[:, horizontal],
            oracle_points[:, vertical],
            s=0.16,
            color=ORACLE_BLUE,
            alpha=0.26,
            rasterized=True,
        )
        axis.scatter(
            union_points[:, horizontal],
            union_points[:, vertical],
            s=0.12,
            color=CANDIDATE_ORANGE,
            alpha=0.18,
            rasterized=True,
        )
        for name, color, size, alpha in (
            ("neither", NEITHER_GRAY, 2.0, 0.12),
            ("both", BOTH_GOLD, 10.0, 0.85),
            ("missed", MISSED_CYAN, 14.0, 0.92),
            ("phantom", PHANTOM_MAGENTA, 16.0, 0.96),
        ):
            mask = classes[name]
            if mask.any():
                axis.scatter(
                    points[mask, horizontal],
                    points[mask, vertical],
                    s=size,
                    color=color,
                    alpha=alpha,
                )
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(title)
        axis.grid(alpha=0.16)
        axis.set_xlabel("xyz"[horizontal] + " (m)")
        axis.set_ylabel("xyz"[vertical] + " (m)")

    timeline_axis = axes.flat[3]
    offset = 0
    for family_index, timeline in enumerate(timelines):
        count = len(timeline["candidate_contact"])
        x = np.arange(count) + offset
        timeline_axis.step(
            x,
            np.asarray(timeline["oracle_contact"], dtype=int) + 0.05,
            where="mid",
            color=ORACLE_BLUE,
            linewidth=1.2,
            label="M contact" if family_index == 0 else None,
        )
        timeline_axis.step(
            x,
            np.asarray(timeline["candidate_contact"], dtype=int) - 0.05,
            where="mid",
            color=CANDIDATE_ORANGE,
            linewidth=1.0,
            label="C contact" if family_index == 0 else None,
        )
        phantom = np.asarray(timeline["candidate_contact"], dtype=bool) & ~np.asarray(
            timeline["oracle_contact"], dtype=bool
        )
        timeline_axis.scatter(
            x[phantom],
            np.full(int(phantom.sum()), 1.18),
            color=PHANTOM_MAGENTA,
            s=10,
            label="phantom pose" if family_index == 0 else None,
        )
        timeline_axis.axvline(offset - 0.5, color="#cccccc", linewidth=0.6)
        timeline_axis.text(
            offset + max(count - 1, 0) / 2,
            -0.18,
            timeline["source_family"],
            ha="center",
            va="top",
            fontsize=8,
        )
        offset += count
    timeline_axis.set_ylim(-0.3, 1.3)
    timeline_axis.set_yticks((0, 1), ("free", "contact"))
    timeline_axis.set_title("Pose-level P contact timeline")
    timeline_axis.set_xlabel("concatenated static pose index")
    timeline_axis.grid(alpha=0.16)
    timeline_axis.legend(fontsize=8, loc="lower right")
    figure.suptitle(
        f"E182 bucket003 · {candidate_id} · K{max_hulls}\n{pose_label}", fontsize=13
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, facecolor="white")
    plt.close(figure)


def diagnose_candidate(
    row: dict[str, str],
    *,
    fixture_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Recompute static P confusion, select one phantom pose, and render evidence."""
    fixture, fixture_sha = _load_frozen_fixture(fixture_path)
    case, fixture_candidate = _resolve_case_and_candidate(
        fixture, CASE_ID, row["candidate_id"]
    )
    candidate, union, oracle = _load_candidate_and_oracle(fixture_candidate)
    parts = _load_parts(candidate)
    union_scene = _raycasting_scene(union)
    oracle_scene = _raycasting_scene(oracle)
    prg_manifest_path = repo_path(case["prg_manifest"]["path"])
    prg = json.loads(prg_manifest_path.read_text(encoding="utf-8"))
    static_entries = [
        entry for entry in prg["chunks"] if entry["source_family"] != "cem_on_a"
    ]
    timelines = []
    representatives = []
    aggregate = {
        "pose_count": 0,
        "true_positive_count": 0,
        "true_negative_count": 0,
        "phantom_contact_count": 0,
        "missed_contact_count": 0,
    }
    for entry in static_entries:
        chunk_path = _tape_chunk_path(entry)
        with np.load(chunk_path, allow_pickle=False) as chunk:
            full = np.arange(len(chunk["point_geom_id"]), dtype=np.int32)
            points = materialize_selected_query_points(chunk, full)
            radii = np.asarray(chunk["point_radius_m"], dtype=np.float64)
            p_mask = np.isin(
                np.asarray(chunk["point_geom_id"], dtype=np.int32),
                prg["consumer_geom_ids"]["P_collision"],
            )
            p_points = points[:, p_mask]
            p_radii = radii[p_mask]
            candidate_clearance = (
                _scene_signed_distance(union_scene, p_points) - p_radii[None, :]
            )
            oracle_clearance = (
                _scene_signed_distance(oracle_scene, p_points) - p_radii[None, :]
            )
            candidate_min = candidate_clearance.min(axis=1)
            oracle_min = oracle_clearance.min(axis=1)
            candidate_contact = candidate_min <= 0.0
            oracle_contact = oracle_min <= 0.0
            phantom = candidate_contact & ~oracle_contact
            missed = ~candidate_contact & oracle_contact
            tp = candidate_contact & oracle_contact
            tn = ~candidate_contact & ~oracle_contact
            counts = {
                "pose_count": len(candidate_min),
                "true_positive_count": int(tp.sum()),
                "true_negative_count": int(tn.sum()),
                "phantom_contact_count": int(phantom.sum()),
                "missed_contact_count": int(missed.sum()),
            }
            for key, value in counts.items():
                aggregate[key] += value
            timelines.append(
                {
                    "source_family": entry["source_family"],
                    "candidate_contact": candidate_contact.tolist(),
                    "oracle_contact": oracle_contact.tolist(),
                    "counts": counts,
                }
            )
            for pose_index in np.flatnonzero(phantom):
                point_classes = _point_classes(
                    candidate_clearance[pose_index], oracle_clearance[pose_index]
                )
                representatives.append(
                    {
                        "source_family": entry["source_family"],
                        "pose_index": int(pose_index),
                        "candidate_min_clearance_m": float(candidate_min[pose_index]),
                        "oracle_min_clearance_m": float(oracle_min[pose_index]),
                        "phantom_point_count": int(point_classes["phantom"].sum()),
                        "points": p_points[pose_index],
                        "candidate_clearance": candidate_clearance[pose_index],
                        "oracle_clearance": oracle_clearance[pose_index],
                    }
                )
    if not representatives:
        raise RuntimeError("failed screen candidate has no recomputed phantom pose")
    representative = min(
        representatives,
        key=lambda value: (
            value["candidate_min_clearance_m"],
            -value["phantom_point_count"],
            value["source_family"],
            value["pose_index"],
        ),
    )
    classes = _point_classes(
        representative["candidate_clearance"], representative["oracle_clearance"]
    )
    pose_label = (
        f"{representative['source_family']} pose={representative['pose_index']} · "
        f"C min={1000 * representative['candidate_min_clearance_m']:.1f}mm · "
        f"M min={1000 * representative['oracle_min_clearance_m']:.1f}mm · "
        f"phantom points={representative['phantom_point_count']}"
    )
    prefix = f"bucket003_K{int(row['max_hulls']):02d}_{row['candidate_id']}"
    three_d_path = output_root / f"{prefix}_task_pose_3d.png"
    two_d_path = output_root / f"{prefix}_task_pose_2d.png"
    _render_3d(
        candidate_id=row["candidate_id"],
        max_hulls=int(row["max_hulls"]),
        parts=parts,
        oracle=oracle,
        points=representative["points"],
        classes=classes,
        pose_label=pose_label,
        output=three_d_path,
    )
    _render_2d(
        candidate_id=row["candidate_id"],
        max_hulls=int(row["max_hulls"]),
        union=union,
        oracle=oracle,
        points=representative["points"],
        classes=classes,
        timelines=timelines,
        pose_label=pose_label,
        output=two_d_path,
    )
    tp = aggregate["true_positive_count"]
    phantom = aggregate["phantom_contact_count"]
    missed = aggregate["missed_contact_count"]
    return {
        "object_key": OBJECT_KEY,
        "case_id": CASE_ID,
        "candidate_id": row["candidate_id"],
        "max_hulls": int(row["max_hulls"]),
        "actual_hulls": int(row["actual_hulls"]),
        "selection_role": "DIAGNOSTIC_BEST_FAILED_PER_K_NOT_FINALIST",
        "score_worst": float(row["score_worst"]),
        "fixture_sha256": fixture_sha,
        "candidate_asset_sha256": candidate["candidate_asset_sha256"],
        "aggregate_contact": {
            **aggregate,
            "precision": tp / (tp + phantom) if tp + phantom else 0.0,
            "recall": tp / (tp + missed) if tp + missed else 0.0,
        },
        "representative_pose": {
            key: value
            for key, value in representative.items()
            if key not in ("points", "candidate_clearance", "oracle_clearance")
        },
        "representative_point_classes": {
            key: int(value.sum()) for key, value in classes.items()
        },
        "three_d": {
            "path": relative_to_repo(three_d_path),
            "sha256": sha256_file(three_d_path),
        },
        "two_d": {
            "path": relative_to_repo(two_d_path),
            "sha256": sha256_file(two_d_path),
        },
    }


def main() -> int:
    """Render all three frozen best-failed bucket003 K diagnostics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--screen-root", type=Path, default=SCREEN_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    table = args.screen_root / "screen_candidates.tsv"
    aggregate = args.screen_root / "screen_aggregate.json"
    rows = select_failed_diagnostic_candidates(table)
    results = [
        diagnose_candidate(row, fixture_path=args.fixture, output_root=args.output_root)
        for row in rows
    ]
    payload = {
        "experiment_id": "E182",
        "stage": "S2_bucket003_P_contact_diagnostic",
        "status": "COMPLETE",
        "selection_eligible": False,
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "screen_table": {
            "path": relative_to_repo(table),
            "sha256": sha256_file(table),
        },
        "screen_aggregate": {
            "path": relative_to_repo(aggregate),
            "sha256": sha256_file(aggregate),
        },
        "candidate_count": len(results),
        "candidates": results,
    }
    atomic_json(args.output_root / "diagnostic_manifest.json", payload)
    print(f"E182_P_CONTACT_DIAGNOSTIC=COMPLETE candidates={len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
