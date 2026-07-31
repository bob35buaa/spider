#!/usr/bin/env python3
"""Render E178-style diagnostics for three rejected E181 CoACD candidates.

The renderer is deliberately post-terminal and read-only with respect to
Gate B.  It never reads heldout/full27 data, never creates ``C*``, and labels
every image ``REJECTED / NOT C*``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from build_authority import REPO_ROOT, relative_to_repo, sha256_file
from evaluate_asset_fidelity import (
    COACD_ROOT,
    CONVEX_CONTAINS_TOLERANCE_M,
    ORACLE_ROOT,
    contains_convex_union,
    convex_equations,
    load_parts,
    open3d_scene,
    scene_signed_distance,
)
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image, ImageDraw

OUTPUT_ROOT = (
    REPO_ROOT / "workspace/core4d/results/E181/s2_asset_eval/rejected_candidate_visuals"
)
FIXTURE_ROOT = REPO_ROOT / "workspace/core4d/results/E181/s2_asset_eval/fixtures"
CANDIDATE_METRICS = (
    REPO_ROOT / "workspace/core4d/results/E181/s2_asset_eval/candidate_metrics.tsv"
)
AZIMUTHS = (35.0, 125.0, 215.0, 305.0)
ELEVATIONS = (18.0, 18.0, 18.0, 35.0)
AXIS_NAMES = "XYZ"

MESH_BLUE = "#185adb"
COLLIDER_ORANGE = "#ff4b1f"
PHANTOM_MAGENTA = "#d01c8b"
M_ONLY_BLUE = np.asarray([0.094, 0.353, 0.859, 0.66])
BOTH_ORANGE = np.asarray([1.0, 0.294, 0.122, 0.58])
C_ONLY_MAGENTA = np.asarray([0.816, 0.110, 0.545, 0.78])
FREE_WHITE = np.asarray([1.0, 1.0, 1.0, 1.0])


@dataclass(frozen=True)
class CandidateSpec:
    """One frozen rejected candidate chosen for visual diagnosis."""

    object_key: str
    candidate_id: str
    selection_reason: str


CANDIDATES = (
    CandidateSpec(
        "bucket003",
        "t020_k32_v032",
        "lowest broader-cavity false occupancy for bucket003 (19.07%)",
    ),
    CandidateSpec(
        "bucket004",
        "t010_k32_v064",
        "closest to PASS: only broader-cavity fails (7.57%)",
    ),
    CandidateSpec(
        "bucket007",
        "t020_k32_v032",
        "lowest broader-cavity false occupancy for bucket007 (4.625%)",
    ),
)


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write stable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_mesh(path: Path) -> trimesh.Trimesh:
    """Load a single triangle mesh without changing topology."""
    loaded = trimesh.load(
        path,
        force="mesh",
        process=False,
        maintain_order=True,
    )
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.dump(concatenate=True)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"expected Trimesh: {path}")
    if not np.isfinite(loaded.vertices).all():
        raise ValueError(f"non-finite mesh vertices: {path}")
    return loaded


def sample_mesh_surface(
    mesh: trimesh.Trimesh,
    count: int,
) -> np.ndarray:
    """Sample a deterministic display point cloud."""
    points, _ = trimesh.sample.sample_surface(mesh, count, seed=181)
    return np.asarray(points, dtype=np.float64)


def part_triangles(parts: list[trimesh.Trimesh]) -> np.ndarray:
    """Concatenate triangles from all convex parts."""
    return np.concatenate(
        [np.asarray(part.triangles, dtype=np.float64) for part in parts],
        axis=0,
    )


def add_parts(
    axis: Any,
    parts: list[trimesh.Trimesh],
    *,
    alpha: float,
    categorical: bool,
) -> None:
    """Add convex parts to a Matplotlib 3D axis."""
    triangles = part_triangles(parts)
    if categorical:
        palette = matplotlib.colormaps["tab20"]
        colors = np.concatenate(
            [
                np.repeat(
                    np.asarray([palette(index % 20)]),
                    len(part.faces),
                    axis=0,
                )
                for index, part in enumerate(parts)
            ],
            axis=0,
        )
        colors[:, 3] = alpha
        edgecolor = (0.18, 0.05, 0.03, min(0.88, alpha + 0.28))
    else:
        colors = np.repeat(
            np.asarray([[1.0, 0.20, 0.03, alpha]]),
            len(triangles),
            axis=0,
        )
        edgecolor = (0.55, 0.03, 0.01, min(0.72, alpha + 0.30))
    collection = Poly3DCollection(
        triangles,
        facecolors=colors,
        edgecolors=edgecolor,
        linewidths=0.22,
    )
    axis.add_collection3d(collection)


def configure_3d_axis(
    axis: Any,
    center: np.ndarray,
    radius: float,
    *,
    azimuth: float,
    elevation: float,
    title: str,
) -> None:
    """Use the same orthographic four-camera semantics as E178."""
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(elev=elevation, azim=azimuth)
    axis.set_proj_type("ortho")
    axis.set_axis_off()
    axis.set_title(title, fontsize=8, pad=0)


def image_record(path: Path) -> dict[str, Any]:
    """Return a stable image artifact record."""
    with Image.open(path) as image:
        width, height = image.size
    return {
        "path": relative_to_repo(path),
        "sha256": sha256_file(path),
        "width": width,
        "height": height,
    }


def failed_gate_names(fidelity: dict[str, Any]) -> list[str]:
    """Return failed Gate B names in frozen evaluator order."""
    return [key for key, passed in fidelity["dev_hard_gates"].items() if not passed]


def render_3d(
    spec: CandidateSpec,
    mesh_points: np.ndarray,
    parts: list[trimesh.Trimesh],
    false_cavity_points: np.ndarray,
    fidelity: dict[str, Any],
    output: Path,
) -> None:
    """Render four E178-style views of M*, C, and their overlay."""
    part_points = np.concatenate([part.vertices for part in parts], axis=0)
    combined = np.concatenate(
        [mesh_points, part_points, false_cavity_points],
        axis=0,
    )
    lower = combined.min(axis=0)
    upper = combined.max(axis=0)
    center = 0.5 * (lower + upper)
    radius = max(float(np.max(upper - lower)) * 0.62, 0.05)

    figure = plt.figure(figsize=(11.2, 10.6), dpi=170)
    for view_index, (azimuth, elevation) in enumerate(
        zip(AZIMUTHS, ELEVATIONS, strict=True)
    ):
        modes = ("oracle M*", "CoACD convex parts", "M* + C + phantom")
        for column, mode in enumerate(modes):
            axis = figure.add_subplot(
                len(AZIMUTHS),
                len(modes),
                view_index * len(modes) + column + 1,
                projection="3d",
            )
            if mode != "CoACD convex parts":
                axis.scatter(
                    mesh_points[:, 0],
                    mesh_points[:, 1],
                    mesh_points[:, 2],
                    s=0.22,
                    c=MESH_BLUE,
                    alpha=0.60 if mode == "oracle M*" else 0.34,
                    depthshade=False,
                    rasterized=True,
                )
            if mode != "oracle M*":
                add_parts(
                    axis,
                    parts,
                    alpha=0.48 if mode == "CoACD convex parts" else 0.13,
                    categorical=mode == "CoACD convex parts",
                )
            if mode == "M* + C + phantom" and len(false_cavity_points):
                axis.scatter(
                    false_cavity_points[:, 0],
                    false_cavity_points[:, 1],
                    false_cavity_points[:, 2],
                    s=2.2,
                    c=PHANTOM_MAGENTA,
                    alpha=0.82,
                    depthshade=False,
                    rasterized=True,
                )
            configure_3d_axis(
                axis,
                center,
                radius,
                azimuth=azimuth,
                elevation=elevation,
                title=f"{mode} · az={azimuth:.0f}°",
            )

    metrics = fidelity["metrics"]
    failed_gates = failed_gate_names(fidelity)
    figure.suptitle(
        f"E181 {spec.object_key} · {spec.candidate_id} · "
        f"{len(parts)} hulls · REJECTED / NOT C*"
        "\n"
        f"broader-cavity={100 * metrics['broader_cavity_false_occupied_fraction']:.3f}% "
        "(gate≤0.1%)"
        "\n"
        f"failed={','.join(failed_gates)}",
        fontsize=11,
        y=0.997,
    )
    figure.text(
        0.5,
        0.007,
        "blue=M* surface · colored/orange=convex parts · "
        "magenta=known-free broad-cavity samples occupied by C",
        ha="center",
        fontsize=9,
    )
    figure.subplots_adjust(
        left=0.01,
        right=0.99,
        bottom=0.03,
        top=0.91,
        wspace=0.01,
        hspace=0.03,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, facecolor="white")
    plt.close(figure)


def plane_specs(
    mesh_bounds: np.ndarray,
) -> tuple[tuple[str, int, int, int, float], ...]:
    """Return E178-compatible XY/XZ/YZ diagnostic planes."""
    lower, upper = mesh_bounds
    center = 0.5 * (lower + upper)
    span = np.maximum(upper - lower, 1e-6)
    return (
        ("XY · mid Z", 0, 1, 2, center[2]),
        ("XY · upper Z", 0, 1, 2, lower[2] + 0.72 * span[2]),
        ("XZ · -Y base", 0, 2, 1, lower[1] + 0.01 * span[1]),
        ("XZ · mid Y", 0, 2, 1, center[1]),
        ("XZ · +Y end", 0, 2, 1, upper[1] - 0.005 * span[1]),
        ("YZ · mid X", 1, 2, 0, center[0]),
    )


def occupancy_rgba(
    m_inside: np.ndarray,
    c_inside: np.ndarray,
) -> np.ndarray:
    """Color 2D M/C occupancy classes."""
    rgba = np.empty((*m_inside.shape, 4), dtype=np.float64)
    rgba[:] = FREE_WHITE
    rgba[m_inside & ~c_inside] = M_ONLY_BLUE
    rgba[m_inside & c_inside] = BOTH_ORANGE
    rgba[~m_inside & c_inside] = C_ONLY_MAGENTA
    return rgba


def render_2d(
    spec: CandidateSpec,
    mesh: trimesh.Trimesh,
    mesh_points: np.ndarray,
    equations: list[np.ndarray],
    broad_points: np.ndarray,
    output: Path,
    *,
    grid_resolution: int,
) -> list[dict[str, Any]]:
    """Render six 2D occupancy/cross-section panels."""
    lower, upper = np.asarray(mesh.bounds, dtype=np.float64)
    margin = 0.04 * np.maximum(upper - lower, 1e-6)
    display_lower = lower - margin
    display_upper = upper + margin
    mesh_span = np.maximum(upper - lower, 1e-6)
    scene = open3d_scene(mesh)

    figure, axes = plt.subplots(2, 3, figsize=(13.8, 9.2), dpi=170)
    plane_records: list[dict[str, Any]] = []
    for axis, (title, horizontal, vertical, normal, level) in zip(
        axes.flat,
        plane_specs(mesh.bounds),
        strict=True,
    ):
        horizontal_values = np.linspace(
            display_lower[horizontal],
            display_upper[horizontal],
            grid_resolution,
        )
        vertical_values = np.linspace(
            display_lower[vertical],
            display_upper[vertical],
            grid_resolution,
        )
        grid_horizontal, grid_vertical = np.meshgrid(
            horizontal_values,
            vertical_values,
        )
        query = np.zeros(
            (grid_horizontal.size, 3),
            dtype=np.float64,
        )
        query[:, horizontal] = grid_horizontal.ravel()
        query[:, vertical] = grid_vertical.ravel()
        query[:, normal] = level

        m_inside = (
            scene_signed_distance(scene, query).reshape(grid_horizontal.shape) <= 0.0
        )
        c_inside = contains_convex_union(
            equations,
            query,
        ).reshape(grid_horizontal.shape)
        phantom = ~m_inside & c_inside
        axis.imshow(
            occupancy_rgba(m_inside, c_inside),
            origin="lower",
            extent=(
                horizontal_values[0],
                horizontal_values[-1],
                vertical_values[0],
                vertical_values[-1],
            ),
            interpolation="nearest",
            aspect="equal",
        )
        if m_inside.any() and (~m_inside).any():
            axis.contour(
                grid_horizontal,
                grid_vertical,
                m_inside.astype(np.float32),
                levels=[0.5],
                colors=[MESH_BLUE],
                linewidths=0.72,
            )
        if c_inside.any() and (~c_inside).any():
            axis.contour(
                grid_horizontal,
                grid_vertical,
                c_inside.astype(np.float32),
                levels=[0.5],
                colors=[COLLIDER_ORANGE],
                linewidths=0.72,
            )

        tolerance = max(0.018 * mesh_span[normal], 0.0025)
        near_mesh = np.abs(mesh_points[:, normal] - level) <= tolerance
        axis.scatter(
            mesh_points[near_mesh, horizontal],
            mesh_points[near_mesh, vertical],
            s=0.55,
            c=MESH_BLUE,
            alpha=0.54,
            rasterized=True,
        )
        near_broad = np.abs(broad_points[:, normal] - level) <= tolerance
        if near_broad.any():
            broad_query = broad_points[near_broad]
            occupied = contains_convex_union(equations, broad_query)
            false_slice = broad_query[occupied]
            axis.scatter(
                false_slice[:, horizontal],
                false_slice[:, vertical],
                s=5.0,
                marker="x",
                c=PHANTOM_MAGENTA,
                linewidths=0.48,
                alpha=0.88,
                rasterized=True,
            )
        else:
            false_slice = np.empty((0, 3), dtype=np.float64)

        axis.axhline(0.0, color="#bbbbbb", linewidth=0.3)
        axis.axvline(0.0, color="#bbbbbb", linewidth=0.3)
        axis.set_xlim(
            display_lower[horizontal],
            display_upper[horizontal],
        )
        axis.set_ylim(display_lower[vertical], display_upper[vertical])
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(
            f"{title}\nlevel={level:.3f}m · C-only={100 * phantom.mean():.2f}%",
            fontsize=9,
        )
        axis.set_xlabel(f"{AXIS_NAMES[horizontal]} (m)", fontsize=8)
        axis.set_ylabel(f"{AXIS_NAMES[vertical]} (m)", fontsize=8)
        axis.tick_params(labelsize=7)
        plane_records.append(
            {
                "name": title,
                "horizontal_axis": AXIS_NAMES[horizontal],
                "vertical_axis": AXIS_NAMES[vertical],
                "normal_axis": AXIS_NAMES[normal],
                "level_m": float(level),
                "grid_resolution": grid_resolution,
                "m_occupied_fraction": float(m_inside.mean()),
                "c_occupied_fraction": float(c_inside.mean()),
                "c_only_phantom_fraction": float(phantom.mean()),
                "false_broad_slice_count": int(len(false_slice)),
            }
        )

    figure.suptitle(
        f"E181 {spec.object_key} · {spec.candidate_id} · "
        "2D occupancy cross-sections · REJECTED / NOT C*\n"
        "blue=M* only · orange=M*∩C · magenta=C-only phantom",
        fontsize=12,
    )
    legend = (
        Patch(facecolor=M_ONLY_BLUE, label="M* only (collider miss)"),
        Patch(facecolor=BOTH_ORANGE, label="M* ∩ C"),
        Patch(facecolor=C_ONLY_MAGENTA, label="C only (phantom solid)"),
    )
    figure.legend(
        handles=legend,
        loc="lower center",
        ncol=3,
        fontsize=9,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.91))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, facecolor="white")
    plt.close(figure)
    return plane_records


def make_montage(
    image_rows: list[tuple[str, Path]],
    output: Path,
) -> None:
    """Create a two-column montage matching E178's handoff style."""
    target_width = 900
    label_height = 42
    rendered: list[tuple[str, Image.Image]] = []
    for label, path in image_rows:
        with Image.open(path) as source:
            image = source.convert("RGB")
            height = round(image.height * target_width / image.width)
            rendered.append(
                (
                    label,
                    image.resize((target_width, height)),
                )
            )
    cell_height = max(image.height for _, image in rendered) + label_height
    rows = (len(rendered) + 1) // 2
    canvas = Image.new(
        "RGB",
        (2 * target_width, rows * cell_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(rendered):
        row, column = divmod(index, 2)
        x = column * target_width
        y = row * cell_height
        draw.text((x + 10, y + 10), label, fill="black")
        canvas.paste(image, (x, y + label_height))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def render_candidate(
    spec: CandidateSpec,
    output_root: Path,
    *,
    sample_count: int,
    grid_resolution: int,
) -> dict[str, Any]:
    """Render one frozen candidate and return its manifest row."""
    candidate_path = COACD_ROOT / spec.object_key / spec.candidate_id / "manifest.json"
    candidate = load_json(candidate_path)
    if (
        candidate.get("status") != "BUILD_PASS"
        or candidate.get("object_key") != spec.object_key
        or candidate.get("candidate_id") != spec.candidate_id
    ):
        raise RuntimeError(f"candidate contract mismatch: {candidate_path}")
    parts = load_parts(candidate)
    equations = [convex_equations(part) for part in parts]

    oracle_manifest_path = ORACLE_ROOT / spec.object_key / "oracle_manifest.json"
    oracle_manifest = load_json(oracle_manifest_path)
    mesh_path = REPO_ROOT / oracle_manifest["cleaned_mesh"]["path"]
    if sha256_file(mesh_path) != oracle_manifest["cleaned_mesh"]["sha256"]:
        raise RuntimeError(f"oracle mesh SHA changed: {mesh_path}")
    mesh = load_mesh(mesh_path)
    mesh_points = sample_mesh_surface(mesh, sample_count)

    fixture_path = FIXTURE_ROOT / f"{spec.object_key}.npz"
    with np.load(fixture_path, allow_pickle=False) as fixture:
        broad_points = np.asarray(
            fixture["cavity_broad_points_m"],
            dtype=np.float64,
        )
    false_occupied = contains_convex_union(equations, broad_points)
    false_cavity_points = broad_points[false_occupied]
    if len(false_cavity_points) > 2_500:
        rng = np.random.default_rng(181)
        indices = rng.choice(
            len(false_cavity_points),
            size=2_500,
            replace=False,
        )
        display_false_cavity = false_cavity_points[indices]
    else:
        display_false_cavity = false_cavity_points

    fidelity_path = candidate_path.parent / "fidelity.json"
    fidelity = load_json(fidelity_path)
    if (
        fidelity.get("status") != "REJECT"
        or fidelity.get("candidate_asset_sha256") != candidate["candidate_asset_sha256"]
        or fidelity.get("heldout_status") != "SEALED_UNTIL_C_STAR"
    ):
        raise RuntimeError(f"fidelity contract mismatch: {fidelity_path}")
    metrics = fidelity["metrics"]
    if int(metrics["broader_cavity_false_occupied_count"]) != len(false_cavity_points):
        raise RuntimeError(
            f"{spec.object_key}: visual occupancy count drift "
            f"{len(false_cavity_points)} != "
            f"{metrics['broader_cavity_false_occupied_count']}"
        )

    prefix = f"{spec.object_key}_{spec.candidate_id}"
    three_d_path = output_root / f"{prefix}_3d_overlay.png"
    two_d_path = output_root / f"{prefix}_2d_cross_sections.png"
    render_3d(
        spec,
        mesh_points,
        parts,
        display_false_cavity,
        fidelity,
        three_d_path,
    )
    plane_records = render_2d(
        spec,
        mesh,
        mesh_points,
        equations,
        broad_points,
        two_d_path,
        grid_resolution=grid_resolution,
    )
    return {
        "object_key": spec.object_key,
        "candidate_id": spec.candidate_id,
        "selection_reason": spec.selection_reason,
        "candidate_status": candidate["status"],
        "gate_b_status": fidelity["status"],
        "heldout_status": fidelity["heldout_status"],
        "canonical_status": "REJECTED_NOT_C_STAR",
        "candidate_asset_sha256": candidate["candidate_asset_sha256"],
        "candidate_manifest": {
            "path": relative_to_repo(candidate_path),
            "sha256": sha256_file(candidate_path),
        },
        "oracle_manifest": {
            "path": relative_to_repo(oracle_manifest_path),
            "sha256": sha256_file(oracle_manifest_path),
        },
        "oracle_mesh": {
            "path": relative_to_repo(mesh_path),
            "sha256": sha256_file(mesh_path),
        },
        "fixture": {
            "path": relative_to_repo(fixture_path),
            "sha256": sha256_file(fixture_path),
        },
        "hull_count": candidate["hull_count"],
        "max_part_vertex_count": candidate["max_part_vertex_count"],
        "failed_dev_gates": failed_gate_names(fidelity),
        "metrics": metrics,
        "false_broad_count_recomputed": int(len(false_cavity_points)),
        "false_broad_display_count": int(len(display_false_cavity)),
        "three_d": image_record(three_d_path),
        "two_d": image_record(two_d_path),
        "two_d_planes": plane_records,
    }


def main() -> int:
    """Render the frozen three-candidate diagnostic package."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--sample-count", type=int, default=20_000)
    parser.add_argument("--grid-resolution", type=int, default=320)
    args = parser.parse_args()
    if args.sample_count < 2_000:
        raise ValueError("--sample-count must be at least 2000")
    if args.grid_resolution < 128:
        raise ValueError("--grid-resolution must be at least 128")

    rows = [
        render_candidate(
            spec,
            args.out_dir,
            sample_count=args.sample_count,
            grid_resolution=args.grid_resolution,
        )
        for spec in CANDIDATES
    ]
    three_d_montage = args.out_dir / "e181_rejected3_3d_montage.png"
    two_d_montage = args.out_dir / "e181_rejected3_2d_montage.png"
    make_montage(
        [
            (
                f"{row['object_key']} · {row['candidate_id']} · REJECTED / NOT C*",
                REPO_ROOT / row["three_d"]["path"],
            )
            for row in rows
        ],
        three_d_montage,
    )
    make_montage(
        [
            (
                f"{row['object_key']} · {row['candidate_id']} · REJECTED / NOT C*",
                REPO_ROOT / row["two_d"]["path"],
            )
            for row in rows
        ],
        two_d_montage,
    )
    manifest = {
        "experiment_id": "E181",
        "stage": "S2_rejected_candidate_visual_diagnostic",
        "status": "PASS",
        "gate_b_status": "ASSET_REJECTED",
        "heldout_status": "SEALED_NO_C_STAR",
        "canonical_status": "NO_C_STAR",
        "renderer_contract": {
            "style_authority": (
                "E178 four-azimuth mesh/collision/overlay and six-plane "
                "cross-section semantics"
            ),
            "azimuth_deg": list(AZIMUTHS),
            "elevation_deg": list(ELEVATIONS),
            "sample_count": args.sample_count,
            "grid_resolution": args.grid_resolution,
            "convex_contains_tolerance_m": CONVEX_CONTAINS_TOLERANCE_M,
            "heldout_access": "forbidden",
        },
        "candidate_metrics": {
            "path": relative_to_repo(CANDIDATE_METRICS),
            "sha256": sha256_file(CANDIDATE_METRICS),
        },
        "rows": rows,
        "montages": {
            "three_d": image_record(three_d_montage),
            "two_d": image_record(two_d_montage),
        },
    }
    manifest_path = args.out_dir / "render_manifest.json"
    write_json(manifest_path, manifest)
    print(
        "E181_REJECTED_VISUALS=PASS "
        f"candidates={len(rows)} output={relative_to_repo(args.out_dir)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
