#!/usr/bin/env python3
"""Render object-local mesh/box proxy overlays with Matplotlib/Agg.

This is a software-rendering fallback for machines without a working
MuJoCo EGL/OSMesa context.  It reads the exact mesh and ``object_collision*``
box geoms recorded by a production builder and produces four-view
mesh/collision/overlay sheets plus an input-sized montage.
"""

from __future__ import annotations

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation


REPO = Path(__file__).resolve().parents[5]
DEFAULT_INPUT = (
    REPO
    / "workspace/core4d/results/E175/scene_snapshot/nonbox_multigeom/"
    "proxy_visual_review.tsv"
)
DEFAULT_OUTPUT = (
    REPO
    / "workspace/core4d/results/E175/scene_snapshot/nonbox_multigeom/"
    "proxy_visual_evidence"
)
BOX_FACES = (
    (0, 1, 3, 2),
    (4, 6, 7, 5),
    (0, 4, 5, 1),
    (2, 3, 7, 6),
    (0, 2, 6, 4),
    (1, 5, 7, 3),
)
EXPERIMENT_LABEL = "E175"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def repo_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPO / path


def rel(path: Path) -> str:
    return str(path.absolute().relative_to(REPO.absolute()))


def values(raw: str | None, count: int, default: list[float]) -> np.ndarray:
    if not raw:
        return np.asarray(default, dtype=np.float64)
    parsed = np.asarray([float(item) for item in raw.split()])
    if parsed.shape != (count,):
        raise ValueError(f"expected {count} values, got {raw!r}")
    return parsed


def transform_points(
    points: np.ndarray, pos: np.ndarray, quat_wxyz: np.ndarray
) -> np.ndarray:
    if np.allclose(quat_wxyz, [1.0, 0.0, 0.0, 0.0]):
        return points + pos
    rotation = Rotation.from_quat(
        [
            quat_wxyz[1],
            quat_wxyz[2],
            quat_wxyz[3],
            quat_wxyz[0],
        ]
    )
    return rotation.apply(points) + pos


def object_body(root: ET.Element) -> ET.Element:
    matches = [
        body for body in root.iter("body") if body.get("name") == "object"
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one object body, got {len(matches)}")
    return matches[0]


def box_vertices(geom: ET.Element) -> np.ndarray:
    if geom.get("type", "sphere") != "box":
        raise ValueError(
            f"non-box object proxy geom: {geom.get('name')} "
            f"type={geom.get('type')}"
        )
    half = values(geom.get("size"), 3, [0.0, 0.0, 0.0])
    signs = np.asarray(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ],
        dtype=np.float64,
    )
    local = signs * half
    return transform_points(
        local,
        values(geom.get("pos"), 3, [0.0, 0.0, 0.0]),
        values(geom.get("quat"), 4, [1.0, 0.0, 0.0, 0.0]),
    )


def scene_proxy(scene_path: Path) -> tuple[list[np.ndarray], ET.Element]:
    root = ET.parse(scene_path).getroot()
    body = object_body(root)
    geoms = [
        geom
        for geom in body.findall("geom")
        if (geom.get("name") or "") == "object_collision"
        or (geom.get("name") or "").startswith("object_collision_")
    ]
    if not geoms:
        raise ValueError(f"no object collision geoms: {scene_path}")
    return [box_vertices(geom) for geom in geoms], body


def load_mesh_points(
    mesh_path: Path, body: ET.Element, sample_count: int
) -> tuple[trimesh.Trimesh, np.ndarray]:
    loaded = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.to_geometry()
    else:
        mesh = loaded
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"unsupported mesh: {type(mesh)}")
    points, _ = trimesh.sample.sample_surface(
        mesh, sample_count, seed=0
    )
    visual = next(
        (
            geom
            for geom in body.findall("geom")
            if geom.get("name") == "object_visual"
        ),
        None,
    )
    if visual is not None:
        points = transform_points(
            points,
            values(visual.get("pos"), 3, [0.0, 0.0, 0.0]),
            values(
                visual.get("quat"), 4, [1.0, 0.0, 0.0, 0.0]
            ),
        )
    return mesh, points


def add_boxes(
    axis: Any, boxes: list[np.ndarray], alpha: float
) -> None:
    polygons = [
        [vertices[index] for index in face]
        for vertices in boxes
        for face in BOX_FACES
    ]
    collection = Poly3DCollection(
        polygons,
        facecolors=(1.0, 0.20, 0.03, alpha),
        edgecolors=(0.55, 0.03, 0.01, min(1.0, alpha + 0.35)),
        linewidths=0.22,
    )
    axis.add_collection3d(collection)


def configure_axis(
    axis: Any,
    center: np.ndarray,
    radius: float,
    azimuth: float,
    elevation: float,
    title: str,
) -> None:
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(elev=elevation, azim=azimuth)
    axis.set_proj_type("ortho")
    axis.set_axis_off()
    axis.set_title(title, fontsize=8, pad=0)


def render_cross_sections(
    object_key: str,
    points: np.ndarray,
    boxes: list[np.ndarray],
    out_dir: Path,
) -> Path:
    combined = np.concatenate([points, *boxes], axis=0)
    lower = combined.min(axis=0)
    upper = combined.max(axis=0)
    center = 0.5 * (lower + upper)
    span = np.maximum(upper - lower, 1e-6)
    mesh_lower = points.min(axis=0)
    mesh_upper = points.max(axis=0)
    mesh_span = np.maximum(mesh_upper - mesh_lower, 1e-6)
    planes = (
        ("XY · mid Z", 0, 1, 2, center[2]),
        ("XY · upper Z", 0, 1, 2, lower[2] + 0.72 * span[2]),
        (
            "XZ · -Y base",
            0,
            2,
            1,
            mesh_lower[1] + 0.01 * mesh_span[1],
        ),
        ("XZ · mid Y", 0, 2, 1, center[1]),
        (
            "XZ · +Y end",
            0,
            2,
            1,
            mesh_upper[1] - 0.005 * mesh_span[1],
        ),
        ("YZ · mid X", 1, 2, 0, center[0]),
    )
    figure, axes = plt.subplots(2, 3, figsize=(12.75, 8.5), dpi=170)
    for axis, (title, horizontal, vertical, normal, level) in zip(
        axes.flat, planes, strict=True
    ):
        tolerance = max(0.018 * span[normal], 0.0025)
        near = np.abs(points[:, normal] - level) <= tolerance
        axis.scatter(
            points[near, horizontal],
            points[near, vertical],
            s=1.0,
            c="#185adb",
            alpha=0.62,
            label="mesh slice",
        )
        intersected = 0
        for vertices in boxes:
            if (
                vertices[:, normal].min() - 1e-12
                <= level
                <= vertices[:, normal].max() + 1e-12
            ):
                lo = vertices[:, [horizontal, vertical]].min(axis=0)
                hi = vertices[:, [horizontal, vertical]].max(axis=0)
                axis.add_patch(
                    Rectangle(
                        lo,
                        *(hi - lo),
                        facecolor=(1.0, 0.20, 0.03, 0.22),
                        edgecolor=(0.60, 0.03, 0.01, 0.75),
                        linewidth=0.45,
                    )
                )
                intersected += 1
        axis.axhline(0.0, color="#bbbbbb", linewidth=0.3)
        axis.axvline(0.0, color="#bbbbbb", linewidth=0.3)
        axis.set_xlim(lower[horizontal], upper[horizontal])
        axis.set_ylim(lower[vertical], upper[vertical])
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(
            f"{title} · level={level:.3f} m · boxes={intersected}",
            fontsize=9,
        )
        axis.set_xlabel("XYZ"[horizontal] + " (m)", fontsize=8)
        axis.set_ylabel("XYZ"[vertical] + " (m)", fontsize=8)
        axis.tick_params(labelsize=7)
    figure.suptitle(
        f"{EXPERIMENT_LABEL} {object_key} · cutaway shell evidence\n"
        "blue=mesh surface slice · orange=proxy boxes intersecting plane",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    output = out_dir / f"{object_key}_proxy_cross_sections.png"
    figure.savefig(output, facecolor="white")
    plt.close(figure)
    return output


def render_object(
    row: dict[str, str],
    out_dir: Path,
    sample_count: int,
) -> dict[str, Any]:
    object_key = row["object_key"]
    scene_path = repo_path(row.get("scene_xml") or row["scene_act"])
    mesh_path = repo_path(row["mesh_path"])
    boxes, body = scene_proxy(scene_path)
    mesh, points = load_mesh_points(mesh_path, body, sample_count)
    box_points = np.concatenate(boxes, axis=0)
    combined = np.concatenate([points, box_points], axis=0)
    lower = combined.min(axis=0)
    upper = combined.max(axis=0)
    center = 0.5 * (lower + upper)
    radius = max(float(np.max(upper - lower)) * 0.62, 0.05)

    azimuths = (35.0, 125.0, 215.0, 305.0)
    elevations = (18.0, 18.0, 18.0, 35.0)
    figure = plt.figure(figsize=(10.5, 10.0), dpi=150)
    for view_index, (azimuth, elevation) in enumerate(
        zip(azimuths, elevations, strict=True)
    ):
        for column, mode in enumerate(
            ("real mesh", "collision proxy", "mesh + collision")
        ):
            axis = figure.add_subplot(
                len(azimuths), 3, view_index * 3 + column + 1,
                projection="3d",
            )
            if mode != "collision proxy":
                axis.scatter(
                    points[:, 0],
                    points[:, 1],
                    points[:, 2],
                    s=0.18,
                    c="#185adb",
                    alpha=0.60 if mode == "real mesh" else 0.34,
                    depthshade=False,
                    rasterized=True,
                )
            if mode != "real mesh":
                add_boxes(
                    axis,
                    boxes,
                    0.48 if mode == "collision proxy" else 0.25,
                )
            configure_axis(
                axis,
                center,
                radius,
                azimuth,
                elevation,
                f"{mode} · az={azimuth:.0f}°",
            )
    figure.suptitle(
        f"{EXPERIMENT_LABEL} {object_key} · {row['proxy_variant']} · "
        f"{len(boxes)} boxes",
        fontsize=13,
        y=0.995,
    )
    figure.subplots_adjust(
        left=0.01, right=0.99, bottom=0.01, top=0.97,
        wspace=0.01, hspace=0.03,
    )
    sheet = out_dir / f"{object_key}_mesh_collision_overlay.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(sheet, facecolor="white")
    plt.close(figure)
    cross_section = (
        render_cross_sections(object_key, points, boxes, out_dir)
        if object_key.startswith("bucket")
        else None
    )
    return {
        "object_key": object_key,
        "proxy_variant": row["proxy_variant"],
        "proxy_policy": row["proxy_policy"],
        "scene_xml": rel(scene_path),
        "mesh_path": rel(mesh_path),
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_faces": int(len(mesh.faces)),
        "sample_count": int(len(points)),
        "object_geom_count": len(boxes),
        "combined_aabb_min": lower.tolist(),
        "combined_aabb_max": upper.tolist(),
        "render_status": "pass",
        "sheet_path": rel(sheet),
        "cross_section_path": (
            rel(cross_section) if cross_section is not None else ""
        ),
    }


def montage(
    rendered: list[dict[str, Any]], output: Path
) -> Path:
    images = [
        Image.open(REPO / row["sheet_path"]).convert("RGB")
        for row in rendered
    ]
    target_width = 760
    resized = []
    for image in images:
        target_height = round(image.height * target_width / image.width)
        resized.append(image.resize((target_width, target_height)))
    label_height = 34
    cell_height = max(image.height for image in resized) + label_height
    grid_rows = (len(resized) + 1) // 2
    canvas = Image.new(
        "RGB", (target_width * 2, cell_height * grid_rows), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for index, (row, image) in enumerate(
        zip(rendered, resized, strict=True)
    ):
        grid_y, grid_x = divmod(index, 2)
        x = grid_x * target_width
        y = grid_y * cell_height
        draw.text(
            (x + 8, y + 8),
            f"{row['object_key']} · {row['object_geom_count']} boxes",
            fill="black",
        )
        canvas.paste(image, (x, y + label_height))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


def cross_section_montage(
    rendered: list[dict[str, Any]], output: Path
) -> Path:
    bucket_rows = [
        row for row in rendered if row["cross_section_path"]
    ]
    images = [
        Image.open(REPO / row["cross_section_path"]).convert("RGB")
        for row in bucket_rows
    ]
    target_width = 700
    resized = [
        image.resize(
            (
                target_width,
                round(image.height * target_width / image.width),
            )
        )
        for image in images
    ]
    cell_height = max(image.height for image in resized) + 32
    grid_rows = (len(resized) + 1) // 2
    canvas = Image.new(
        "RGB", (target_width * 2, cell_height * grid_rows), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for index, (row, image) in enumerate(
        zip(bucket_rows, resized, strict=True)
    ):
        grid_y, grid_x = divmod(index, 2)
        x = grid_x * target_width
        y = grid_y * cell_height
        draw.text((x + 8, y + 8), row["object_key"], fill="black")
        canvas.paste(image, (x, y + 32))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


def main() -> int:
    global EXPERIMENT_LABEL
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-tsv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-count", type=int, default=12000)
    parser.add_argument("--experiment-label", default="E175")
    args = parser.parse_args()
    EXPERIMENT_LABEL = args.experiment_label
    file_prefix = args.experiment_label.lower()
    rows = read_tsv(args.input_tsv)
    if not rows:
        raise ValueError(f"no object review rows: {args.input_tsv}")
    rendered = [
        render_object(row, args.out_dir, args.sample_count)
        for row in rows
    ]
    montage_label = (
        "six_object" if len(rendered) == 6 else f"{len(rendered)}_object"
    )
    montage_path = montage(
        rendered,
        args.out_dir / f"{file_prefix}_{montage_label}_proxy_montage.png",
    )
    cross_section_path = cross_section_montage(
        rendered,
        args.out_dir / f"{file_prefix}_bucket_cross_section_montage.png",
    )
    write_tsv(args.out_dir / "proxy_overlay_render_manifest.tsv", rendered)
    write_json(
        args.out_dir / "proxy_overlay_render_manifest.json", rendered
    )
    summary = {
        "created_at": now(),
        "rows": len(rendered),
        "status_counts": dict(
            Counter(row["render_status"] for row in rendered)
        ),
        "montage": rel(montage_path),
        "bucket_cross_section_montage": rel(cross_section_path),
        "status": (
            "pass"
            if rendered
            and all(row["render_status"] == "pass" for row in rendered)
            else "fail"
        ),
    }
    write_json(args.out_dir / "proxy_overlay_render_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
