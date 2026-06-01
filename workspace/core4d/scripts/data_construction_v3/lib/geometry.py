#!/usr/bin/env python3
"""Shared geometry helpers for Core4D data-construction v3."""

from __future__ import annotations

from pathlib import Path
import sys

SCRIPT_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "data_construction_v3")
for _path in (SCRIPT_ROOT / "lib", SCRIPT_ROOT / "state", SCRIPT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np


FACE_LABELS = ("+x", "-x", "+y", "-y", "+z", "-z")
CONTACT_POS_SOURCES = ("fk_palm_site", "raw_fingertip", "external_target")


def load_obj_vertices(path: Path) -> np.ndarray:
    vertices: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            except ValueError:
                continue
    if not vertices:
        raise ValueError(f"no vertices found in object mesh: {path}")
    return np.asarray(vertices, dtype=np.float64)


def aabb_extents(vertices: np.ndarray) -> np.ndarray:
    arr = np.asarray(vertices, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
        raise ValueError(f"vertices must have shape (N, 3), got {arr.shape}")
    return arr.max(axis=0) - arr.min(axis=0)


def aabb_center(vertices: np.ndarray) -> np.ndarray:
    arr = np.asarray(vertices, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
        raise ValueError(f"vertices must have shape (N, 3), got {arr.shape}")
    return (arr.max(axis=0) + arr.min(axis=0)) * 0.5


def face_argmax_3d(points_object_local: np.ndarray, box_center: np.ndarray, half_extents: np.ndarray) -> list[str]:
    """Assign each point to the nearest of six AABB faces in object-local 3D."""
    points = np.asarray(points_object_local, dtype=np.float64)
    center = np.asarray(box_center, dtype=np.float64).reshape(3)
    half = np.asarray(half_extents, dtype=np.float64).reshape(3)
    if points.ndim == 1:
        points = points.reshape(1, 3)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    if np.any(half <= 0):
        raise ValueError(f"half_extents must be positive, got {half}")

    rel = points - center
    normalized = np.stack(
        [
            rel[:, 0] / half[0],
            -rel[:, 0] / half[0],
            rel[:, 1] / half[1],
            -rel[:, 1] / half[1],
            rel[:, 2] / half[2],
            -rel[:, 2] / half[2],
        ],
        axis=1,
    )
    indices = np.argmax(normalized, axis=1)
    return [FACE_LABELS[int(index)] for index in indices]


def normalize_contact_pos_source(has_contact_pos: bool, source_hint: str = "") -> str:
    hint = str(source_hint or "").strip()
    if hint in CONTACT_POS_SOURCES:
        return hint
    if has_contact_pos:
        return "fk_palm_site"
    return ""


def self_test() -> None:
    half = np.asarray([1.0, 2.0, 3.0])
    points = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, -2.0, 0.0],
            [0.0, 0.0, 3.0],
            [0.0, 0.0, -3.0],
        ]
    )
    labels = face_argmax_3d(points, np.zeros(3), half)
    if labels != list(FACE_LABELS):
        raise AssertionError(f"face_argmax_3d failed: {labels}")
    if normalize_contact_pos_source(True) != "fk_palm_site":
        raise AssertionError("contact_pos default source failed")


if __name__ == "__main__":
    self_test()
