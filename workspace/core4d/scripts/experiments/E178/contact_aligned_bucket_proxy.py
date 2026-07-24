#!/usr/bin/env python3
"""E178 contact-aligned top-segment refinement of the E177 bucket proxy."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
E177_DIR = HERE.parent / "E177"
sys.path.insert(0, str(E177_DIR))

import semantic_bucket_proxy as e177  # noqa: E402


LOWER_BODY_XZ_SCALE_BY_OBJECT = {
    "bucket003": 0.94,
    "bucket007": 0.82,
}
TOP_SEGMENT_XZ_SCALE_BY_OBJECT = {
    "bucket003": (0.95, 0.95),
    "bucket007": (0.97, 0.885),
}
EXPECTED_BOXES_BY_OBJECT = e177.EXPECTED_BOXES_BY_OBJECT
ProxyBox = e177.ProxyBox
fidelity_metrics = e177.fidelity_metrics
load_mesh = e177.load_mesh
point_to_proxy_surface_distance = e177.point_to_proxy_surface_distance
proxy_xml = e177.proxy_xml
union_fidelity_metrics = e177.union_fidelity_metrics


def build_contact_aligned_boxes(
    mesh_path: Path,
    object_key: str,
) -> tuple[list[ProxyBox], dict[str, Any]]:
    boxes, raw_metadata = e177.build_semantic_boxes(mesh_path, object_key)
    metadata = copy.deepcopy(raw_metadata)
    if object_key == "bucket004":
        metadata["top_segment_xz_scale"] = None
        metadata["lower_body_xz_scale"] = None
        return boxes, metadata

    lower_scale = LOWER_BODY_XZ_SCALE_BY_OBJECT[object_key]
    top_x_scale, top_z_scale = TOP_SEGMENT_XZ_SCALE_BY_OBJECT[
        object_key
    ]
    if len(boxes) != 5:
        raise AssertionError(f"{object_key} expected 5 boxes")
    if not np.isclose(
        float(metadata["body_xz_inward_scale"]),
        lower_scale,
    ):
        raise AssertionError(f"{object_key} E177 lower scale drift")

    top = boxes[-1]
    top_half = top.half_size.copy()
    top_half[0] *= top_x_scale / lower_scale
    top_half[2] *= top_z_scale / lower_scale
    boxes[-1] = ProxyBox(
        center=top.center.copy(),
        half_size=top_half,
        label=top.label,
    )

    mesh = load_mesh(mesh_path)
    proxy_lower = np.min(
        [box.center - box.half_size for box in boxes],
        axis=0,
    )
    proxy_upper = np.max(
        [box.center + box.half_size for box in boxes],
        axis=0,
    )
    tolerance = 0.005
    if np.any(proxy_lower < mesh.bounds[0] - tolerance) or np.any(
        proxy_upper > mesh.bounds[1] + tolerance
    ):
        raise AssertionError(
            f"{object_key} E178 proxy exceeds mesh AABB tolerance"
        )

    top_row = metadata["layer_rows"][-1]
    top_row["xz_lower_m"] = (
        boxes[-1].center[[0, 2]] - boxes[-1].half_size[[0, 2]]
    ).tolist()
    top_row["xz_upper_m"] = (
        boxes[-1].center[[0, 2]] + boxes[-1].half_size[[0, 2]]
    ).tolist()
    top_row["xz_inward_scale"] = [top_x_scale, top_z_scale]
    top_row["contact_aligned_top_segment"] = True
    metadata.update(
        {
            "collision_policy": (
                "contact_aligned_top_segment_five_solid_body_steps_no_lid"
            ),
            "proxy_aabb_lower_m": proxy_lower.tolist(),
            "proxy_aabb_upper_m": proxy_upper.tolist(),
            "lower_body_xz_scale": lower_scale,
            "top_segment_xz_scale": [top_x_scale, top_z_scale],
            "contact_alignment_source": (
                "E177_ref_fk_contact_fidelity_active_targets"
            ),
        }
    )
    return boxes, metadata


__all__ = [
    "EXPECTED_BOXES_BY_OBJECT",
    "LOWER_BODY_XZ_SCALE_BY_OBJECT",
    "TOP_SEGMENT_XZ_SCALE_BY_OBJECT",
    "ProxyBox",
    "build_contact_aligned_boxes",
    "fidelity_metrics",
    "load_mesh",
    "point_to_proxy_surface_distance",
    "proxy_xml",
    "union_fidelity_metrics",
]
