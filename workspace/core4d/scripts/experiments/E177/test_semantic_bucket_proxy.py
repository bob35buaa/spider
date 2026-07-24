#!/usr/bin/env python3
"""Regression tests for E177 semantic bucket proxies."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from semantic_bucket_proxy import (
    EXPECTED_BOXES_BY_OBJECT,
    build_semantic_boxes,
    fidelity_metrics,
    load_mesh,
    proxy_xml,
    union_fidelity_metrics,
)


REPO = Path(__file__).resolve().parents[5]
OBJECT_ROOT = (
    REPO / "example_datasets/processed/core4d/assets/objects"
)


def main() -> int:
    for object_key, expected in EXPECTED_BOXES_BY_OBJECT.items():
        mesh_path = OBJECT_ROOT / object_key / f"{object_key}_m.obj"
        mesh = load_mesh(mesh_path)
        boxes, metadata = build_semantic_boxes(mesh_path, object_key)
        assert len(boxes) == expected
        assert metadata["object_geom_count"] == expected
        assert metadata["mesh_center_inside_count"] >= 1
        names_xml, names = proxy_xml(boxes)
        assert names[0] == "object_collision"
        assert len(names) == len(set(names)) == expected
        assert names_xml.count("<geom") == expected

        for box in boxes:
            assert box.center.shape == (3,)
            assert box.half_size.shape == (3,)
            assert np.isfinite(box.center).all()
            assert np.isfinite(box.half_size).all()
            assert np.all(box.half_size > 0.0)

        proxy_lower = np.min(
            [box.center - box.half_size for box in boxes],
            axis=0,
        )
        proxy_upper = np.max(
            [box.center + box.half_size for box in boxes],
            axis=0,
        )
        assert np.all(proxy_lower >= mesh.bounds[0] - 0.005)
        assert np.all(proxy_upper <= mesh.bounds[1] + 0.005)
        if object_key == "bucket004":
            np.testing.assert_allclose(proxy_lower, mesh.bounds[0])
            np.testing.assert_allclose(proxy_upper, mesh.bounds[1])
        else:
            assert metadata["body_layers"] == 5
            assert metadata["has_separate_lid"] is False
            assert all(
                box.label == f"body_{index:03d}"
                for index, box in enumerate(boxes)
            )
            assert metadata["frustum_axis"] == "local_y_positive"
            assert metadata["cross_section_axes"] == "local_xz"
            assert all(
                left.center[1] < right.center[1]
                for left, right in zip(boxes[:-1], boxes[1:], strict=True)
            )
            assert (
                boxes[0].center[1] - boxes[0].half_size[1]
                <= mesh.bounds[0, 1] + 1e-9
            )
            assert (
                boxes[-1].center[1] + boxes[-1].half_size[1]
                >= mesh.bounds[1, 1] - 1e-9
            )

        metrics = fidelity_metrics(
            mesh_path,
            boxes,
            mesh_sample_count=8_000,
            proxy_samples_per_face=32,
        )
        union_metrics = union_fidelity_metrics(
            mesh_path,
            boxes,
            mesh_sample_count=8_000,
            proxy_samples_per_face=32,
        )
        if object_key != "bucket004":
            assert union_metrics["union_mesh_to_proxy_p90_m"] <= 0.04
            assert union_metrics["union_proxy_to_mesh_p90_m"] <= 0.04
        print(
            object_key,
            f"boxes={len(boxes)}",
            f"mesh2proxy_p90={metrics['mesh_to_proxy_p90_m']:.4f}",
            f"proxy2mesh_p90={metrics['proxy_to_mesh_p90_m']:.4f}",
            "union="
            f"{union_metrics['union_mesh_to_proxy_p90_m']:.4f}/"
            f"{union_metrics['union_proxy_to_mesh_p90_m']:.4f}",
            f"policy={metadata['collision_policy']}",
        )
    print("E177 semantic bucket proxy tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
