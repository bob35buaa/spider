#!/usr/bin/env python3
"""Regression and geometry gates for E178 contact-aligned bucket proxies."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from contact_aligned_bucket_proxy import (
    EXPECTED_BOXES_BY_OBJECT,
    LOWER_BODY_XZ_SCALE_BY_OBJECT,
    TOP_SEGMENT_XZ_SCALE_BY_OBJECT,
    build_contact_aligned_boxes,
    union_fidelity_metrics,
)


REPO = Path(__file__).resolve().parents[5]
OBJECT_ROOT = (
    REPO / "example_datasets/processed/core4d/assets/objects"
)


def main() -> int:
    for object_key, expected in EXPECTED_BOXES_BY_OBJECT.items():
        mesh_path = OBJECT_ROOT / object_key / f"{object_key}_m.obj"
        boxes, metadata = build_contact_aligned_boxes(
            mesh_path,
            object_key,
        )
        assert len(boxes) == expected
        assert metadata["object_geom_count"] == expected
        assert metadata["has_separate_lid"] is False
        if object_key != "bucket004":
            assert len(boxes) == 5
            assert np.isclose(
                metadata["lower_body_xz_scale"],
                LOWER_BODY_XZ_SCALE_BY_OBJECT[object_key],
            )
            np.testing.assert_allclose(
                metadata["top_segment_xz_scale"],
                TOP_SEGMENT_XZ_SCALE_BY_OBJECT[object_key],
            )
            assert metadata["layer_rows"][-1][
                "contact_aligned_top_segment"
            ]
            assert all(
                np.isclose(
                    metadata["layer_rows"][index]["xz_inward_scale"],
                    LOWER_BODY_XZ_SCALE_BY_OBJECT[object_key],
                )
                for index in range(4)
            )

        metrics = union_fidelity_metrics(
            mesh_path,
            boxes,
            mesh_sample_count=8_000,
            proxy_samples_per_face=64,
        )
        assert metrics["union_mesh_to_proxy_p90_m"] <= 0.04
        assert metrics["union_proxy_to_mesh_p90_m"] <= 0.04
        print(
            object_key,
            f"boxes={len(boxes)}",
            "union="
            f"{metrics['union_mesh_to_proxy_p90_m']:.4f}/"
            f"{metrics['union_proxy_to_mesh_p90_m']:.4f}",
        )
    print("E178 contact-aligned bucket proxy tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
