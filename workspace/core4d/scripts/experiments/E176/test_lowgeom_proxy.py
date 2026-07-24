#!/usr/bin/env python3
"""Regression tests for E176 <=9-box coarse surface proxies."""

from __future__ import annotations

import csv
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(HERE))
sys.path.insert(
    0,
    str(REPO / "workspace/core4d/scripts/experiments/E175"),
)

from build_nonbox_multigeom_production import resolve_object_mesh  # noqa: E402
from lowgeom_proxy import (  # noqa: E402
    EXPECTED_BOXES_BY_OBJECT,
    MAX_OBJECT_GEOMS,
    build_lowgeom_boxes,
    fidelity_metrics,
    proxy_xml,
)


E174_MANIFEST = (
    REPO
    / "workspace/core4d/results/E174/s6_downstream/manifests/"
    "cem_full_manifest.tsv"
)


def main() -> int:
    rows = list(
        csv.DictReader(
            E174_MANIFEST.open(encoding="utf-8"),
            delimiter="\t",
        )
    )
    source_by_object = {}
    for row in rows:
        source_by_object.setdefault(row["object_key"], row)
    assert set(source_by_object) == set(EXPECTED_BOXES_BY_OBJECT)

    for object_key, expected in EXPECTED_BOXES_BY_OBJECT.items():
        mesh = resolve_object_mesh(
            Path(source_by_object[object_key]["scene_act"])
        )
        boxes, metadata = build_lowgeom_boxes(mesh, object_key)
        xml, names = proxy_xml(boxes)
        elements = ET.fromstring(f"<body>{xml}</body>").findall("geom")

        assert len(boxes) == expected
        assert len(boxes) <= MAX_OBJECT_GEOMS
        assert metadata["center_inside_count"] == 0
        assert len(names) == len(set(names)) == len(elements)
        assert names[0] == "object_collision"
        assert all(element.get("type") == "box" for element in elements)
        assert all(
            np.all(box.half_size > 0.0)
            and np.isfinite(box.center).all()
            and np.isfinite(box.half_size).all()
            for box in boxes
        )

        metrics = fidelity_metrics(
            mesh,
            boxes,
            mesh_sample_count=4_000,
            proxy_samples_per_face=32,
        )
        assert metrics["mesh_to_proxy_p90_m"] <= 0.08, (
            object_key,
            metrics,
        )
        assert metrics["proxy_to_mesh_p90_m"] <= 0.16, (
            object_key,
            metrics,
        )
        print(
            object_key,
            f"cells={metadata['target_cells']}",
            f"boxes={len(boxes)}",
            f"mesh2proxy_p90={metrics['mesh_to_proxy_p90_m']:.4f}",
            f"proxy2mesh_p90={metrics['proxy_to_mesh_p90_m']:.4f}",
        )
    print("E176 low-geom proxy tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

