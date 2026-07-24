#!/usr/bin/env python3
"""Build the E178 27-case contact-aligned bucket proxy authority."""

from __future__ import annotations

import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
E177_DIR = HERE.parent / "E177"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(E177_DIR))

import build_semantic_bucket_production as production  # noqa: E402
from contact_aligned_bucket_proxy import (  # noqa: E402
    build_contact_aligned_boxes,
)


production.EXPERIMENT_ID = "E178"
production.RESULTS = REPO / "workspace/core4d/results/E178"
production.SCENE_NAME = "scene_act_E178_contactAlignedTop"
production.METHOD_ID = (
    "E178_semantic_bucket_contact_aligned_top_segment_union_r1"
)
production.CELL_ID = "contact_aligned_top_segment"
production.PROXY_TAG = "contactAlignedTop"
production.FIVE_BODY_PROXY_VARIANT = (
    "five_body_steps_no_lid_contact_aligned_top"
)
production.build_semantic_boxes = build_contact_aligned_boxes


if __name__ == "__main__":
    raise SystemExit(production.main())
