#!/usr/bin/env python3
"""Compatibility wrapper for the E077 contact mask script path."""

from __future__ import annotations

import runpy
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]


if __name__ == "__main__":
    runpy.run_path(
        str(REPO / "workspace/core4d/data_preprocess/generate_core4d_contact_masks.py"),
        run_name="__main__",
    )
