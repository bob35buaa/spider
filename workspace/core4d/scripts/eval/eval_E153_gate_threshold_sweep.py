#!/usr/bin/env python3
"""Compatibility wrapper for the migrated E153 evaluator."""

from __future__ import annotations

import runpy
from pathlib import Path


TARGET = Path(__file__).resolve().parent / "runners" / Path(__file__).name
runpy.run_path(str(TARGET), run_name="__main__")
