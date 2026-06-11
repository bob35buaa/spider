#!/usr/bin/env python3
"""Compatibility wrapper for the migrated E154 comparison workbook generator."""

from __future__ import annotations

import runpy
from pathlib import Path


TARGET = Path(__file__).resolve().parents[1] / "eval/reports/gen_E154_compare_table.py"
runpy.run_path(str(TARGET), run_name="__main__")
