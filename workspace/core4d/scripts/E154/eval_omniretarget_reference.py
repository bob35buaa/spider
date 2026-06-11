#!/usr/bin/env python3
"""Compatibility wrapper for the migrated E154 OmniRetarget evaluator."""

from __future__ import annotations

import runpy
from pathlib import Path


TARGET = Path(__file__).resolve().parents[1] / "eval/runners/eval_E154_omniretarget_reference.py"
runpy.run_path(str(TARGET), run_name="__main__")
