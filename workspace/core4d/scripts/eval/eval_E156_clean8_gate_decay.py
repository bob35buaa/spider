#!/usr/bin/env python3
"""Compatibility wrapper for E156 clean8 gate/decay evaluation."""

from __future__ import annotations

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent / "runners" / "eval_E156_clean8_gate_decay.py"), run_name="__main__")
