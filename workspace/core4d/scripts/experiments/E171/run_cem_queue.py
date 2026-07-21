#!/usr/bin/env python3
"""E171 entry point for the regression-tested generic E169 CEM queue runner.

Thin shim (same pattern as E170): loads and runs the shared E169 runner, which
reads scene_name / override_id / target_task per manifest row (no object-specific
assumptions) and validates the frozen leg-gate config against each row.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


RUNNER = Path(__file__).resolve().parents[1] / "E169/run_cem_queue.py"
SPEC = importlib.util.spec_from_file_location("core4d_cem_queue_runner", RUNNER)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load queue runner: {RUNNER}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


if __name__ == "__main__":
    raise SystemExit(MODULE.main())
