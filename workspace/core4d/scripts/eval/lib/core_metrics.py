"""Compatibility wrapper for the migrated CORE4D evaluation core.

New code should import from `eval/core/core_metrics.py` via the scripts root.
This module remains so historical evaluators using `from lib.core_metrics import ...`
continue to work.
"""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[2]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from eval.core.core_metrics import *  # noqa: F401,F403,E402
