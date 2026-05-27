#!/usr/bin/env python3
"""E082 evaluator.

E082 intentionally reuses the E081 leg/foot-object metric implementation.
Output fields retain the E081_* prefix where they are inherited metric names.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]


def _env_path(name: str, default: Path) -> Path:
    path = Path(os.environ.get(name, str(default)))
    if not path.is_absolute():
        path = REPO / path
    os.environ[name] = str(path)
    return path


RESULTS = _env_path("RESULTS", REPO / "workspace/core4d/results/E082")
VARIANTS_FILE = _env_path("VARIANTS_FILE", REPO / "workspace/core4d/scripts/E082/variants.tsv")

EVAL_DIR = REPO / "workspace/core4d/scripts/eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import eval_E081  # noqa: E402


eval_E081.RESULTS = RESULTS
eval_E081.VARIANTS_FILE = VARIANTS_FILE


if __name__ == "__main__":
    eval_E081.main()
