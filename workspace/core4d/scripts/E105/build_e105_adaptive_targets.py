#!/usr/bin/env python3
"""Build E105 adaptive-support targets by reusing the E094 projection implementation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))

from e105_common import REPO, RESULTS_ROOT, rel, rows  # noqa: E402


OUT_ROOT = RESULTS_ROOT / "adaptive_targets"


def main() -> None:
    cmd = [
        sys.executable,
        "workspace/core4d/scripts/E094/build_handbox_target_projection.py",
        "--out-root",
        str(OUT_ROOT),
        "--case-ids",
        "box026_039_p2",
        "box026_135_p2",
        "--reward-mode",
        "adaptive_support",
        "--force",
    ]
    print("[E105-adaptive] " + " ".join(cmd))
    subprocess.run(cmd, cwd=REPO, check=True)

    expected = [REPO / row["target_npz"] for row in rows() if row["route"] == "adaptive_clean"]
    missing = [path for path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing E105 adaptive targets: " + ", ".join(map(str, missing)))

    metadata = []
    case_targets = OUT_ROOT / "case_targets.json"
    if case_targets.is_file():
        metadata = json.loads(case_targets.read_text(encoding="utf-8"))
    (OUT_ROOT / "e105_adaptive_target_meta.json").write_text(
        json.dumps({"target_root": rel(OUT_ROOT), "targets": metadata}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[E105-adaptive] wrote {rel(OUT_ROOT)}")


if __name__ == "__main__":
    main()
