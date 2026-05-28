"""Compare original vs B-path-repaired trajectory_kinematic.npz under the
G1-Feasibility gate.

`g1_feasibility_gate.py` now defines `top_face_frac` as the face whose outward
normal best aligns with world +Z, so rotated boxes such as box021 no longer
need a supplementary metric.

Usage:
    python gate_compare_b_path.py <task_orig> <task_repaired>
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import json

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "workspace/exp_diagnostic/scripts"))
from g1_feasibility_gate import evaluate, fmt_row, GATE  # noqa: E402

TASKS_DIR = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"


def main():
    tasks = sys.argv[1:]
    rows = []
    for t in tasks:
        td = TASKS_DIR / t
        r = evaluate(td)
        rows.append(r)
    print(f"Gate: {GATE}\n")
    for r in rows:
        print(fmt_row(r))
    out = REPO / "workspace/exp_diagnostic/findings/08_B_path_gate_compare.json"
    out.write_text(json.dumps({"rows": [{
        "task": r.get("task"),
        "T": r.get("T"),
        "L": r.get("L"),
        "R": r.get("R"),
        "pelvis_z_min": r.get("pelvis_z_min"),
        "gate_pass": r.get("gate_pass"),
        "gate_reject_reasons": r.get("gate_reject_reasons"),
    } for r in rows]}, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
