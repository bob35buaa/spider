"""E100 build_fingertip_aware_target 单元测试 - 3 个 known case."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from build_fingertip_aware_target import build_target  # noqa: E402


def _check(label: str, ok: bool, msg: str = "") -> int:
    marker = "✓" if ok else "✗"
    print(f"{marker} {label}  {msg}")
    return 1 if ok else 0


def main() -> None:
    base = Path("example_datasets/processed/core4d/unitree_g1/humanoid_object")
    fdir = Path("workspace/core4d/results/E099/fingertip_vote_per_case")

    cases = [
        # (case, hand, vote_face, expected_axis_value)
        ("box023_person2", "R", "+z", "expect R target z ≈ +half_z"),
        ("e091_box026_20231018_039_p2", "L", "-z", "expect L target z ≈ -half_z"),
        ("d003_box021_20231018_030_p1", "R", "+x", "expect R target x ≈ +half_x"),
    ]
    passed = 0
    total = 0
    for case, hand, exp_face, msg in cases:
        scene = base / case / "scene.xml"
        traj = base / case / "0" / "trajectory_kinematic.npz"
        vote_path = fdir / f"{case}.json"
        vote = json.loads(vote_path.read_text())
        r = build_target(case, scene, traj, vote)
        target = r["spider_contact_target_object_local"]  # (T, 2, 3)
        half = np.array(r["summary"]["half"])
        h_idx = 0 if hand == "L" else 1
        axis = "xyz".index(exp_face[1])
        sign = 1.0 if exp_face[0] == "+" else -1.0
        expected = sign * half[axis]
        # 取活跃帧的平均 target axis 值，应贴 face (容差 0.5 cm)
        active = r["active"][:, h_idx]
        if not active.any():
            passed += _check(case, False, f"{hand} no active frames")
            total += 1
            continue
        mean_axis = float(target[active, h_idx, axis].mean())
        ok = abs(mean_axis - expected) < 0.005
        total += 1
        passed += _check(
            f"{case} {hand}", ok,
            f"vote={exp_face} expected_axis={expected:.4f} got={mean_axis:.4f} ({msg})"
        )
    print(f"\n{passed}/{total} PASS")
    if passed < total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
