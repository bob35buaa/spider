"""E099 fingertip_face_vote 单元测试 - 3 个 known case 投票主面."""

from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from fingertip_face_vote import vote_case  # noqa: E402

EXPECT = [
    # (case, hand, expected_face, min_frac, min_contact)
    # box023_person2: 已知双手提箱顶面 → +z
    ("box023_person2", "L", "+z", 0.90, 10),
    ("box023_person2", "R", "+z", 0.85, 30),
    # box025_person2: 双手托底/提顶；fingertip 全程 +z 100%
    ("box025_person2", "L", "+z", 0.95, 30),
    ("box025_person2", "R", "+z", 0.95, 30),
    # d003_box021_20231018_030_p1: L 顶面 +z；R 侧面 +x
    ("d003_box021_20231018_030_p1", "L", "+z", 0.85, 20),
    ("d003_box021_20231018_030_p1", "R", "+x", 0.80, 20),
    # e091_box026_20231018_039_p2: 双手底面托 -z
    ("e091_box026_20231018_039_p2", "L", "-z", 0.95, 50),
    ("e091_box026_20231018_039_p2", "R", "-z", 0.95, 50),
]


def main() -> None:
    scene_base = Path("example_datasets/processed/core4d/unitree_g1/humanoid_object")
    passed = 0
    failed = 0
    for case, hand, exp_face, min_frac, min_contact in EXPECT:
        scene = scene_base / case / "scene.xml"
        r = vote_case(case, scene_xml=scene if scene.is_file() else None)
        if r["status"] != "ok":
            print(f"✗ {case} {hand}: status={r['status']}")
            failed += 1
            continue
        info = r[hand]
        ok = (
            info["vote_face"] == exp_face
            and info["vote_frac"] >= min_frac
            and info["contact_frames"] >= min_contact
        )
        marker = "✓" if ok else "✗"
        print(
            f"{marker} {case} {hand}: vote={info['vote_face']}({info['vote_frac']*100:.1f}%, n={info['contact_frames']}) "
            f"vs exp {exp_face} (≥{min_frac*100:.0f}%, n≥{min_contact})"
        )
        if ok:
            passed += 1
        else:
            failed += 1
    total = passed + failed
    print(f"\n{passed}/{total} PASS")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
