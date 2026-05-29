"""单元测试：covers 6 面 + corner / edge / inside 各种情况。

运行方式:
    .venv/bin/python workspace/core4d/scripts/E098/test_face_utils.py
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import face_utils as fu  # noqa: E402


def assert_eq(actual, expected, msg=""):
    if actual != expected:
        raise AssertionError(f"{msg}: expected {expected!r}, got {actual!r}")


def test_face_label_six_faces():
    half = np.array([0.16, 0.21, 0.265])
    # +x face
    assert_eq(fu.face_label(np.array([0.15, 0.0, 0.0]), half), "+x", "near +x face")
    # -x face
    assert_eq(fu.face_label(np.array([-0.15, 0.0, 0.0]), half), "-x", "near -x face")
    # +y face
    assert_eq(fu.face_label(np.array([0.0, 0.20, 0.0]), half), "+y", "near +y face")
    # -y face
    assert_eq(fu.face_label(np.array([0.0, -0.20, 0.0]), half), "-y", "near -y face")
    # +z face — 这是 B1 修复的关键，旧代码永远不会返回 ±z
    assert_eq(fu.face_label(np.array([0.0, 0.0, 0.26]), half), "+z", "near +z face (B1 fix)")
    # -z face
    assert_eq(fu.face_label(np.array([0.0, 0.0, -0.26]), half), "-z", "near -z face (B1 fix)")
    print("✓ test_face_label_six_faces PASS")


def test_face_label_z_dominant_real_case():
    """模拟 02 §3 B1 表里 box021 18029_p2 R 主面 = +z 的情况。"""
    half = np.array([0.16, 0.21, 0.265])
    # 模拟一个 z 主导的接触点（手掌贴在 box 顶部偏外）
    point = np.array([0.05, 0.04, 0.20])
    # 归一化 (0.31, 0.19, 0.74) → +z
    assert_eq(fu.face_label(point, half), "+z", "z-dominant point (B1 critical)")
    # 旧 xy-only 算法会输出 +x（因为 |0.05/0.16|=0.31 > |0.04/0.21|=0.19）
    print("✓ test_face_label_z_dominant_real_case PASS")


def test_face_label_batch():
    half = np.array([1.0, 1.0, 1.0])
    points = np.array(
        [
            [0.9, 0.0, 0.0],  # +x
            [0.0, -0.9, 0.0],  # -y
            [0.0, 0.0, 0.9],  # +z
            [0.0, 0.0, -0.9],  # -z
        ]
    )
    labels = fu.face_label(points, half)
    assert_eq(labels, ["+x", "-y", "+z", "-z"], "batch face_label")
    print("✓ test_face_label_batch PASS")


def test_face_axis_sign():
    assert_eq(fu.face_axis_sign("+x"), (0, 1.0), "axis_sign +x")
    assert_eq(fu.face_axis_sign("-x"), (0, -1.0), "axis_sign -x")
    assert_eq(fu.face_axis_sign("+y"), (1, 1.0), "axis_sign +y")
    assert_eq(fu.face_axis_sign("-y"), (1, -1.0), "axis_sign -y")
    assert_eq(fu.face_axis_sign("+z"), (2, 1.0), "axis_sign +z (B2/B3 fix)")
    assert_eq(fu.face_axis_sign("-z"), (2, -1.0), "axis_sign -z (B2/B3 fix)")
    print("✓ test_face_axis_sign PASS")


def test_face_axis_sign_invalid():
    try:
        fu.face_axis_sign("x")  # missing sign
        raise AssertionError("Expected ValueError for invalid face")
    except ValueError:
        pass
    try:
        fu.face_axis_sign("+w")
        raise AssertionError("Expected ValueError for invalid axis")
    except ValueError:
        pass
    print("✓ test_face_axis_sign_invalid PASS")


def test_project_to_face_z():
    """B2 修复：±z 投影必须正确改 z 分量，不能 silent fallback 到 y."""
    half = np.array([0.16, 0.21, 0.265])
    point = np.array([0.05, -0.10, 0.0])
    # 投到 +z
    out = fu.project_to_face(point, half, "+z")
    expected = np.array([0.05, -0.10, 0.265])
    if not np.allclose(out, expected):
        raise AssertionError(f"project_to_face +z: expected {expected}, got {out}")
    # 投到 -z
    out = fu.project_to_face(point, half, "-z")
    expected = np.array([0.05, -0.10, -0.265])
    if not np.allclose(out, expected):
        raise AssertionError(f"project_to_face -z: expected {expected}, got {out}")
    print("✓ test_project_to_face_z PASS")


def test_snap_to_face_top():
    """snap to +z 面：x,y clip 到 ±0.65·half，z = +half[2]."""
    half = np.array([0.16, 0.21, 0.265])
    point = np.array([0.20, 0.30, -0.10])  # x,y 都超出 0.65·half
    out = fu.snap_to_face(point, half, "+z")
    assert np.isclose(out[2], 0.265), f"z must = +half_z, got {out[2]}"
    assert abs(out[0]) <= 0.65 * 0.16 + 1e-9, f"x clipped: {out[0]}"
    assert abs(out[1]) <= 0.65 * 0.21 + 1e-9, f"y clipped: {out[1]}"
    print("✓ test_snap_to_face_top PASS")


def test_majority_face_corner():
    """corner 情况：多面票数相近，按 FACE_ORDER 优先。"""
    half = np.array([1.0, 1.0, 1.0])
    # 三个点分别接近 +x, +y, +z；总票数相等
    pts = np.array(
        [
            [0.95, 0.0, 0.0],
            [0.0, 0.95, 0.0],
            [0.0, 0.0, 0.95],
        ]
    )
    face, votes, total = fu.majority_face(pts, half)
    # 平局，FACE_ORDER 第一名是 +x
    assert_eq((face, votes, total), ("+x", 1, 3), "majority corner tie -> +x")
    print("✓ test_majority_face_corner PASS")


def test_face_stats_full_six():
    """6 个 face 各 1 点，应当 6 个面各 1 票。"""
    half = np.array([1.0, 1.0, 1.0])
    pts = np.array(
        [
            [0.9, 0.0, 0.0],
            [-0.9, 0.0, 0.0],
            [0.0, 0.9, 0.0],
            [0.0, -0.9, 0.0],
            [0.0, 0.0, 0.9],
            [0.0, 0.0, -0.9],
        ]
    )
    counts = fu.face_stats(pts, half)
    for face in fu.ALL_FACES:
        if counts[face] != 1:
            raise AssertionError(f"face {face} expected 1 vote, got {counts[face]}")
    print("✓ test_face_stats_full_six PASS")


def main():
    tests = [
        test_face_axis_sign,
        test_face_axis_sign_invalid,
        test_face_label_six_faces,
        test_face_label_z_dominant_real_case,
        test_face_label_batch,
        test_project_to_face_z,
        test_snap_to_face_top,
        test_majority_face_corner,
        test_face_stats_full_six,
    ]
    failures = []
    for t in tests:
        try:
            t()
        except Exception as e:
            failures.append((t.__name__, str(e)))
            print(f"✗ {t.__name__} FAIL: {e}")
    print()
    print(f"Ran {len(tests)} tests, {len(tests) - len(failures)} passed, {len(failures)} failed.")
    if failures:
        for name, err in failures:
            print(f"  - {name}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
