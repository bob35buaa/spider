"""统一的 face 选择 / 投影 helper。

修复 exp_diagnostic_v2 §2 的 B1/B2/B3：
- B1: 老 `face_label` 只用 xy 二维 argmax，完全屏蔽 ±z 面
- B2: E028b._project_to_face 把 ±z 静默映射到 ±y
- B3: E029 D6 asset writer 把非侧面默认回退 +x

所有 caller 应当 import 这里的函数，不要再各自实现。

约定：
- ``point`` 是 box local frame 下的 3D 坐标，shape (3,) 或 (N, 3)
- ``half`` 是 box 的半边长 (hx, hy, hz)，shape (3,)
- ``face`` 字符串 in {"+x","-x","+y","-y","+z","-z"}
"""

from __future__ import annotations

import numpy as np

FACE_ORDER: list[str] = ["+x", "-x", "+y", "-y", "+z", "-z"]
ALL_FACES: tuple[str, ...] = ("+x", "-x", "+y", "-y", "+z", "-z")
SIDE_FACES: tuple[str, ...] = ("+x", "-x", "+y", "-y")  # 历史含义保留：仅水平 4 面


def _validate_face(face: str) -> None:
    if face not in ALL_FACES:
        raise ValueError(
            f"Invalid face {face!r}; must be one of {ALL_FACES}"
        )


def face_axis_sign(face: str) -> tuple[int, float]:
    """返回 (axis, sign)，axis ∈ {0,1,2} 对应 xyz，sign ∈ {+1,-1}。

    修复 B1/B2/B3：原来许多调用站点写
        axis = 0 if face.endswith('x') else 1
    把 ±z 静默归到 y 上，是 wrong-target bug。
    """
    _validate_face(face)
    axis = "xyz".index(face[1])
    sign = 1.0 if face[0] == "+" else -1.0
    return axis, sign


def face_label(point: np.ndarray, half: np.ndarray) -> str:
    """返回 ``point`` 最贴近的 6 面之一（全 3D argmax）。

    修复 B1：原 `face_label(point, half)` 只看 ``point[:2]``，永远不返回 ±z。

    Args:
        point: shape (3,) 或 (N,3)，box local frame 坐标
        half: shape (3,) 半边长，正数

    Returns:
        per-point face string；若 point 是 1D 则返回单个 str，否则返回 list[str]
    """
    half = np.asarray(half, dtype=np.float64)
    if np.any(half <= 0):
        raise ValueError(f"half must be positive, got {half}")
    pt = np.asarray(point, dtype=np.float64)
    if pt.ndim == 1:
        norm = np.abs(pt) / np.clip(half, 1e-6, None)
        axis = int(np.argmax(norm))
        sign = "+" if pt[axis] >= 0.0 else "-"
        return f"{sign}{'xyz'[axis]}"
    if pt.ndim == 2 and pt.shape[1] == 3:
        norm = np.abs(pt) / np.clip(half, 1e-6, None)[None, :]
        axes = np.argmax(norm, axis=1)
        signs = np.where(pt[np.arange(len(pt)), axes] >= 0.0, "+", "-")
        names = np.array(list("xyz"))[axes]
        return [f"{s}{n}" for s, n in zip(signs, names)]
    raise ValueError(f"point must have shape (3,) or (N,3), got {pt.shape}")


def face_stats(
    points: np.ndarray,
    half: np.ndarray,
) -> dict[str, int]:
    """统计点云在 6 个面上的投票数。

    Args:
        points: shape (N,3)
        half: shape (3,)

    Returns:
        ``{face: count}``，含 6 面（票数为 0 也填）
    """
    labels = face_label(points, half)
    counts = {face: 0 for face in ALL_FACES}
    for lab in labels:
        counts[lab] += 1
    return counts


def majority_face(
    points: np.ndarray,
    half: np.ndarray,
) -> tuple[str, int, int]:
    """返回得票最多的面 + 票数 + 总点数；平局时按 FACE_ORDER 优先。

    Args:
        points: shape (N,3)
        half: shape (3,)

    Returns:
        (face, votes, total)
    """
    counts = face_stats(points, half)
    total = sum(counts.values())
    # 按 (-票数, FACE_ORDER index) 排序
    ranked = sorted(
        counts.items(),
        key=lambda kv: (-kv[1], FACE_ORDER.index(kv[0])),
    )
    top_face, top_count = ranked[0]
    return top_face, top_count, total


def clip_to_box_interior(
    point: np.ndarray,
    half: np.ndarray,
    margin_frac: float = 0.90,
) -> np.ndarray:
    """将 3D 点 clip 到 box 内部边界（每个轴在 ±margin_frac·half 之内）。

    修复 B1：原 `_clip_anchor` 把 z 强压到 `[-0.25·hz, 0.65·hz]`，意味着锚点
    永远不能在底面或顶面；新版对称裁剪所有 3 个轴。
    """
    pt = np.asarray(point, dtype=np.float64).copy()
    half = np.asarray(half, dtype=np.float64)
    pt = np.clip(pt, -margin_frac * half, margin_frac * half)
    return pt


def snap_to_face(
    point: np.ndarray,
    half: np.ndarray,
    face: str,
    other_axis_margin_frac: float = 0.65,
) -> np.ndarray:
    """把 3D 点 snap 到指定面的中心区域内。

    修复 B1：原 `_snap_to_face` 用 `other = 1 - axis`，只对 2D 有效。
    新版：把 `face` 对应轴设为 ±half；其它两个轴 clip 到 ±margin·half。
    """
    _validate_face(face)
    axis, sign = face_axis_sign(face)
    pt = clip_to_box_interior(point, half, margin_frac=0.90)
    pt[axis] = sign * half[axis]
    others = [i for i in (0, 1, 2) if i != axis]
    for j in others:
        pt[j] = float(np.clip(pt[j], -other_axis_margin_frac * half[j], other_axis_margin_frac * half[j]))
    return pt


def project_to_face(
    point: np.ndarray,
    half: np.ndarray,
    face: str,
) -> np.ndarray:
    """把 3D 点投影到指定面（保留另外两轴的坐标，沿 face 法向贴到 ±half）。

    修复 B2：原 E028b._project_to_face `axis = 0 if face.endswith("x") else 1`
    把 ±z 静默映射到 y。新版用 face_axis_sign 正确解析 6 面。
    """
    _validate_face(face)
    axis, sign = face_axis_sign(face)
    pt = np.asarray(point, dtype=np.float64).copy()
    pt[axis] = sign * half[axis]
    return pt
