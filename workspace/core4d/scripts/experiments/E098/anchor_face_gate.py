"""B5 修复 (exp_diagnostic_v2 §2)：anchor_face_review=true hard block helper。

老流程：E028/manifest.tsv 里 anchor_face_review=true 只是 informational，
flagged case 仍然进入 E082-E088 full CEM 队列。本 helper 在 manifest
读取处拦截，强制要求经过 B1 修复后的 anchor refit 重做才放行。

使用方式（推荐由 caller 在读 manifest 后立即调用）::

    from workspace.core4d.scripts.E098 import anchor_face_gate
    rows = read_manifest(...)
    anchor_face_gate.assert_no_unrefit_review(rows, allow_refit_tag="E098_refit")

如果有 row 标了 anchor_face_review=true 且 anchor_refit_tag 不在白名单，
即 raise AnchorFaceReviewError。
"""

from __future__ import annotations

from collections.abc import Iterable


class AnchorFaceReviewError(RuntimeError):
    """Raise when a manifest row carries anchor_face_review=true but没经过新版
    B1-fixed anchor refit。"""


def _truthy(val) -> bool:
    if isinstance(val, bool):
        return val
    if val is None:
        return False
    s = str(val).strip().lower()
    return s in {"1", "true", "yes", "y", "t"}


def assert_no_unrefit_review(
    rows: Iterable[dict],
    *,
    refit_tag_field: str = "anchor_refit_tag",
    review_field: str = "anchor_face_review",
    allow_refit_tags: tuple[str, ...] = ("E098_refit",),
    case_field: str = "source_task",
) -> None:
    """检查 rows 中没有未经 E098 refit 的 anchor_face_review case。

    Args:
        rows: 任意可迭代 dict（如 read_manifest 输出）。
        refit_tag_field: 标识该 row 是否经过新 refit 的字段名。
        review_field: anchor_face_review flag 字段名。
        allow_refit_tags: 允许通过的 refit tag。默认仅 "E098_refit"。
        case_field: 出错信息里展示的 case 名字段。

    Raises:
        AnchorFaceReviewError: 任何一行同时满足 review=true 且 refit_tag
            不在白名单。
    """
    offenders: list[tuple[str, str]] = []
    for row in rows:
        if not _truthy(row.get(review_field)):
            continue
        tag = str(row.get(refit_tag_field, "")).strip()
        if tag in allow_refit_tags:
            continue
        offenders.append((str(row.get(case_field, "?")), tag or "<missing>"))
    if not offenders:
        return
    msg_lines = [
        f"B5 hard block: {len(offenders)} manifest rows have "
        f"{review_field}=true but no allowed {refit_tag_field}.",
        "These cases must be re-anchored with the B1-fixed face_utils helper "
        "before entering full CEM. Allowed tags: " + ", ".join(allow_refit_tags),
        "Offending rows:",
    ]
    for case, tag in offenders[:20]:
        msg_lines.append(f"  - case={case!r}  current_tag={tag!r}")
    if len(offenders) > 20:
        msg_lines.append(f"  ... and {len(offenders) - 20} more")
    raise AnchorFaceReviewError("\n".join(msg_lines))


# 简单自检
if __name__ == "__main__":
    # 情况 1：无 review 的 row 应放行
    assert_no_unrefit_review([{"source_task": "x", "anchor_face_review": "false"}])
    print("✓ no-review rows pass")
    # 情况 2：review=true + 正确 refit tag 应放行
    assert_no_unrefit_review([
        {"source_task": "x", "anchor_face_review": "true",
         "anchor_refit_tag": "E098_refit"}
    ])
    print("✓ refit-tagged rows pass")
    # 情况 3：review=true + 无 tag 应 raise
    try:
        assert_no_unrefit_review([
            {"source_task": "x", "anchor_face_review": "true"}
        ])
    except AnchorFaceReviewError as e:
        print(f"✓ unrefit raises: {str(e).splitlines()[0]}")
    else:
        raise SystemExit("FAIL: should have raised")
    print("All anchor_face_gate self-checks PASS")
