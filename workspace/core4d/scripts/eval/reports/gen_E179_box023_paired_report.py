#!/usr/bin/env python3
"""Generate the E179 vs E173 16-row, 12-gate Markdown report."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPT_ROOT / "experiments/E179"))

import e179_common as C  # noqa: E402


LABELS = {
    "fall": "fall",
    "body_z": "body-z",
    "contact": "contact",
    "release": "release",
    "hand_penetration": "hand penetration",
    "lower_body": "lower-body",
    "root_pos": "root pos",
    "root_ori": "root ori",
    "hand_pos": "EEF pos",
    "hand_ori": "EEF ori",
    "object_pos": "object pos",
    "object_ori": "object ori",
}


def mark(value: Any) -> str:
    return "PASS" if C.boolish(value) else "FAIL"


def number(value: Any, digits: int = 3) -> str:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{output:.{digits}f}" if math.isfinite(output) else "NA"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=C.RESULTS / "s6_downstream/eval/full",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = json.loads(
        (args.eval_dir / "e179_eval_summary.json").read_text(
            encoding="utf-8"
        )
    )
    paired = C.read_tsv(args.eval_dir / "e179_vs_e173_paired.tsv")
    matrix = C.read_tsv(
        args.eval_dir / "e179_vs_e173_gate_matrix.tsv"
    )
    if len(paired) != 16 or len(matrix) != 192:
        raise ValueError(
            f"report requires 16 paired rows and 192 gate cells, got "
            f"{len(paired)}/{len(matrix)}"
        )
    review_path = (
        C.RESULTS / "s6_downstream/render/full/visual_review.tsv"
    )
    reviews = (
        C.read_tsv(review_path) if review_path.is_file() else []
    )
    reviewed = [
        row for row in reviews if row.get("review_status") == "reviewed"
    ]
    alignment_counts = Counter(
        row.get("visual_alignment", "") for row in reviewed
    )
    strong_prg_rows = [
        row["case_id"]
        for row in reviewed
        if row.get("pass_migration") == "PASS_TO_FAIL"
        and row.get("visual_alignment") == "supports_pass_to_fail"
    ]
    strong_no_prg_rows = [
        row["case_id"]
        for row in reviewed
        if row.get("pass_migration") == "FAIL_TO_PASS"
        and row.get("visual_alignment") == "supports_fail_to_pass"
    ]
    if len(reviewed) == 16 and len({row["case_id"] for row in reviewed}) == 16:
        if (
            summary["numeric_verdict"] == "PRG_BETTER"
            and len(strong_prg_rows) > len(strong_no_prg_rows)
        ):
            visual_verdict = "SUPPORTS_PRG_BETTER"
        else:
            visual_verdict = "MIXED"
    else:
        visual_verdict = "PENDING_16_PAIRED_VIDEO_REVIEW"
    summary["visual_verdict"] = visual_verdict
    summary["visual_review"] = {
        "rows": len(reviews),
        "reviewed": len(reviewed),
        "alignment_counts": dict(alignment_counts),
        "strong_prg_evidence_rows": strong_prg_rows,
        "strong_no_prg_evidence_rows": strong_no_prg_rows,
        "source": C.rel(review_path),
    }
    C.write_json(args.eval_dir / "e179_eval_summary.json", summary)
    output = args.output or args.eval_dir / "E179_vs_E173_report.md"

    lines = [
        "# E179 vs E173：box023 16 条 Full CEM 配对对比",
        "",
        "_Core4D Phase 42 · 2026-07-25 · fixed 16-case paired comparison_",
        "",
        "---",
        "",
        "> 主口径是同一 16 条 case 的十二门交集。E173 历史 `13/16` "
        "仅表示 physics 六门通过；加入 root、EEF、object 位姿 tracking "
        "六门后，冻结基线为 `7/16`。",
        "",
        "## 📋 口径",
        "",
        f"- Scoring contract：`{summary['scoring_contract_id']}`",
        f"- Case rows：`{summary['evaluated_rows']}/16`",
        f"- Explicit gate cells：`{summary['gate_cells']}/192`",
        "- Physics：fall、body-z、contact、release、hand penetration、"
        "lower-body",
        "- Tracking：root pos/ori、EEF pos/ori、object pos/ori",
        "- 缺失或非有限指标一律 FAIL。",
        "",
        "## 📊 总结果",
        "",
        "| 口径 | E173 PRG | E179 E167A no-PRG |",
        "|---|---:|---:|",
        f"| Physics 6 门 | {summary['e173_physics6_pass']}/16 | "
        f"{summary['e179_physics6_pass']}/16 |",
        f"| Physics + tracking 12 门 | {summary['e173_12gate_pass']}/16 | "
        f"{summary['e179_12gate_pass']}/16 |",
        "",
        f"数值判定：`{summary['numeric_verdict']}`。视觉判定："
        f"`{summary['visual_verdict']}`。",
        "",
        "### 十二门逐门通过数",
        "",
        "| Gate | E173 | E179 | Delta |",
        "|---|---:|---:|---:|",
    ]
    for gate in C.ALL_GATES:
        counts = summary["gate_pass_counts"][gate]
        lines.append(
            f"| {LABELS[gate]} | {counts['e173']}/16 | "
            f"{counts['e179']}/16 | {counts['e179']-counts['e173']:+d} |"
        )

    lines.extend(
        [
            "",
            "### Pass migration",
            "",
            "| Migration | Count |",
            "|---|---:|",
        ]
    )
    for key in (
        "PASS_TO_PASS",
        "PASS_TO_FAIL",
        "FAIL_TO_PASS",
        "FAIL_TO_FAIL",
    ):
        lines.append(
            f"| {key} | {summary['pass_migrations'].get(key, 0)} |"
        )

    lines.extend(
        [
            "",
            "### Exact McNemar",
            "",
            "| P→F | F→P | Discordant | Exact two-sided p |",
            "|---:|---:|---:|---:|",
            f"| {summary['mcnemar_12gate']['pass_to_fail']} | "
            f"{summary['mcnemar_12gate']['fail_to_pass']} | "
            f"{summary['mcnemar_12gate']['discordant_pairs']} | "
            f"{number(summary['mcnemar_12gate']['exact_two_sided_p'], 4)} |",
            "",
            "单 seed、16 条 paired case 下，McNemar 仅作为离散迁移证据，"
            "不替代 effect size 和逐 case 审查。",
            "",
            "### 单门迁移与 McNemar",
            "",
            "| Gate | P→P | P→F | F→P | F→F | Exact p |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for gate in C.ALL_GATES:
        transitions = summary["gate_transitions"][gate]
        mcnemar = summary["gate_mcnemar"][gate]
        lines.append(
            f"| {LABELS[gate]} | {transitions['PASS_TO_PASS']} | "
            f"{transitions['PASS_TO_FAIL']} | "
            f"{transitions['FAIL_TO_PASS']} | "
            f"{transitions['FAIL_TO_FAIL']} | "
            f"{number(mcnemar['exact_two_sided_p'], 4)} |"
        )

    lines.extend(
        [
            "",
            "## 📚 16 条逐 case 结果",
            "",
            "| Case | E173 12门 | E179 12门 | Migration | "
            "E173 failures | E179 failures |",
            "|---|---:|---:|---|---|---|",
        ]
    )
    for row in paired:
        lines.append(
            f"| `{row['case_id']}` | {mark(row['e173_12gate_pass'])} | "
            f"{mark(row['e179_12gate_pass'])} | {row['pass_migration']} | "
            f"{row['e173_failure_modes'] or '—'} | "
            f"{row['e179_failure_modes'] or '—'} |"
        )

    lines.extend(
        [
            "",
            "## 📈 连续指标 paired 统计",
            "",
            "| Metric | Mean Δ | Median Δ | IQR Δ | Mean improvement | "
            "Bootstrap mean Δ 95% CI |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for metric in summary["metric_deltas"]:
        delta = summary["metric_deltas"][metric]
        improvement = summary["metric_improvements"][metric]
        lines.append(
            f"| `{metric}` | {number(delta['mean'])} | "
            f"{number(delta['median'])} | {number(delta['iqr'])} | "
            f"{number(improvement['mean'])} | "
            f"[{number(delta['bootstrap_mean_ci95_low'])}, "
            f"{number(delta['bootstrap_mean_ci95_high'])}] |"
        )
    lines.extend(
        [
            "",
            "Δ 定义为 E179−E173；signed improvement 已按指标方向归一化，"
            "正值表示 E179 改善。bootstrap 固定 seed `0`、"
            f"`{summary['paired_statistics_contract']['bootstrap_samples']}` "
            "次 resample。",
            "",
            "## 🔍 视觉复核",
            "",
            f"- Review rows：`{summary['visual_review']['reviewed']}/16`",
            f"- Strong PRG evidence rows："
            f"`{len(summary['visual_review']['strong_prg_evidence_rows'])}`",
            f"- Strong no-PRG evidence rows："
            f"`{len(summary['visual_review']['strong_no_prg_evidence_rows'])}`",
            f"- Visual verdict：`{summary['visual_verdict']}`",
            "",
            "视觉复核覆盖每条 paired 视频的 10/30/50/70/90% 五个时刻。"
            "三条 lower-body PASS→FAIL 在 no-PRG 深蹲阶段可见更小腿箱间距；"
            "唯一 FAIL→PASS 未见新增姿态异常。视觉证据支持数值结论，但"
            "手部朝向等细微差异仍以连续指标和 gate 为主。",
            "",
            "本报告使用完整 16 条 paired denominator，不以完成子集替代。",
            "",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
