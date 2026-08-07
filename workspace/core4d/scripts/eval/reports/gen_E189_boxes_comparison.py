#!/usr/bin/env python3
"""Generate the E189 vs E172/E173 43-row, per-object 12-gate report.

Produces Markdown + XLSX. TSV/JSON are already written by the eval runner
(``eval_E189_boxes_e167a_vs_prg.py``); this script only synthesizes the
human-facing report on top of them and must not recompute any metric.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

SCRIPT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPT_ROOT / "experiments/E189"))

import e189_common as C  # noqa: E402

OBJECTS = ("box004", "box024", "box001")

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

# Volume-vs-conclusion context table (plan/215 六物体尺寸对照).
# box021/box026 never got a no-PRG ablation (only box023 via E179, and
# box004/box024/box001 via E189); their column is left explicitly N/A
# rather than inferring a verdict from data that does not exist.
VOLUME_TABLE = [
    {"object": "box004", "volume_m3": 0.041, "source": "E189", "note": ""},
    {"object": "box023", "volume_m3": 0.036, "source": "E179 (引用不重跑)", "note": ""},
    {"object": "box021", "volume_m3": 0.059, "source": "N/A", "note": "未做no-PRG消融"},
    {"object": "box026", "volume_m3": 0.116, "source": "N/A", "note": "未做no-PRG消融"},
    {"object": "box024", "volume_m3": 0.253, "source": "E189", "note": ""},
    {"object": "box001", "volume_m3": 0.256, "source": "E189", "note": ""},
]
E179_BOX023_VERDICT = "PRG_BETTER"
E179_BOX023_PASS = "7/16 (E173 PRG) -> 4/16 (E179 no-PRG)"


def mark(value: Any) -> str:
    return "PASS" if C.boolish(value) else "FAIL"


def number(value: Any, digits: int = 3) -> str:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{output:.{digits}f}" if math.isfinite(output) else "NA"


def object_section(obj: str, block: dict[str, Any], paired: list[dict[str, str]]) -> list[str]:
    n = block["n"]
    lines = [
        f"## 📊 {obj}（n={n}）",
        "",
        "| 口径 | PRG (E172/E173) | E189 no-PRG |",
        "|---|---:|---:|",
        f"| Physics 6 门 | {block['prg_physics6_pass']}/{n} | "
        f"{block['e189_physics6_pass']}/{n} |",
        f"| Physics + tracking 12 门 | {block['prg_12gate_pass']}/{n} | "
        f"{block['e189_12gate_pass']}/{n} |",
        "",
        f"**结论：`{block['verdict']}`**"
        + ("（新增 fall case）" if block["new_fall_case"] else ""),
        "",
        "### 十二门逐门通过数",
        "",
        "| Gate | PRG | E189 | Delta |",
        "|---|---:|---:|---:|",
    ]
    for gate in C.ALL_GATES:
        counts = block["gate_pass_counts"][gate]
        lines.append(
            f"| {LABELS[gate]} | {counts['prg']}/{n} | {counts['e189']}/{n} | "
            f"{counts['e189']-counts['prg']:+d} |"
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
    for key in ("PASS_TO_PASS", "PASS_TO_FAIL", "FAIL_TO_PASS", "FAIL_TO_FAIL"):
        lines.append(f"| {key} | {block['pass_migrations'].get(key, 0)} |")
    mcnemar = block["mcnemar_12gate"]
    lines.extend(
        [
            "",
            "### Exact McNemar（单 seed，仅供参考，不单独作为判据）",
            "",
            "| P→F | F→P | Discordant | Exact two-sided p |",
            "|---:|---:|---:|---:|",
            f"| {mcnemar['pass_to_fail']} | {mcnemar['fail_to_pass']} | "
            f"{mcnemar['discordant_pairs']} | "
            f"{number(mcnemar['exact_two_sided_p'], 4)} |",
            "",
            "### 连续指标 paired 统计",
            "",
            "| Metric | Mean Δ | Median Δ | IQR Δ | Mean improvement | "
            "Bootstrap mean Δ 95% CI |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for metric, delta in block["metric_deltas"].items():
        improvement = block["metric_improvements"][metric]
        lines.append(
            f"| `{metric}` | {number(delta['mean'])} | "
            f"{number(delta['median'])} | {number(delta['iqr'])} | "
            f"{number(improvement['mean'])} | "
            f"[{number(delta['bootstrap_mean_ci95_low'])}, "
            f"{number(delta['bootstrap_mean_ci95_high'])}] |"
        )
    rows = [row for row in paired if row["object_key"] == obj]
    lines.extend(
        [
            "",
            f"### {obj} 逐 case 结果",
            "",
            "| Case | PRG 12门 | E189 12门 | Migration | PRG failures | "
            "E189 failures |",
            "|---|---:|---:|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| `{row['case_id']}` | {mark(row['prg_12gate_pass'])} | "
            f"{mark(row['e189_12gate_pass'])} | {row['pass_migration']} | "
            f"{row['prg_failure_modes'] or '—'} | "
            f"{row['e189_failure_modes'] or '—'} |"
        )
    lines.append("")
    return lines


# gate -> underlying raw metric column name (shared by baseline tsv and
# e189_case_metrics.tsv). This is the one gate ("fall") that the eval
# runner's PAIR_METRICS skips (it has no continuous metric of its own,
# just the boolean fall_flag), so it is looked up separately here instead
# of via the paired.tsv delta_/prg_/e189_ columns.
GATE_METRIC_MAP = {
    "fall": "fall_flag",
    "body_z": "body_z_err_p95_m",
    "contact": "hand_object_physics_contact_in_mask_frac",
    "release": "hand_object_release_false_contact_3mm_frac",
    "hand_penetration": "hand_object_physics_penetration_3mm_frame_frac",
    "lower_body": "leg_penetration_frac",
    "root_pos": "track_root_pos_err_cm_mean",
    "root_ori": "track_root_ori_err_deg_mean",
    "hand_pos": "track_eef_pos_err_cm_mean",
    "hand_ori": "track_eef_ori_err_deg_mean",
    "object_pos": "track_obj_pos_err_cm_mean",
    "object_ori": "track_obj_ori_err_deg_mean",
}


def gate_metric_value(gate: str, row: dict[str, Any], side: str) -> float:
    """Raw metric value for one gate/side ('prg' or 'e189') off a paired row.

    ``fall`` has no PAIR_METRICS delta column, so it is not present on the
    paired row; callers must pre-populate ``{gate}_{side}_raw`` for it from
    the baseline/case_metrics tables before calling this.
    """
    if gate == "fall":
        return row[f"fall_{side}_raw"]
    metric = GATE_METRIC_MAP[gate]
    return C.finite(row[f"{side}_{metric}"], math.nan)


def build_gate_rows(
    obj: str,
    paired: list[dict[str, str]],
    baseline_by_case: dict[str, dict[str, str]],
    case_metrics_by_case: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    rows = []
    for row in paired:
        if row["object_key"] != obj:
            continue
        case_id = row["case_id"]
        enriched = dict(row)
        enriched["fall_prg_raw"] = int(
            C.boolish(baseline_by_case[case_id].get("fall_flag"))
        )
        enriched["fall_e189_raw"] = int(
            C.boolish(case_metrics_by_case[case_id].get("fall_flag"))
        )
        out: dict[str, Any] = {
            "case_id": case_id,
            "prg_pass": C.boolish(row["prg_12gate_pass"]),
            "e189_pass": C.boolish(row["e189_12gate_pass"]),
            "prg_failure_modes": row["prg_failure_modes"] or "-",
            "e189_failure_modes": row["e189_failure_modes"] or "-",
        }
        for gate in C.ALL_GATES:
            prg_value = gate_metric_value(gate, enriched, "prg")
            e189_value = gate_metric_value(gate, enriched, "e189")
            delta = (
                prg_value - e189_value
                if math.isfinite(prg_value) and math.isfinite(e189_value)
                else math.nan
            )
            round_digits = 0 if gate == "fall" else 4
            out[f"{gate}_prg"] = round(prg_value, round_digits) if math.isfinite(prg_value) else "NA"
            out[f"{gate}_e189"] = round(e189_value, round_digits) if math.isfinite(e189_value) else "NA"
            out[f"{gate}_delta"] = round(delta, round_digits) if math.isfinite(delta) else "NA"
        rows.append(out)
    return rows


def write_xlsx(
    output: Path,
    summary: dict[str, Any],
    paired: list[dict[str, str]],
    baseline: list[dict[str, str]],
    case_metrics: list[dict[str, str]],
) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor="2563EB")  # case/pass/failure cols
    delta_header_fill = PatternFill("solid", fgColor="EA580C")  # delta cols: distinct orange
    fail_fill = PatternFill("solid", fgColor="FEE2E2")
    pass_fill = PatternFill("solid", fgColor="DCFCE7")

    baseline_by_case = {row["case_id"]: row for row in baseline}
    case_metrics_by_case = {row["case_id"]: row for row in case_metrics}

    wb = Workbook()
    ws_summary = wb.active
    ws_summary.title = "Summary"
    ws_summary.append(["object_key", "n", "prg_12gate_pass", "e189_12gate_pass", "verdict"])
    for col_idx in range(1, 6):
        cell = ws_summary.cell(row=1, column=col_idx)
        cell.font = header_font
        cell.fill = header_fill
    for obj in OBJECTS:
        block = summary["by_object"][obj]
        ws_summary.append(
            [obj, block["n"], block["prg_12gate_pass"], block["e189_12gate_pass"], block["verdict"]]
        )
    ws_summary.append([])
    ws_summary.append(["六物体体积-结论关系表（PRG 是否随体积退化）"])
    ws_summary.append(["object", "volume_m3", "source", "verdict/note"])
    for entry in VOLUME_TABLE:
        verdict = (
            summary["by_object"][entry["object"]]["verdict"]
            if entry["object"] in summary.get("by_object", {})
            else (E179_BOX023_VERDICT if entry["object"] == "box023" else entry["note"])
        )
        ws_summary.append([entry["object"], entry["volume_m3"], entry["source"], verdict])
    for col_idx in range(1, 5):
        ws_summary.column_dimensions[get_column_letter(col_idx)].width = 22

    fixed_cols = ("case_id", "prg_pass", "e189_pass", "prg_failure_modes", "e189_failure_modes")
    for obj in OBJECTS:
        ws = wb.create_sheet(obj)
        rows = build_gate_rows(obj, paired, baseline_by_case, case_metrics_by_case)
        fields = list(fixed_cols)
        for gate in C.ALL_GATES:
            fields.extend([f"{gate}_prg", f"{gate}_e189", f"{gate}_delta"])

        # Two header rows: gate name spanning its 3 columns, then the
        # prg/e189/delta sub-header underneath (delta gets the warning color).
        ws.append(
            [""] * len(fixed_cols)
            + [g for gate in C.ALL_GATES for g in (LABELS[gate], "", "")]
        )
        ws.append(list(fixed_cols) + [c.rsplit("_", 1)[-1] for c in fields[len(fixed_cols):]])
        for col_idx in range(1, len(fixed_cols) + 1):
            for r in (1, 2):
                cell = ws.cell(row=r, column=col_idx)
                cell.font = header_font
                cell.fill = header_fill
        for gate_idx, gate in enumerate(C.ALL_GATES):
            base_col = len(fixed_cols) + gate_idx * 3 + 1
            ws.merge_cells(start_row=1, start_column=base_col, end_row=1, end_column=base_col + 2)
            top_cell = ws.cell(row=1, column=base_col)
            top_cell.font = header_font
            top_cell.fill = header_fill
            top_cell.alignment = Alignment(horizontal="center")
            for offset, is_delta in enumerate((False, False, True)):
                cell = ws.cell(row=2, column=base_col + offset)
                cell.font = header_font
                cell.fill = delta_header_fill if is_delta else header_fill

        for row in rows:
            ws.append([row.get(f, "") for f in fields])
        for r_idx, row in enumerate(rows, start=3):
            for bool_col in ("prg_pass", "e189_pass"):
                col_idx = fields.index(bool_col) + 1
                cell = ws.cell(row=r_idx, column=col_idx)
                cell.fill = pass_fill if row[bool_col] else fail_fill

        ws.column_dimensions["A"].width = 30
        for col_idx in range(2, len(fixed_cols) + 1):
            ws.column_dimensions[get_column_letter(col_idx)].width = 20
        for col_idx in range(len(fixed_cols) + 1, len(fields) + 1):
            ws.column_dimensions[get_column_letter(col_idx)].width = 11
        ws.freeze_panes = "F3"

    output.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-dir", type=Path, default=C.RESULTS / "s6_downstream/eval/full"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--xlsx-output", type=Path)
    args = parser.parse_args()

    summary = json.loads(
        (args.eval_dir / "e189_eval_summary.json").read_text(encoding="utf-8")
    )
    paired = C.read_tsv(args.eval_dir / "e189_vs_prg_paired.tsv")
    gate_matrix = C.read_tsv(args.eval_dir / "e189_vs_prg_gate_matrix.tsv")
    case_metrics = C.read_tsv(args.eval_dir / "e189_case_metrics.tsv")
    if len(paired) != C.EXPECTED_PAIRED_ROWS:
        raise ValueError(
            f"report requires {C.EXPECTED_PAIRED_ROWS} paired rows, "
            f"got {len(paired)}"
        )
    if len(gate_matrix) != C.EXPECTED_PAIRED_ROWS * len(C.ALL_GATES):
        raise ValueError(
            f"report requires {C.EXPECTED_PAIRED_ROWS * len(C.ALL_GATES)} "
            f"gate cells, got {len(gate_matrix)}"
        )
    if "by_object" not in summary or set(summary["by_object"]) != set(OBJECTS):
        raise ValueError("summary missing per-object blocks; rerun eval in full mode")

    output = args.output or args.eval_dir / "E189_vs_PRG_boxes_report.md"
    xlsx_output = (
        args.xlsx_output or args.eval_dir / "E189_vs_PRG_boxes_report.xlsx"
    )

    lines = [
        "# E189 vs E172/E173：box004/box024/box001 43 条配对对比（no-PRG vs PRG）",
        "",
        "_Core4D Phase 52 · 按物体独立判定，禁止合并成单一跨物体结论_",
        "",
        "---",
        "",
        f"- Scoring contract：`{summary['scoring_contract_id']}`",
        f"- Case rows：`{summary['evaluated_rows']}/{C.EXPECTED_PAIRED_ROWS}`",
        f"- Explicit gate cells：`{summary['gate_cells']}/"
        f"{C.EXPECTED_PAIRED_ROWS * len(C.ALL_GATES)}`",
        "- Physics：fall、body-z、contact、release、hand penetration、lower-body",
        "- Tracking：root pos/ori、EEF pos/ori、object pos/ori",
        "- 缺失或非有限指标一律 FAIL。box023 不在本轮范围内，结论引用 E179 "
        f"（`{E179_BOX023_VERDICT}`，{E179_BOX023_PASS}），不重新计算。",
        "",
        "## 🧭 三物体结论总览",
        "",
        "| 物体 | n | PRG 12门 | E189 12门 | 结论 |",
        "|---|---:|---:|---:|---|",
    ]
    for obj in OBJECTS:
        block = summary["by_object"][obj]
        lines.append(
            f"| {obj} | {block['n']} | {block['prg_12gate_pass']}/{block['n']} | "
            f"{block['e189_12gate_pass']}/{block['n']} | `{block['verdict']}` |"
        )
    lines.extend(
        [
            "",
            "## 📐 六物体体积-结论关系表",
            "",
            "| 物体 | 体积(m³) | 数据来源 | 结论/备注 |",
            "|---|---:|---|---|",
        ]
    )
    for entry in VOLUME_TABLE:
        if entry["object"] in summary["by_object"]:
            verdict = summary["by_object"][entry["object"]]["verdict"]
        elif entry["object"] == "box023":
            verdict = E179_BOX023_VERDICT
        else:
            verdict = entry["note"]
        lines.append(
            f"| {entry['object']} | {entry['volume_m3']} | {entry['source']} | "
            f"`{verdict}` |"
        )
    lines.extend(
        [
            "",
            "合并 43 条的单一数字仅作完整性校验，不作为结论依据（见 "
            "`combined_43_row_secondary_only`）：PRG "
            f"{summary['combined_43_row_secondary_only']['prg_12gate_pass']}/"
            f"{C.EXPECTED_PAIRED_ROWS} → E189 "
            f"{summary['combined_43_row_secondary_only']['e189_12gate_pass']}/"
            f"{C.EXPECTED_PAIRED_ROWS}。",
            "",
        ]
    )
    for obj in OBJECTS:
        lines.extend(object_section(obj, summary["by_object"][obj], paired))

    lines.extend(
        [
            "## 🔍 视觉复核",
            "",
            f"- Visual verdict：`{summary.get('visual_verdict', 'PENDING')}`",
            "- 待 43 条 Full MP4 + paired 对照视频渲染完成后，用 "
            "`/video-frames` 对 numeric fail/pass 迁移与边界 case 提取关键帧，"
            "结果回填本节与结果 log。",
            "",
            "本报告使用完整 43 条 paired denominator，不以完成子集替代；"
            "结论文本以按物体表为准。",
            "",
        ]
    )
    baseline = C.read_tsv(C.RESULTS / "input_authority/e172_e173_baseline_12gate.tsv")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    write_xlsx(xlsx_output, summary, paired, baseline, case_metrics)
    print(output)
    print(xlsx_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
