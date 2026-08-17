#!/usr/bin/env python3
"""E201 · funnel workbook — per-rollout 14-gate wide/narrow + layer, plus sizing.

Reads the classifier output (`<exp>_funnel_rollout.tsv`, which already carries the
recomputed body_z + layer + family flag) and joins the underlying numeric values
from the source case_metrics.tsv. Emits:

  * detail sheet — per rollout: id block; for each of the 14 gates its numeric
    value + narrow PASS/FAIL + wide PASS/FAIL; then layer + family_flag.
  * summary sheet — layer counts (ALL / aug / orig / per-object) + auto-decided %.

Usage:
    .venv/bin/python workspace/core4d/scripts/eval/reports/gen_E201_funnel_xlsx.py --exp E199
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/experiments/E201"))
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/eval/reports"))
import funnel_config as FC  # noqa: E402
import gen_E199_fullscale_gate_xlsx as X  # noqa: E402

EXPS = {"E199": X.CASE_METRICS}
FUNNEL_DIR = REPO / "workspace/core4d/results/E201/funnel"

GREEN = PatternFill("solid", fgColor="C6EFCE")
RED = PatternFill("solid", fgColor="FFC7CE")
GREY = PatternFill("solid", fgColor="D9D9D9")
NAVY = PatternFill("solid", fgColor="1F3864")
HEAD_FONT = Font(bold=True, color="FFFFFF")
LAYER_FILL = {
    "L1_reject": PatternFill("solid", fgColor="C00000"),
    "L2_review": PatternFill("solid", fgColor="FFD966"),
    "L3_auto": PatternFill("solid", fgColor="70AD47"),
    "L3_review": PatternFill("solid", fgColor="F4B183"),
}
LAYERS = ["L1_reject", "L2_review", "L3_auto", "L3_review"]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _cell(ws, r, c, value, fill=None, font=None):
    cell = ws.cell(row=r, column=c, value=value)
    cell.alignment = Alignment(horizontal="center", vertical="center")
    if fill is not None:
        cell.fill = fill
    if font is not None:
        cell.font = font
    return cell


def _num(ws, r, c, value):
    v = X.finite(value)
    _cell(ws, r, c, round(v, 4) if math.isfinite(v) else "")


def _passfill(ok: bool) -> PatternFill:
    return GREEN if ok else RED


# gate value fields for display (hard gates first, then banded, matching FunnelConfig)
HARD_FIELDS = [(n, f) for n, f, *_ in FC.HARD_GATES]
BANDED = FC.BANDED_GATES


def write_detail(ws, rows: list[dict[str, Any]]) -> None:
    header = ["object_key", "case_id", "group", "aug_variant", "layer", "family_flag"]
    # hard gates: value + pass
    for name, _f in HARD_FIELDS:
        header += [f"{name}", f"{name}✓"]
    # banded gates: value + narrow + wide
    for name, _f, _op, narrow, wide in BANDED:
        header += [f"{name}", f"{name} N({narrow:g})", f"{name} W({wide:g})"]

    for c, name in enumerate(header, start=1):
        cell = _cell(ws, 1, c, name, NAVY, HEAD_FONT)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.freeze_panes = "G2"

    for r, row in enumerate(rows, start=2):
        _cell(ws, r, 1, row.get("object_key", ""))
        _cell(ws, r, 2, row.get("case_id", ""))
        _cell(ws, r, 3, row.get("group", ""))
        _cell(ws, r, 4, row.get("aug_variant", ""))
        _cell(ws, r, 5, row.get("layer", ""), LAYER_FILL.get(row.get("layer", "")), HEAD_FONT)
        _cell(ws, r, 6, row.get("family_flag", ""))
        c = 7
        hard_failed = set((row.get("hard_failed") or "").split(","))
        for name, field in HARD_FIELDS:
            if field == "fall_flag":
                _cell(ws, r, c, str(row.get("fall_flag", row.get("_fall", ""))))
            elif name == "body_z":
                _num(ws, r, c, row.get("body_z_err_p95_m"))
            else:
                _num(ws, r, c, row.get(field))
            c += 1
            ok = name not in hard_failed
            _cell(ws, r, c, "PASS" if ok else "FAIL", _passfill(ok))
            c += 1
        wide_failed = set((row.get("wide_failed") or "").split(","))
        narrow_failed = set((row.get("narrow_failed") or "").split(","))
        for name, field, op, narrow, wide in BANDED:
            _num(ws, r, c, row.get(field)); c += 1
            n_ok = name not in narrow_failed
            _cell(ws, r, c, "PASS" if n_ok else "FAIL", _passfill(n_ok)); c += 1
            w_ok = name not in wide_failed
            _cell(ws, r, c, "PASS" if w_ok else "FAIL", _passfill(w_ok)); c += 1

    widths = [10, 34, 8, 10, 11, 20] + [11, 6] * len(HARD_FIELDS) + [11, 9, 9] * len(BANDED)
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w


def write_summary(ws, rows: list[dict[str, Any]]) -> None:
    r0 = 1

    def block(title: str, subset: list[dict[str, Any]], top: int) -> int:
        n = len(subset)
        counts = {lay: sum(1 for x in subset if x["layer"] == lay) for lay in LAYERS}
        auto = counts["L1_reject"] + counts["L3_auto"]
        human = counts["L2_review"] + counts["L3_review"]
        _cell(ws, top, 1, title, font=Font(bold=True))
        _cell(ws, top, 2, f"total={n}")
        row = top + 1
        for lay in LAYERS:
            _cell(ws, row, 1, lay, LAYER_FILL.get(lay), HEAD_FONT)
            _cell(ws, row, 2, counts[lay])
            _cell(ws, row, 3, round(counts[lay] / n, 3) if n else "")
            row += 1
        _cell(ws, row, 1, "auto-decided (L1+L3auto)", GREEN)
        _cell(ws, row, 2, f"{auto}/{n}")
        _cell(ws, row, 3, round(auto / n, 3) if n else "")
        row += 1
        _cell(ws, row, 1, "human (L2+L3review)", PatternFill("solid", fgColor="FFD966"))
        _cell(ws, row, 2, f"{human}/{n}")
        _cell(ws, row, 3, round(human / n, 3) if n else "")
        return row + 2

    r0 = block("ALL", rows, r0)
    r0 = block("aug", [r for r in rows if r["group"] == "aug"], r0)
    r0 = block("orig", [r for r in rows if r["group"] == "orig"], r0)
    for obj in sorted({r["object_key"] for r in rows}):
        r0 = block(obj, [r for r in rows if r["object_key"] == obj], r0)
    ws.column_dimensions["A"].width = 28
    ws.column_dimensions["B"].width = 12
    ws.column_dimensions["C"].width = 10


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="E199", choices=sorted(EXPS))
    ap.add_argument("--rollout", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rollout_tsv = args.rollout or (FUNNEL_DIR / f"{args.exp}_funnel_rollout.tsv")
    layer_rows = read_tsv(rollout_tsv)
    # join numeric values from source case_metrics
    src = {(m["case_id"], m.get("aug_variant", "")): m for m in read_tsv(EXPS[args.exp])}
    merged: list[dict[str, Any]] = []
    for lr in layer_rows:
        key = (lr["case_id"], lr.get("aug_variant", ""))
        row = dict(src.get(key, {}))
        row.update(lr)  # layer fields + body_z override win
        merged.append(row)

    wb = Workbook()
    write_detail(wb.active, merged)
    wb.active.title = "detail"
    write_summary(wb.create_sheet("summary"), merged)

    out = args.out or (FUNNEL_DIR / f"E201_{args.exp}_funnel.xlsx")
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    print(f"[done] wrote {out} ({len(merged)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
