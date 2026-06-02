#!/usr/bin/env python3
"""Summarize E109 fair-eval proxy metrics."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import as_float, read_csv, write_tsv


SUMMARY_FIELDS = [
    "method",
    "n_rows",
    "n_ready",
    "n_fall",
    "mean_pelvis_min_m",
    "mean_hand_near_frac_03cm",
    "mean_hand_near_frac_05cm",
    "mean_hand_near_frac_08cm",
    "mean_leg_object_interference_frac",
    "mean_deep_penetration_frac_2cm",
    "notes",
]


OBJECT_SUMMARY_FIELDS = [
    "object_key",
    "n_rows",
    "n_ready",
    "n_fall",
    "mean_pelvis_min_m",
    "mean_hand_near_frac_03cm",
    "mean_hand_near_frac_05cm",
    "mean_hand_near_frac_08cm",
    "mean_leg_object_interference_frac",
    "mean_deep_penetration_frac_2cm",
]


def mean(values: list[float]) -> str:
    vals = [v for v in values if math.isfinite(v)]
    return f"{sum(vals) / len(vals):.6f}" if vals else ""


def method_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row.get("method", "")].append(row)
    out: list[dict[str, Any]] = []
    for method, rs in sorted(groups.items()):
        out.append(
            {
                "method": method,
                "n_rows": len(rs),
                "n_ready": sum(str(r.get("fair_metric_ready", "")).lower() == "true" for r in rs),
                "n_fall": sum(str(r.get("fall_flag", "")).lower() == "true" for r in rs),
                "mean_pelvis_min_m": mean([as_float(r.get("pelvis_min_m")) for r in rs]),
                "mean_hand_near_frac_03cm": mean([as_float(r.get("hand_near_frac_03cm")) for r in rs]),
                "mean_hand_near_frac_05cm": mean([as_float(r.get("hand_near_frac_05cm")) for r in rs]),
                "mean_hand_near_frac_08cm": mean([as_float(r.get("hand_near_frac_08cm")) for r in rs]),
                "mean_leg_object_interference_frac": mean([as_float(r.get("leg_object_interference_frac")) for r in rs]),
                "mean_deep_penetration_frac_2cm": mean([as_float(r.get("deep_penetration_frac_2cm")) for r in rs]),
                "notes": "Do not use method-ref object/body tracking for fair win/loss.",
            }
        )
    return out


def object_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row.get("object_key", "") or "unknown"].append(row)
    out: list[dict[str, Any]] = []
    for object_key, rs in sorted(groups.items()):
        out.append(
            {
                "object_key": object_key,
                "n_rows": len(rs),
                "n_ready": sum(str(r.get("fair_metric_ready", "")).lower() == "true" for r in rs),
                "n_fall": sum(str(r.get("fall_flag", "")).lower() == "true" for r in rs),
                "mean_pelvis_min_m": mean([as_float(r.get("pelvis_min_m")) for r in rs]),
                "mean_hand_near_frac_03cm": mean([as_float(r.get("hand_near_frac_03cm")) for r in rs]),
                "mean_hand_near_frac_05cm": mean([as_float(r.get("hand_near_frac_05cm")) for r in rs]),
                "mean_hand_near_frac_08cm": mean([as_float(r.get("hand_near_frac_08cm")) for r in rs]),
                "mean_leg_object_interference_frac": mean([as_float(r.get("leg_object_interference_frac")) for r in rs]),
                "mean_deep_penetration_frac_2cm": mean([as_float(r.get("deep_penetration_frac_2cm")) for r in rs]),
            }
        )
    return out


def threshold_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for tag in ("03cm", "05cm", "08cm"):
        vals = [as_float(r.get(f"hand_near_frac_{tag}")) for r in rows]
        leg = [as_float(r.get(f"leg_near_frac_{tag}")) for r in rows]
        out.append(
            {
                "threshold_label": tag,
                "threshold_m": as_float(rows[0].get(f"threshold_m_{tag}")) if rows else "",
                "mean_hand_near_frac": mean(vals),
                "mean_leg_near_frac": mean(leg),
                "num_hand_values": sum(math.isfinite(v) for v in vals),
                "num_leg_values": sum(math.isfinite(v) for v in leg),
            }
        )
    return out


def write_markdown(
    path: Path,
    method_rows: list[dict[str, Any]],
    object_rows: list[dict[str, Any]],
    threshold_rows: list[dict[str, Any]],
    warnings_count: int,
) -> None:
    lines = [
        "# E109 Spider vs OmniRetarget 指标对比首版",
        "",
        "_自动生成；这份表的重点是把“哪些指标能比较、哪些指标不能比较”先分清楚。_",
        "",
        "## 先看结论",
        "",
        "- 现在还不能直接说 Spider 比 OmniRetarget 好多少，因为历史表里很多 GT 本身来自 OmniRetarget 或各自方法的 reference。",
        "- 当前先做第一步：把评测 case 整理到同一张 case bank；如果混入历史 method-reference 指标，就把它们单独标成不能判胜负。",
        "- 当前可以先看的指标是几何/物理 proxy：手离物体多近、有没有穿透、腿有没有碰物体、机器人有没有摔。",
        "- 这些 proxy 已按 `3cm / 5cm / 8cm` 三个尺度输出；凡是接触类指标，必须带着阈值一起读。",
        "- 最终要证明 Spider 相对 OmniRetarget 的增益，还需要给 OmniRetarget 也补同一套几何 proxy，并接入 Core4D raw mocap/object/contact GT。",
        "",
        "## 怎么读这份表",
        "",
        "- `diagnostic_only`：只做背景说明，不能判胜负。E026 历史行基本都在这里。",
        "- `fair_metric_ready=True`：这条结果有可读的几何/物理 proxy，但仍不是最终 raw-GT paper 指标。",
        "- `hand_near_frac_*` 越大表示手更接近物体，通常更好，但太大同时要看 penetration。",
        "- `leg_near_frac_*`、`leg_object_interference_frac`、`deep_penetration_frac_2cm` 越小越好。",
        f"- warnings rows: `{warnings_count}`，主要是提醒 self-ref/method-ref 或 raw-GT 尚未接入。",
        "",
        "## Method summary",
        "",
        "| method | N | ready | fall | pelvis min | hand 3cm | hand 5cm | hand 8cm | leg interference | deep pen 2cm |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in method_rows:
        lines.append(
            f"| `{row['method']}` | {row['n_rows']} | {row['n_ready']} | {row['n_fall']} | "
            f"{row['mean_pelvis_min_m']} | {row['mean_hand_near_frac_03cm']} | "
            f"{row['mean_hand_near_frac_05cm']} | {row['mean_hand_near_frac_08cm']} | "
            f"{row['mean_leg_object_interference_frac']} | {row['mean_deep_penetration_frac_2cm']} |"
        )
    lines.extend(
        [
            "",
            "## Object summary",
            "",
            "| object | N | ready | fall | pelvis min | hand 3cm | hand 5cm | hand 8cm | leg interference | deep pen 2cm |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in object_rows:
        lines.append(
            f"| `{row['object_key']}` | {row['n_rows']} | {row['n_ready']} | {row['n_fall']} | "
            f"{row['mean_pelvis_min_m']} | {row['mean_hand_near_frac_03cm']} | "
            f"{row['mean_hand_near_frac_05cm']} | {row['mean_hand_near_frac_08cm']} | "
            f"{row['mean_leg_object_interference_frac']} | {row['mean_deep_penetration_frac_2cm']} |"
        )
    lines.extend(
        [
            "",
            "## Threshold summary",
            "",
            "| threshold | threshold_m | hand near mean | leg near mean | hand N | leg N |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in threshold_rows:
        lines.append(
            f"| `{row['threshold_label']}` | {row['threshold_m']} | {row['mean_hand_near_frac']} | "
            f"{row['mean_leg_near_frac']} | {row['num_hand_values']} | {row['num_leg_values']} |"
        )
    lines.extend(
        [
            "",
            "## 下一步",
            "",
            "1. 接入 Core4D raw object pose，重算 raw object SE(3) error，替换 method-ref object tracking。",
            "2. 接入 S1 raw contact mask，输出 contact precision/recall/F1，而不是只看 near-contact fraction。",
            "3. 对 OmniRetarget kinematic 输出补同一套 geometry proxy，避免只比较 Spider CEM rows。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = read_csv(args.out_dir / "method_case_metrics.tsv", delimiter="\t")
    warnings = read_csv(args.out_dir / "warnings.tsv", delimiter="\t")
    method_rows = method_summary(rows)
    object_rows = object_summary(rows)
    threshold_rows = threshold_summary(rows)
    write_tsv(args.out_dir / "method_summary.tsv", method_rows, SUMMARY_FIELDS)
    write_tsv(args.out_dir / "object_summary.tsv", object_rows, OBJECT_SUMMARY_FIELDS)
    write_tsv(
        args.out_dir / "threshold_summary.tsv",
        threshold_rows,
        ["threshold_label", "threshold_m", "mean_hand_near_frac", "mean_leg_near_frac", "num_hand_values", "num_leg_values"],
    )
    write_markdown(args.out_dir / "summary.md", method_rows, object_rows, threshold_rows, len(warnings))
    print(f"[summarize_fair_eval] wrote {args.out_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
