#!/usr/bin/env python3
"""Evaluate E107 Box021 selected-4 clean ref-FK full CEM rollouts."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


THIS = Path(__file__).resolve()
REPO = THIS.parents[4]
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/E107"))
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/E105"))

from e107_common import RESULTS_ROOT, TASK_ROOT, VARIANTS_TSV, read_variants, rel  # noqa: E402


E090_EVAL = REPO / "workspace/core4d/scripts/eval/eval_E090.py"
E105_EVAL = REPO / "workspace/core4d/scripts/eval/eval_E105_box026_clean_cem.py"


def repo_path_from_env(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    path = Path(raw) if raw else default
    return path if path.is_absolute() else REPO / path


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expected_outputs(stage: str, variants: list[dict[str, str]], results_dir: Path) -> list[Path]:
    out = []
    for row in variants:
        out.append(results_dir / f"{row['variant']}.npz")
        out.append(results_dir / f"{row['variant']}_{stage}.mp4")
        out.append(results_dir / f"{row['variant']}_outdir_{stage}/trajectory_mjwp_act.npz")
    return out


def selected_rows(variants: list[dict[str, str]], wanted: list[str]) -> list[dict[str, str]]:
    if not wanted:
        return variants
    want = set(wanted)
    return [row for row in variants if row["variant"] in want]


def fail_if_missing_outputs(stage: str, rows: list[dict[str, str]], results_dir: Path) -> None:
    missing = [path for path in expected_outputs(stage, rows, results_dir) if not path.is_file()]
    if missing:
        print("E107 eval is deferred until all selected CEM outputs exist. Missing:")
        for path in missing[:80]:
            print(f"- {rel(path)}")
        if len(missing) > 80:
            print(f"- ... {len(missing) - 80} more")
        raise SystemExit(1)


def review_status(variant: str) -> str:
    review = RESULTS_ROOT / "pre_cem_visual_review" / variant / "REVIEW.md"
    if not review.is_file():
        return "MISSING"
    for line in review.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("Status:"):
            return line.split(":", 1)[1].strip()
    return "UNKNOWN"


def e107_replay_metrics(eval_e105, npz: Path, scene: Path) -> dict[str, Any]:
    """E107 replay gate policy.

    Keep pelvis tilt as a diagnostic number, but do not use it as a pass/fail
    gate for E107 selected-4. The active replay gate only blocks pelvis-low
    and lie-on-box cases.
    """
    metrics = eval_e105.replay_metrics(npz, scene)
    metrics["gate_pelvis_tilt_diagnostic"] = bool(metrics.get("gate_pelvis_tilt", False))
    metrics["gate_pelvis_tilt"] = False
    replay_fail = bool(metrics.get("gate_pelvis_low", False)) or bool(
        metrics.get("gate_lie_on_box", False)
    )
    metrics["replay_gate_fail"] = replay_fail
    metrics["replay_gate_pass"] = not replay_fail
    metrics["replay_gate_policy"] = "E107_no_tilt_gate_low_or_lie_only"
    return metrics


def e107_status(stage: str, metrics: dict[str, Any]) -> tuple[str, bool]:
    """E107 状态判定口径。

    不沿用 E105 batch status 里的时长与 pelvis tilt 失败项。时长是数据属性，
    pelvis tilt 在本轮 selected-4 audit 中只作为诊断数值。
    """
    obj_mean = float(metrics["obj_err_mean_m"])
    obj_max = float(metrics["obj_err_max_m"])
    pelvis = float(metrics["pelvis_min_m"])
    head = float(metrics["head_pen_frac"])
    upper = float(metrics["upper_pen_frac"])
    lh_floor = float(metrics["handL_floor_lt_5cm_frac"])
    rh_floor = float(metrics["handR_floor_lt_5cm_frac"])
    contact = float(metrics["contact_frac_either"])
    replay_gate_pass = bool(metrics.get("replay_gate_pass", True))
    if stage == "smoke":
        passed = (
            obj_mean <= 0.15
            and pelvis >= 0.45
            and max(head, upper, lh_floor, rh_floor) <= 0.10
            and replay_gate_pass
        )
        return ("PASS", True) if passed else ("FAIL", False)
    work = (
        obj_mean <= 0.10
        and obj_max <= 0.30
        and pelvis >= 0.55
        and max(head, upper, lh_floor, rh_floor) <= 0.05
        and contact >= 0.30
        and replay_gate_pass
    )
    if work:
        return "WORK", True
    review = (
        obj_mean <= 0.12
        and obj_max <= 0.35
        and pelvis >= 0.45
        and max(head, upper, lh_floor, rh_floor) <= 0.08
        and replay_gate_pass
    )
    return ("REVIEW+", False) if review else ("FAIL", False)


def evaluate_rollouts(stage: str, rows: list[dict[str, str]], results_dir: Path) -> dict[str, dict[str, Any]]:
    eval_e090 = load_module("eval_E090_for_E107", E090_EVAL)
    eval_e105 = load_module("eval_E105_for_E107", E105_EVAL)
    results: dict[str, dict[str, Any]] = {}
    for row in rows:
        variant = row["variant"]
        npz = results_dir / f"{variant}_outdir_{stage}/trajectory_mjwp_act.npz"
        scene = TASK_ROOT / row["derived_task"] / "scene_act.xml"
        metrics = eval_e090.compute_sim_metrics(npz, scene)
        metrics.update(e107_replay_metrics(eval_e105, npz, scene))
        metrics.update(eval_e105.leg_object_metrics(variant, npz, scene, results_dir))
        work_status, stage_pass = e107_status(stage, metrics)
        lowerbody_pass = bool(metrics["lowerbody_strict_pass"])
        strict_work_status = "WORK" if work_status == "WORK" and lowerbody_pass else "FAIL"
        metrics.update(
            {
                "variant": variant,
                "ordinal": row.get("ordinal", ""),
                "source_task": row["source_task"],
                "derived_task": row["derived_task"],
                "split": row.get("split", ""),
                "selected_id": row.get("selected_id", ""),
                "route": "ref_fk_clean",
                "stage": stage,
                "npz_path": rel(npz),
                "root_npz_path": rel(results_dir / f"{variant}.npz"),
                "video_path": rel(results_dir / f"{variant}_{stage}.mp4"),
                "scene_xml": rel(scene),
                "pre_cem_review_status": review_status(variant),
                "work_status": work_status,
                "work_status_lowerbody_strict": strict_work_status,
                "stage_pass": stage_pass,
                "advance_to_rl": bool(stage == "full" and work_status == "WORK"),
                "advance_to_rl_lowerbody_strict": bool(stage == "full" and strict_work_status == "WORK"),
            }
        )
        results[variant] = metrics
    return results


def write_eval_outputs(stage: str, results: dict[str, dict[str, Any]], results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    out_json = results_dir / f"{stage}_eval_summary.json"
    out_json.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    keys = sorted({key for row in results.values() for key in row})
    out_csv = results_dir / f"{stage}_eval_summary.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results.values())

    out_md = results_dir / f"{stage}_eval_summary.md"
    lines = [
        f"# E107 Box021 selected-4 clean ref-FK {stage} 汇总",
        "",
        f"- CEM 变体数：`{len(results)}`",
        f"- variants 文件：`{rel(VARIANTS_TSV)}`",
        "",
        "状态判定复用 E105 的上半身安全阈值，并补充 E026/E081 的腿部 strict proxy。E107 当前口径不把源序列时长和 pelvis tilt 当失败 gate；replay gate 只拦截 pelvis-low 或 lie-on-box。",
        "",
        "| 变体 | 分配 | T | 接触率 | 物体均值误差 | 物体最大误差 | pelvis 最低 | replay | 上半身状态 | 腿部干涉 | lower strict | RL strict |",
        "|---|---|---:|---:|---:|---:|---:|---|---|---:|---|---|",
    ]
    for variant, row in results.items():
        lines.append(
            f"| `{variant}` | `{row['split']}` | {row['T']} | "
            f"{float(row['contact_frac_either']) * 100:.1f}% | "
            f"{float(row['obj_err_mean_m']):.3f}m | "
            f"{float(row['obj_err_max_m']):.3f}m | "
            f"{float(row['pelvis_min_m']):.3f}m | "
            f"{'PASS' if row['replay_gate_pass'] else 'FAIL'} | "
            f"{row['work_status']} | "
            f"{float(row['leg_box_interference_frac']) * 100:.1f}% | "
            f"{'PASS' if row['lowerbody_strict_pass'] else 'FAIL'} | "
            f"{'YES' if row['advance_to_rl_lowerbody_strict'] else 'NO'} |"
        )

    lines.extend(
        [
            "",
            "上半身安全 gate 沿用 E105 口径，包含 head/upper penetration 和手-地面近接触比例。",
            "",
            "| 变体 | 头部穿箱 | 上半身穿箱 | 左手距地<5cm | 右手距地<5cm | 接触率 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for variant, row in results.items():
        lines.append(
            f"| `{variant}` | {float(row['head_pen_frac']) * 100:.1f}% | "
            f"{float(row['upper_pen_frac']) * 100:.1f}% | "
            f"{float(row['handL_floor_lt_5cm_frac']) * 100:.1f}% | "
            f"{float(row['handR_floor_lt_5cm_frac']) * 100:.1f}% | "
            f"{float(row['contact_frac_either']) * 100:.1f}% |"
        )

    lines.extend(
        [
            "",
            "E107 replay gate 阈值：`pelvis_end_z>=0.55m` 且 `lie_on_box_frac<0.30`。Pelvis tilt 只作为诊断数值报告，不参与 pass/fail。",
            "",
            "| 变体 | pelvis 末段高度 | tilt 末段角度（仅诊断） | 趴箱比例 | replay 失败标记 |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for variant, row in results.items():
        flags = ",".join(
            name
            for name in ["gate_pelvis_low", "gate_lie_on_box"]
            if bool(row.get(name))
        )
        tilt_note = " diag_fail" if bool(row.get("gate_pelvis_tilt_diagnostic")) else ""
        lines.append(
            f"| `{variant}` | {float(row['pelvis_end_z_m']):.3f}m | "
            f"{float(row['pelvis_tilt_end_deg']):.1f}deg{tilt_note} | "
            f"{float(row['lie_on_box_frac']) * 100:.1f}% | {flags or 'none'} |"
        )

    lines.extend(
        [
            "",
            "下半身 strict proxy 沿用 E026/E081：`leg_box_interference_frac <= 5%`。",
            "",
            "| 变体 | 腿部干涉 | 腿部接触 | 最小 leg SDF | 最差 geom | 物体-地面接触 |",
            "|---|---:|---:|---:|---|---:|",
        ]
    )
    for variant, row in results.items():
        lines.append(
            f"| `{variant}` | {float(row['leg_box_interference_frac']) * 100:.1f}% | "
            f"{float(row['leg_object_contact_frac']) * 100:.1f}% | "
            f"{float(row['leg_box_sdf_min_m']):.3f}m | "
            f"`{row['leg_box_sdf_argmin']}` | "
            f"{float(row['object_floor_contact_frac']) * 100:.1f}% |"
        )

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {rel(out_json)}")
    print(f"wrote {rel(out_csv)}")
    print(f"wrote {rel(out_md)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="full", choices=["smoke", "full"])
    parser.add_argument("variants", nargs="*")
    args = parser.parse_args()

    rows = selected_rows(read_variants(), args.variants)
    results_dir = repo_path_from_env("RESULTS", RESULTS_ROOT / "cem" / args.stage)
    fail_if_missing_outputs(args.stage, rows, results_dir)
    results = evaluate_rollouts(args.stage, rows, results_dir)
    if not results:
        raise SystemExit("No E107 rollout results found.")
    write_eval_outputs(args.stage, results, results_dir)


if __name__ == "__main__":
    main()
