#!/usr/bin/env python3
"""Compute E172/E173 object z-position tracking error for four box objects."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "workspace/core4d/scripts"))

from eval.core.core_metrics import _table4_tracking_metrics, npz_qpos  # noqa: E402


E173_INPUT = REPO / "workspace/core4d/results/E173/s6_downstream/eval/full/e173_case_metrics.tsv"
E172_INPUT = REPO / "workspace/core4d/results/E172/s6_downstream/eval/full/e171_case_metrics.tsv"
E194_INPUT = REPO / "workspace/core4d/results/E194/s6_downstream/eval/full/e194_arm_case_metrics.tsv"
E194_EXPANSION_INPUT = REPO / "workspace/core4d/results/E194/s6_downstream/eval/full_g1_expansion/e194_g1_expansion_case_metrics.tsv"
DEFAULT_OUT = E173_INPUT.parent
SOURCE_SPECS = (
    ("E173", "A0", E173_INPUT, ("box001", "box023", "box024"), ""),
    ("E172", "A0", E172_INPUT, ("box004",), ""),
    ("E194", "G1", E194_INPUT, ("box024", "box004"), "G1"),
    ("E194", "G1", E194_EXPANSION_INPUT, ("box001", "box023", "box021"), "G1"),
)
GROUPS = (
    ("E173", "A0", "box001"),
    ("E173", "A0", "box024"),
    ("E173", "A0", "box023"),
    ("E172", "A0", "box004"),
    ("E194", "G1", "box024"),
    ("E194", "G1", "box004"),
    ("E194", "G1", "box001"),
    ("E194", "G1", "box023"),
    ("E194", "G1", "box021"),
)
EXPECTED_COUNTS = {
    ("E173", "A0", "box001"): 28,
    ("E173", "A0", "box024"): 9,
    ("E173", "A0", "box023"): 16,
    ("E172", "A0", "box004"): 6,
    ("E194", "G1", "box024"): 9,
    ("E194", "G1", "box004"): 6,
    ("E194", "G1", "box001"): 28,
    ("E194", "G1", "box023"): 16,
    ("E194", "G1", "box021"): 28,
}
METRIC = "track_obj_z_abs_err_cm_mean"
FROZEN_3D_TOL_CM = 1e-4


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute() and path.is_file():
        return path
    if not path.is_absolute() and (REPO / path).is_file():
        return REPO / path
    marker = "core4d/results/results/"
    if marker in raw:
        return REPO / "workspace/core4d/results" / raw.split(marker, 1)[1]
    if path.is_absolute():
        return path
    return REPO / path


def resolve_reference(row: dict[str, str]) -> Path:
    """Resolve the fixed reference, including the persisted S3 trimmed copy."""
    direct = repo_path(row["trajectory"])
    if direct.is_file():
        return direct
    variant = row.get("retarget_variant_id", "omnirt_v1")
    case_id = row["case_id"]
    source_exp = row["source_exp_id"]
    root = (
        REPO
        / f"workspace/core4d/results/{source_exp}/s3_retarget"
        / variant
        / "ref_fk/results"
        / f"{variant}_ref_fk"
        / f"holosoma_dcv3_{variant}_ref_fk_{case_id}"
        / "trimmed"
    )
    candidates = sorted(root.glob("*.npz"))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"{case_id}: missing reference trajectory: {direct} (fallback={root})")


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def compute_case(row: dict[str, str]) -> dict[str, Any]:
    qpos_path = repo_path(row["qpos_path"])
    trajectory_path = resolve_reference(row)
    scene_path = repo_path(row["scene_xml"])
    for label, path in (
        ("qpos_path", qpos_path),
        ("trajectory", trajectory_path),
        ("scene_xml", scene_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{row['case_id']}: missing {label}: {path}")

    run_qpos, _ = npz_qpos(qpos_path)
    kin_qpos = np.asarray(np.load(trajectory_path, allow_pickle=True)["qpos"], dtype=np.float64)
    if kin_qpos.ndim == 3:
        kin_qpos = kin_qpos[:, 0, :]
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    metrics = _table4_tracking_metrics(run_qpos, kin_qpos, model)
    value = float(metrics[METRIC])
    pos_3d = float(metrics["track_obj_pos_err_cm_mean"])
    frozen_pos_3d = float(row["track_obj_pos_err_cm_mean"])
    if not math.isfinite(value):
        raise ValueError(f"{row['case_id']}: non-finite {METRIC}")
    frozen_delta = pos_3d - frozen_pos_3d
    if abs(frozen_delta) > FROZEN_3D_TOL_CM:
        raise ValueError(
            f"{row['case_id']}: reference mismatch: recomputed 3D={pos_3d}, "
            f"frozen 3D={frozen_pos_3d}"
        )
    if value > pos_3d + 1e-9:
        raise ValueError(f"{row['case_id']}: z absolute error exceeds 3D position error")
    return {
        "source_exp_id": row["source_exp_id"],
        "evaluation_variant": row["evaluation_variant"],
        "case_id": row["case_id"],
        "object_key": row["object_key"],
        "frames": min(len(run_qpos), len(kin_qpos)),
        METRIC: value,
        "track_obj_pos_err_cm_mean": pos_3d,
        "frozen_track_obj_pos_err_cm_mean": frozen_pos_3d,
        "recomputed_minus_frozen_3d_cm": frozen_delta,
        "z_error_share_of_3d": value / pos_3d if pos_3d > 0.0 else math.nan,
        "qpos_path": str(qpos_path.relative_to(REPO)),
        "trajectory": str(trajectory_path.relative_to(REPO)),
        "scene_xml": str(scene_path.relative_to(REPO)),
    }


def object_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["source_exp_id"]),
            str(row["evaluation_variant"]),
            str(row["object_key"]),
        )
        groups[key].append(row)
    output = []
    for source_exp, evaluation_variant, object_key in GROUPS:
        group = groups[(source_exp, evaluation_variant, object_key)]
        values = [float(row[METRIC]) for row in group]
        pos_3d = [float(row["track_obj_pos_err_cm_mean"]) for row in group]
        output.append(
            {
                "source_exp_id": source_exp,
                "evaluation_variant": evaluation_variant,
                "object_key": object_key,
                "case_count": len(group),
                "frame_count": sum(int(row["frames"]) for row in group),
                f"{METRIC}_case_macro_mean": statistics.fmean(values),
                f"{METRIC}_case_median": statistics.median(values),
                f"{METRIC}_case_min": min(values),
                f"{METRIC}_case_max": max(values),
                "track_obj_pos_err_cm_mean_case_macro_mean": statistics.fmean(pos_3d),
                "z_error_share_of_3d_case_macro_mean": statistics.fmean(
                    float(row["z_error_share_of_3d"]) for row in group
                ),
            }
        )
    return output


def render_report(case_rows: list[dict[str, Any]], object_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# E172/E173/E194 object tracking position error：z 方向",
        "",
        "_E173 box001/box023/box024、E172 box004 baseline 与 E194 G1；full CEM，更新于 2026-08-10_",
        "",
        "---",
        "",
        "## 📐 指标定义",
        "",
        f"主指标为 `{METRIC}`（cm，越低越好）：",
        "",
        "```text",
        "mean_t(abs(z_sim(t) - z_ref(t))) * 100",
        "```",
        "",
        "`z_sim` 是相应来源实验 full CEM 最终轨迹中 object body 的世界坐标 z；",
        "`z_ref` 是与现有 `track_obj_pos_err_cm_mean` 相同的固定 kinematic reference。",
        "逐 case 先在共同帧 `min(T_sim, T_ref)` 上取帧均值；by-object 再对 case 指标做等权宏平均。",
        "绝对值可避免上偏和下偏相互抵消。本指标是诊断指标，不改写 E172/E173/E194 既有 release gate。",
        "",
        "## 📊 By-object 平均",
        "",
        f"| Source | Variant | Object | Cases | Frames | z MAE macro mean (cm) | Median (cm) | Min–max (cm) | 3D pos mean (cm) | z / 3D mean |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in object_rows:
        lines.append(
            f"| {row['source_exp_id']} | {row['evaluation_variant']} | {row['object_key']} | {row['case_count']} | {row['frame_count']} | "
            f"{fmt(row[f'{METRIC}_case_macro_mean'])} | "
            f"{fmt(row[f'{METRIC}_case_median'])} | "
            f"{fmt(row[f'{METRIC}_case_min'])}–{fmt(row[f'{METRIC}_case_max'])} | "
            f"{fmt(row['track_obj_pos_err_cm_mean_case_macro_mean'])} | "
            f"{fmt(100.0 * row['z_error_share_of_3d_case_macro_mean'], 2)}% |"
        )

    lines.extend(
        [
            "",
            "## 📋 By-case 指标",
            "",
            "| Source | Variant | Case | Object | Frames | z MAE (cm) | 3D pos mean (cm) | z / 3D |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in case_rows:
        lines.append(
            f"| {row['source_exp_id']} | {row['evaluation_variant']} | `{row['case_id']}` | {row['object_key']} | {row['frames']} | "
            f"{fmt(row[METRIC])} | {fmt(row['track_obj_pos_err_cm_mean'])} | "
            f"{fmt(100.0 * row['z_error_share_of_3d'], 2)}% |"
        )
    lines.extend(
        [
            "",
            "## ✅ 一致性检查",
            "",
            f"- 覆盖 `{len(case_rows)}` 条 evaluation rows、`{len({r['case_id'] for r in case_rows})}` 个 case IDs；其中 E194 G1 为 `{sum(r['source_exp_id'] == 'E194' and r['evaluation_variant'] == 'G1' for r in case_rows)}` 条",
            f"- 每条重算的 3D object position error 与相应冻结输入表差值不超过 `{FROZEN_3D_TOL_CM:g} cm`",
            "- 每条 z 绝对误差均不超过对应的 3D L2 position error",
            "- 输入、脚本与输出 SHA256 记录在配套 summary JSON 中",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    selected: list[dict[str, str]] = []
    for source_exp, evaluation_variant, source_path, source_objects, required_arm in SOURCE_SPECS:
        for row in read_tsv(source_path):
            if row.get("object_key") in source_objects and (not required_arm or row.get("arm") == required_arm):
                row["source_exp_id"] = source_exp
                row["evaluation_variant"] = evaluation_variant
                selected.append(row)
    observed = {
        group: sum(
            (row["source_exp_id"], row["evaluation_variant"], row["object_key"]) == group
            for row in selected
        )
        for group in GROUPS
    }
    unique_rows = {
        (row["source_exp_id"], row["evaluation_variant"], row["case_id"])
        for row in selected
    }
    if observed != EXPECTED_COUNTS or len(unique_rows) != sum(EXPECTED_COUNTS.values()):
        raise ValueError(f"unexpected source authority: expected={EXPECTED_COUNTS}, observed={observed}")
    case_rows = [compute_case(row) for row in selected]
    group_rank = {group: index for index, group in enumerate(GROUPS)}
    case_rows.sort(
        key=lambda row: (
            group_rank[(str(row["source_exp_id"]), str(row["evaluation_variant"]), str(row["object_key"]))],
            str(row["case_id"]),
        )
    )
    object_rows = object_summary(case_rows)

    case_path = args.out_dir / "e173_object_tracking_position_error_z_by_case.tsv"
    object_path = args.out_dir / "e173_object_tracking_position_error_z_by_object.tsv"
    report_path = args.out_dir / "E173_object_tracking_position_error_z_report.md"
    summary_path = args.out_dir / "e173_object_tracking_position_error_z_summary.json"
    write_tsv(case_path, case_rows, list(case_rows[0]))
    write_tsv(object_path, object_rows, list(object_rows[0]))
    report_path.write_text(render_report(case_rows, object_rows), encoding="utf-8")

    summary = {
        "experiments": ["E173", "E172", "E194"],
        "groups": [list(group) for group in GROUPS],
        "metric": METRIC,
        "unit": "cm",
        "direction": "lower_is_better",
        "formula": "mean_t(abs(z_sim(t) - z_ref(t))) * 100",
        "case_aggregation": "equal-weight macro mean",
        "case_count": len(case_rows),
        "inputs": {
            str(path.relative_to(REPO)): sha256(path)
            for _, _, path, _, _ in SOURCE_SPECS
        },
        "generator": str(Path(__file__).resolve().relative_to(REPO)),
        "generator_sha256": sha256(Path(__file__)),
        "outputs": {},
        "by_object": object_rows,
    }
    for path in (case_path, object_path, report_path):
        summary["outputs"][str(path.relative_to(REPO))] = sha256(path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
