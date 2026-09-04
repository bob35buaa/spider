#!/usr/bin/env python3
"""Compute E178 full-CEM object z-height diff vs the fixed kinematic reference."""

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
from openpyxl import Workbook
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "workspace/core4d/scripts"))

from eval.core.core_metrics import npz_qpos  # noqa: E402


EVAL_DIR = REPO / "workspace/core4d/results/E178/s6_downstream/eval/full"
MANIFEST = EVAL_DIR / "evaluated_manifest_snapshot.tsv"
CASE_METRICS = EVAL_DIR / "e178_case_metrics.tsv"
OBJECT_ORDER = ("bucket003", "bucket004", "bucket007")
EXPECTED_COUNTS = {"bucket003": 9, "bucket004": 4, "bucket007": 14}
FROZEN_3D_TOL_CM = 1e-4
MAX_FRAME_ROWS = 40000

CASE_FIELDS = [
    "case_id",
    "object_key",
    "person",
    "variant",
    "frames",
    "duration_s",
    "z_diff_mean_cm",
    "z_mae_cm",
    "z_rmse_cm",
    "z_abs_p95_cm",
    "z_abs_max_cm",
    "z_diff_min_cm",
    "z_diff_max_cm",
    "z_diff_initial_cm",
    "z_diff_terminal_cm",
    "z_abs_gt_2cm_frac",
    "z_abs_gt_5cm_frac",
    "sim_obj_z_mean_m",
    "ref_obj_z_mean_m",
    "ref_obj_z_range_m",
    "track_obj_pos_err_cm_mean",
    "frozen_track_obj_pos_err_cm_mean",
    "recomputed_minus_frozen_3d_cm",
    "z_mae_share_of_3d",
    "manual_quality_label",
    "numeric_release_pass",
    "qpos_path",
    "trajectory",
    "scene_xml",
]

OBJECT_FIELDS = [
    "object_key",
    "case_count",
    "frame_count",
    "z_diff_mean_cm_macro_mean",
    "z_mae_cm_macro_mean",
    "z_mae_cm_median",
    "z_mae_cm_min",
    "z_mae_cm_max",
    "z_rmse_cm_macro_mean",
    "z_abs_p95_cm_macro_mean",
    "z_abs_max_cm_worst",
    "z_diff_terminal_cm_macro_mean",
    "z_abs_gt_2cm_frac_macro_mean",
    "z_abs_gt_5cm_frac_macro_mean",
    "track_obj_pos_err_cm_mean_macro_mean",
    "z_mae_share_of_3d_macro_mean",
    "ref_obj_z_range_m_macro_mean",
]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, delimiter="\t", lineterminator="\n", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel_label(path: Path) -> str:
    """Repo-relative label that survives the symlinked `results/` tree.

    `workspace/core4d/results` points at another mount, so `resolve()` escapes
    REPO; compare lexically instead and fall back to the path as given.
    """
    candidate = path if path.is_absolute() else REPO / path
    try:
        return str(candidate.relative_to(REPO))
    except ValueError:
        return str(path)


def fmt(value: float, digits: int = 3) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def signed(value: float, digits: int = 3) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    return f"{float(value):+.{digits}f}"


def object_z_series(
    run_qpos: np.ndarray, kin_qpos: np.ndarray, model: mujoco.MjModel
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (z_sim, z_ref, obj_pos_err_3d) over the common frame window.

    Mirrors `_table4_tracking_metrics` object handling: the run scene uses a
    6-DoF object (42 qpos) while the kinematic reference carries a 7-DoF
    freejoint object after the shared 36-dim robot prefix (43 qpos).
    """
    object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    if object_id < 0:
        raise ValueError("scene has no `object` body")
    horizon = min(run_qpos.shape[0], kin_qpos.shape[0])
    nq_robot = min(36, max(0, model.nq - 6), run_qpos.shape[1], kin_qpos.shape[1])
    if nq_robot <= 7:
        raise ValueError(f"unsupported qpos layout: nq_robot={nq_robot}")

    data_run = mujoco.MjData(model)
    data_ref = mujoco.MjData(model)
    z_sim = np.empty(horizon, dtype=np.float64)
    z_ref = np.empty(horizon, dtype=np.float64)
    err_3d = np.empty(horizon, dtype=np.float64)

    for i in range(horizon):
        q_run = run_qpos[i]
        data_run.qpos[:] = q_run
        data_run.qvel[:] = 0.0
        mujoco.mj_forward(model, data_run)

        if kin_qpos.shape[1] == model.nq:
            q_ref = kin_qpos[i, : model.nq].copy()
            data_ref.qpos[:] = q_ref
            data_ref.qvel[:] = 0.0
            mujoco.mj_forward(model, data_ref)
            ref_obj_pos = data_ref.xpos[object_id].copy()
        elif kin_qpos.shape[1] >= nq_robot + 7:
            ref_obj_pos = np.asarray(kin_qpos[i, nq_robot : nq_robot + 3], dtype=np.float64)
        else:
            raise ValueError(f"cannot read reference object pose: kin nq={kin_qpos.shape[1]}")

        z_sim[i] = float(data_run.xpos[object_id, 2])
        z_ref[i] = float(ref_obj_pos[2])
        err_3d[i] = float(np.linalg.norm(data_run.xpos[object_id] - ref_obj_pos))

    return z_sim, z_ref, err_3d


def compute_case(
    row: dict[str, str], frozen: dict[str, str], extra: dict[str, str] | None = None
) -> tuple[dict[str, Any], np.ndarray]:
    qpos_path = REPO / row["outdir_npz"]
    trajectory_path = REPO / row["trajectory"]
    scene_path = REPO / row["scene_act"]
    for label, path in (
        ("outdir_npz", qpos_path),
        ("trajectory", trajectory_path),
        ("scene_act", scene_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{row['case_id']}: missing {label}: {path}")

    expected_scene_sha = row.get("effective_scene_sha256", "")
    if not expected_scene_sha:
        raise ValueError(f"{row['case_id']}: manifest has no effective_scene_sha256")
    actual_scene_sha = sha256(scene_path)
    if actual_scene_sha != expected_scene_sha:
        raise ValueError(
            f"{row['case_id']}: scene XML drifted from the frozen manifest: "
            f"{scene_path} sha256={actual_scene_sha}, expected={expected_scene_sha}"
        )

    run_qpos, _ = npz_qpos(qpos_path)
    kin_qpos = np.asarray(np.load(trajectory_path, allow_pickle=True)["qpos"], dtype=np.float64)
    if kin_qpos.ndim == 3:
        kin_qpos = kin_qpos[:, 0, :]
    model = mujoco.MjModel.from_xml_path(str(scene_path))

    z_sim, z_ref, err_3d = object_z_series(run_qpos, kin_qpos, model)
    dz_cm = (z_sim - z_ref) * 100.0
    abs_dz = np.abs(dz_cm)
    pos_3d = float(np.mean(err_3d) * 100.0)
    frozen_pos_3d = float(frozen["track_obj_pos_err_cm_mean"])
    frozen_delta = pos_3d - frozen_pos_3d
    if abs(frozen_delta) > FROZEN_3D_TOL_CM:
        raise ValueError(
            f"{row['case_id']}: reference mismatch: recomputed 3D={pos_3d}, frozen={frozen_pos_3d}"
        )
    z_mae = float(np.mean(abs_dz))
    if z_mae > pos_3d + 1e-9:
        raise ValueError(f"{row['case_id']}: z MAE exceeds 3D position error")

    frames = int(len(dz_cm))
    duration = float(frozen["duration_s"]) if frozen.get("duration_s") else math.nan
    record = {
        "case_id": row["case_id"],
        "object_key": row["object_key"],
        "person": row["person"],
        "variant": row["variant"],
        "frames": frames,
        "duration_s": duration,
        "z_diff_mean_cm": float(np.mean(dz_cm)),
        "z_mae_cm": z_mae,
        "z_rmse_cm": float(np.sqrt(np.mean(dz_cm**2))),
        "z_abs_p95_cm": float(np.percentile(abs_dz, 95)),
        "z_abs_max_cm": float(np.max(abs_dz)),
        "z_diff_min_cm": float(np.min(dz_cm)),
        "z_diff_max_cm": float(np.max(dz_cm)),
        "z_diff_initial_cm": float(dz_cm[0]),
        "z_diff_terminal_cm": float(dz_cm[-1]),
        "z_abs_gt_2cm_frac": float(np.mean(abs_dz > 2.0)),
        "z_abs_gt_5cm_frac": float(np.mean(abs_dz > 5.0)),
        "sim_obj_z_mean_m": float(np.mean(z_sim)),
        "ref_obj_z_mean_m": float(np.mean(z_ref)),
        "ref_obj_z_range_m": float(np.max(z_ref) - np.min(z_ref)),
        "track_obj_pos_err_cm_mean": pos_3d,
        "frozen_track_obj_pos_err_cm_mean": frozen_pos_3d,
        "recomputed_minus_frozen_3d_cm": frozen_delta,
        "z_mae_share_of_3d": z_mae / pos_3d if pos_3d > 0.0 else math.nan,
        "manual_quality_label": (extra or {}).get("manual_quality_label", ""),
        "numeric_release_pass": (extra or {}).get("numeric_release_pass", ""),
        "qpos_path": row["outdir_npz"],
        "trajectory": row["trajectory"],
        "scene_xml": row["scene_act"],
    }
    series = np.stack([z_sim, z_ref, dz_cm], axis=1)
    return record, series


def object_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["object_key"])].append(row)

    def block(key: str, group: list[dict[str, Any]]) -> dict[str, Any]:
        mae = [float(r["z_mae_cm"]) for r in group]
        return {
            "object_key": key,
            "case_count": len(group),
            "frame_count": sum(int(r["frames"]) for r in group),
            "z_diff_mean_cm_macro_mean": statistics.fmean(
                float(r["z_diff_mean_cm"]) for r in group
            ),
            "z_mae_cm_macro_mean": statistics.fmean(mae),
            "z_mae_cm_median": statistics.median(mae),
            "z_mae_cm_min": min(mae),
            "z_mae_cm_max": max(mae),
            "z_rmse_cm_macro_mean": statistics.fmean(float(r["z_rmse_cm"]) for r in group),
            "z_abs_p95_cm_macro_mean": statistics.fmean(float(r["z_abs_p95_cm"]) for r in group),
            "z_abs_max_cm_worst": max(float(r["z_abs_max_cm"]) for r in group),
            "z_diff_terminal_cm_macro_mean": statistics.fmean(
                float(r["z_diff_terminal_cm"]) for r in group
            ),
            "z_abs_gt_2cm_frac_macro_mean": statistics.fmean(
                float(r["z_abs_gt_2cm_frac"]) for r in group
            ),
            "z_abs_gt_5cm_frac_macro_mean": statistics.fmean(
                float(r["z_abs_gt_5cm_frac"]) for r in group
            ),
            "track_obj_pos_err_cm_mean_macro_mean": statistics.fmean(
                float(r["track_obj_pos_err_cm_mean"]) for r in group
            ),
            "z_mae_share_of_3d_macro_mean": statistics.fmean(
                float(r["z_mae_share_of_3d"]) for r in group
            ),
            "ref_obj_z_range_m_macro_mean": statistics.fmean(
                float(r["ref_obj_z_range_m"]) for r in group
            ),
        }

    output = [block(key, groups[key]) for key in OBJECT_ORDER if groups.get(key)]
    output.append(block("ALL", rows))
    return output


HEADER_FILL = PatternFill("solid", fgColor="1F3864")
HEADER_FONT = Font(bold=True, color="FFFFFF", size=10)
TOTAL_FILL = PatternFill("solid", fgColor="FFF2CC")
BAD_FILL = PatternFill("solid", fgColor="F8CBAD")


def style_sheet(sheet, fields: list[str], freeze: str = "B2") -> None:
    for index, name in enumerate(fields, start=1):
        cell = sheet.cell(row=1, column=index)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        width = max(11, min(38, len(name) + 3))
        sheet.column_dimensions[get_column_letter(index)].width = width
    sheet.row_dimensions[1].height = 34
    sheet.freeze_panes = freeze
    sheet.auto_filter.ref = sheet.dimensions


def write_xlsx(
    path: Path,
    case_rows: list[dict[str, Any]],
    object_rows: list[dict[str, Any]],
    frame_rows: list[tuple[str, str, int, float, float, float, float]],
    meta: list[tuple[str, str]],
) -> None:
    workbook = Workbook()

    readme = workbook.active
    readme.title = "README"
    readme.column_dimensions["A"].width = 34
    readme.column_dimensions["B"].width = 104
    for row_index, (key, value) in enumerate(meta, start=1):
        readme.cell(row=row_index, column=1, value=key).font = Font(bold=True)
        cell = readme.cell(row=row_index, column=2, value=value)
        cell.alignment = Alignment(wrap_text=True, vertical="top")

    by_object = workbook.create_sheet("By-object")
    by_object.append(OBJECT_FIELDS)
    for row in object_rows:
        by_object.append([row[field] for field in OBJECT_FIELDS])
    style_sheet(by_object, OBJECT_FIELDS)
    for cell in by_object[by_object.max_row]:
        cell.fill = TOTAL_FILL
        cell.font = Font(bold=True)
    for index, field in enumerate(OBJECT_FIELDS, start=1):
        if field in {"object_key", "case_count", "frame_count"}:
            continue
        column = get_column_letter(index)
        number_format = "0.0000" if "share" in field or "frac" in field else "0.000"
        for row_index in range(2, by_object.max_row + 1):
            by_object[f"{column}{row_index}"].number_format = number_format

    by_case = workbook.create_sheet("By-case")
    by_case.append(CASE_FIELDS)
    for row in case_rows:
        by_case.append([row[field] for field in CASE_FIELDS])
    style_sheet(by_case, CASE_FIELDS)
    numeric_columns = {
        field: get_column_letter(index)
        for index, field in enumerate(CASE_FIELDS, start=1)
        if field
        not in {
            "case_id",
            "object_key",
            "person",
            "variant",
            "manual_quality_label",
            "numeric_release_pass",
            "qpos_path",
            "trajectory",
            "scene_xml",
        }
    }
    for field, column in numeric_columns.items():
        number_format = "0.0000" if "share" in field or "frac" in field else "0.000"
        if field == "frames":
            number_format = "0"
        for row_index in range(2, by_case.max_row + 1):
            by_case[f"{column}{row_index}"].number_format = number_format
    last = by_case.max_row
    for field in ("z_mae_cm", "z_abs_p95_cm", "z_abs_max_cm", "z_abs_gt_2cm_frac", "z_abs_gt_5cm_frac"):
        column = numeric_columns[field]
        by_case.conditional_formatting.add(
            f"{column}2:{column}{last}",
            ColorScaleRule(
                start_type="min",
                start_color="C6EFCE",
                mid_type="percentile",
                mid_value=50,
                mid_color="FFEB9C",
                end_type="max",
                end_color="F8CBAD",
            ),
        )
    mae_column = numeric_columns["z_mae_cm"]
    for row_index in range(2, last + 1):
        if float(by_case[f"{mae_column}{row_index}"].value) >= 5.0:
            by_case.cell(row=row_index, column=1).fill = BAD_FILL

    frames_sheet = workbook.create_sheet("Frame-series")
    frame_fields = ["case_id", "object_key", "frame", "t_s", "z_sim_m", "z_ref_m", "z_diff_cm"]
    frames_sheet.append(frame_fields)
    for record in frame_rows:
        frames_sheet.append(list(record))
    style_sheet(frames_sheet, frame_fields)
    for column, number_format in (("D", "0.000"), ("E", "0.0000"), ("F", "0.0000"), ("G", "0.000")):
        for row_index in range(2, frames_sheet.max_row + 1):
            frames_sheet[f"{column}{row_index}"].number_format = number_format

    workbook.save(path)


def render_report(
    case_rows: list[dict[str, Any]],
    object_rows: list[dict[str, Any]],
    generated_at: str,
    scope: dict[str, str],
    compare: list[dict[str, Any]] | None = None,
) -> str:
    total = object_rows[-1]
    worst = sorted(case_rows, key=lambda r: -float(r["z_mae_cm"]))[:5]
    higher = [r for r in case_rows if float(r["z_diff_mean_cm"]) > 0.0]
    lift_range = np.array([float(r["ref_obj_z_range_m"]) for r in case_rows])
    bias = np.array([float(r["z_diff_mean_cm"]) for r in case_rows])
    mae = np.array([float(r["z_mae_cm"]) for r in case_rows])
    corr_bias = float(np.corrcoef(lift_range, bias)[0, 1])
    corr_mae = float(np.corrcoef(lift_range, mae)[0, 1])
    slope, intercept = (float(v) for v in np.polyfit(lift_range, bias, 1))
    per_object = sorted(
        (r for r in object_rows if r["object_key"] != "ALL"),
        key=lambda r: float(r["ref_obj_z_range_m_macro_mean"]),
    )
    if len(per_object) >= 2:
        lo, hi = per_object[0], per_object[-1]
        lift_sentence = (
            f"抬升幅度最大的 {hi['object_key']}（参考 z range 均值 "
            f"{fmt(hi['ref_obj_z_range_m_macro_mean'])} m）欠抬升最严重"
            f"（bias {signed(hi['z_diff_mean_cm_macro_mean'], 2)} cm）；"
            f"抬升最小的 {lo['object_key']}（{fmt(lo['ref_obj_z_range_m_macro_mean'])} m）"
            f"最轻（bias {signed(lo['z_diff_mean_cm_macro_mean'], 2)} cm）。"
        )
    else:
        lift_sentence = ""
    lines = [
        f"# {scope['title']}",
        "",
        f"_{scope['blurb'].format(n=len(case_rows))}；生成于 {generated_at}_",
        "",
        "---",
        "",
        "## 📐 指标定义",
        "",
        "对每一帧 `t`，取物理仿真结果与固定 kinematic reference 的 object body 世界坐标 z：",
        "",
        "```text",
        "z_diff(t) = z_sim(t) - z_ref(t)          # 单位 cm，正值 = 重定向结果比参考更高",
        "z_mae     = mean_t( |z_diff(t)| )",
        "z_bias    = mean_t(   z_diff(t)  )       # 符号保留，反映系统性偏高/偏低",
        "```",
        "",
        "- `z_sim`：E178 full CEM 最终轨迹 `trajectory_mjwp_act.npz`，经 scene XML 前向运动学取 `object` body 的世界 z。",
        "- `z_ref`：manifest 中冻结的参考轨迹 `trajectory_kinematic.npz`（`omnirt_v1 / ref_fk`）"
        "的 object freejoint 世界位姿 z，与既有 `track_obj_pos_err_cm_mean` 使用同一参考。",
        "- 逐 case 在公共帧窗 `min(T_sim, T_ref)` 上取帧均值；by-object 再对 case 指标做等权宏平均。",
        "- `z_bias` 与 `z_mae` 同时给出：仅看 MAE 会掩盖方向，仅看 bias 会让上下偏移相互抵消。",
        "- 本指标为**诊断指标**，不改写 E178 既有 release gate。",
        "",
        "## 📊 总体结果",
        "",
        f"- 覆盖 `{total['case_count']}` 个 case、`{total['frame_count']}` 帧。",
        f"- **z MAE 宏平均 = {fmt(total['z_mae_cm_macro_mean'])} cm**，中位数 {fmt(total['z_mae_cm_median'])} cm，"
        f"范围 {fmt(total['z_mae_cm_min'])}–{fmt(total['z_mae_cm_max'])} cm。",
        f"- **z bias 宏平均 = {signed(total['z_diff_mean_cm_macro_mean'])} cm**"
        f"（{len(higher)}/{len(case_rows)} 个 case 的平均 z 高于参考）。",
        f"- 单帧最大绝对偏差 {fmt(total['z_abs_max_cm_worst'])} cm；"
        f"|z_diff| > 2 cm 的帧占比宏平均 {fmt(100.0 * total['z_abs_gt_2cm_frac_macro_mean'], 2)}%，"
        f"> 5 cm 为 {fmt(100.0 * total['z_abs_gt_5cm_frac_macro_mean'], 2)}%。",
        f"- z 方向占 3D object position error 的比例宏平均 "
        f"{fmt(100.0 * total['z_mae_share_of_3d_macro_mean'], 2)}%"
        f"（3D 宏平均 {fmt(total['track_obj_pos_err_cm_mean_macro_mean'])} cm）。",
        "",
        "## 🔍 主要发现：抬升越高，物体越跟不上",
        "",
        f"`z_diff` 并非零均值噪声，而是明显的**单向欠抬升**：{len(case_rows) - len(higher)}/{len(case_rows)} "
        "个 case 的物体平均低于参考轨迹。把每个 case 的参考抬升幅度 "
        "`ref_obj_z_range_m`（参考轨迹中物体 z 的最大−最小）与 z bias 做相关：",
        "",
        "```text",
        f"pearson r(ref_z_range, z_bias) = {corr_bias:+.3f}    (n={len(case_rows)})",
        f"pearson r(ref_z_range, z_mae)  = {corr_mae:+.3f}",
        f"线性拟合: z_bias ≈ {slope:.1f} * ref_z_range {intercept:+.2f}   (cm, range 单位 m)",
        "```",
        "",
        f"即参考每多抬高 10 cm，物理结果平均多欠 {abs(slope) * 0.1:.1f} cm。{lift_sentence}",
        "",
        "这与「随机跟踪误差」的解释不符，更像是抓握力/接触支撑不足导致物体在抬升段被带不动或下滑。"
        "需要视频复看确认：是全程贴着参考往下偏，还是在抬升峰值附近掉落。",
        "",
        "## 📦 By-object 汇总",
        "",
        "| Object | Cases | Frames | 参考抬升 (m) | z bias (cm) | z MAE (cm) | MAE 中位 | MAE min–max | z RMSE (cm) "
        "| z \\|p95\\| (cm) | 最差单帧 (cm) | 末帧 bias (cm) | >2cm 帧占比 | >5cm 帧占比 | 3D pos (cm) | z/3D |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in object_rows:
        label = "**ALL**" if row["object_key"] == "ALL" else row["object_key"]
        lines.append(
            f"| {label} | {row['case_count']} | {row['frame_count']} | "
            f"{fmt(row['ref_obj_z_range_m_macro_mean'])} | "
            f"{signed(row['z_diff_mean_cm_macro_mean'])} | {fmt(row['z_mae_cm_macro_mean'])} | "
            f"{fmt(row['z_mae_cm_median'])} | {fmt(row['z_mae_cm_min'])}–{fmt(row['z_mae_cm_max'])} | "
            f"{fmt(row['z_rmse_cm_macro_mean'])} | {fmt(row['z_abs_p95_cm_macro_mean'])} | "
            f"{fmt(row['z_abs_max_cm_worst'])} | {signed(row['z_diff_terminal_cm_macro_mean'])} | "
            f"{fmt(100.0 * row['z_abs_gt_2cm_frac_macro_mean'], 2)}% | "
            f"{fmt(100.0 * row['z_abs_gt_5cm_frac_macro_mean'], 2)}% | "
            f"{fmt(row['track_obj_pos_err_cm_mean_macro_mean'])} | "
            f"{fmt(100.0 * row['z_mae_share_of_3d_macro_mean'], 2)}% |"
        )

    if compare:
        selected = {str(r["case_id"]) for r in case_rows}
        excluded = [r for r in compare if str(r["case_id"]) not in selected]

        def agg(group: list[dict[str, Any]]) -> tuple[int, float, float, float, float]:
            return (
                len(group),
                statistics.fmean(float(r["z_diff_mean_cm"]) for r in group),
                statistics.fmean(float(r["z_mae_cm"]) for r in group),
                max(float(r["z_abs_max_cm"]) for r in group),
                statistics.fmean(float(r["z_abs_gt_5cm_frac"]) for r in group),
            )

        blocks = [("本报告子集", case_rows), ("E178 全量", compare)]
        if excluded:
            blocks.append(("被排除的 case", excluded))
        lines.extend(
            [
                "",
                "## ⚖️ 与 E178 全量的对比",
                "",
                "| 集合 | Cases | z bias (cm) | z MAE (cm) | 最差单帧 (cm) | >5cm 帧占比 |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for label, group in blocks:
            count, bias_mean, mae_mean, worst_frame, gt5 = agg(group)
            lines.append(
                f"| {label} | {count} | {signed(bias_mean)} | {fmt(mae_mean)} | "
                f"{fmt(worst_frame)} | {fmt(100.0 * gt5, 2)}% |"
            )
        sub_mae = statistics.fmean(float(r["z_mae_cm"]) for r in case_rows)
        all_mae = statistics.fmean(float(r["z_mae_cm"]) for r in compare)
        direction = "优于" if sub_mae < all_mae else "差于"
        lines.append("")
        lines.append(
            f"子集 z MAE 宏平均 {fmt(sub_mae)} cm，{direction}全量的 {fmt(all_mae)} cm"
            f"（差 {signed(sub_mae - all_mae)} cm）。"
        )

    labels = [str(r["manual_quality_label"]) for r in case_rows if r["manual_quality_label"]]
    if labels:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in case_rows:
            if row["manual_quality_label"]:
                groups[str(row["manual_quality_label"])].append(row)
        lines.extend(
            [
                "",
                "## 🧑‍⚖️ 按人工审查标签",
                "",
                "| manual_quality_label | Cases | z bias (cm) | z MAE (cm) | 最差单帧 (cm) | >5cm 帧占比 |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for label in sorted(groups):
            group = groups[label]
            lines.append(
                f"| {label} | {len(group)} | "
                f"{signed(statistics.fmean(float(r['z_diff_mean_cm']) for r in group))} | "
                f"{fmt(statistics.fmean(float(r['z_mae_cm']) for r in group))} | "
                f"{fmt(max(float(r['z_abs_max_cm']) for r in group))} | "
                f"{fmt(100.0 * statistics.fmean(float(r['z_abs_gt_5cm_frac']) for r in group), 2)}% |"
            )

    lines.extend(
        [
            "",
            "## 🔺 z MAE 最差的 5 个 case",
            "",
            "| Case | Object | z MAE (cm) | z bias (cm) | 单帧最差 (cm) | 末帧 bias (cm) | >5cm 帧占比 | 3D pos (cm) |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in worst:
        lines.append(
            f"| `{row['case_id']}` | {row['object_key']} | {fmt(row['z_mae_cm'])} | "
            f"{signed(row['z_diff_mean_cm'])} | {fmt(row['z_abs_max_cm'])} | "
            f"{signed(row['z_diff_terminal_cm'])} | "
            f"{fmt(100.0 * float(row['z_abs_gt_5cm_frac']), 2)}% | "
            f"{fmt(row['track_obj_pos_err_cm_mean'])} |"
        )

    lines.extend(
        [
            "",
            "## 📋 By-case 明细",
            "",
            "| Case | Object | Person | Frames | z bias (cm) | z MAE (cm) | z RMSE (cm) | "
            "\\|p95\\| (cm) | 单帧最差 (cm) | 首帧 (cm) | 末帧 (cm) | >2cm | >5cm | 3D pos (cm) | z/3D |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in case_rows:
        lines.append(
            f"| `{row['case_id']}` | {row['object_key']} | {row['person']} | {row['frames']} | "
            f"{signed(row['z_diff_mean_cm'])} | {fmt(row['z_mae_cm'])} | {fmt(row['z_rmse_cm'])} | "
            f"{fmt(row['z_abs_p95_cm'])} | {fmt(row['z_abs_max_cm'])} | "
            f"{signed(row['z_diff_initial_cm'])} | {signed(row['z_diff_terminal_cm'])} | "
            f"{fmt(100.0 * float(row['z_abs_gt_2cm_frac']), 1)}% | "
            f"{fmt(100.0 * float(row['z_abs_gt_5cm_frac']), 1)}% | "
            f"{fmt(row['track_obj_pos_err_cm_mean'])} | "
            f"{fmt(100.0 * float(row['z_mae_share_of_3d']), 1)}% |"
        )

    lines.extend(
        [
            "",
            "## ✅ 一致性检查",
            "",
            f"- {scope['consistency']}",
            f"- 每个 case 重算的 3D object position error 与 `e178_case_metrics.tsv` 冻结值"
            f"差值不超过 `{FROZEN_3D_TOL_CM:g} cm`（同一参考、同一帧窗，确认口径未漂移）。",
            "- 每个 case 的 z MAE 均不超过对应的 3D L2 position error（z 是 3D 的一个分量，必然成立）。",
            f"- {len(case_rows)} 个 `scene_act` XML 的 sha256 与 manifest "
            "`effective_scene_sha256` 全部逐字节匹配（由本脚本逐 case 校验，不匹配即中止）。",
            "- 输入、脚本与输出的 SHA256 记录在配套 summary JSON 中。",
            "",
            "## ⚠️ 口径说明",
            "",
            "- 参考轨迹是 `omnirt_v1 / ref_fk` 的 **kinematic reference**（OmniRetarget 输出后经 FK 的运动学轨迹），"
            "而非 CORE4D 原始人体/物体 mocap。它是 E178 CEM 的跟踪目标，也是既有 `track_obj_*` 指标的同一基准。",
            "- z 差异同时包含物理仿真的下沉/抬升与参考轨迹本身的物理不可行性，本指标不区分二者归因。",
            "- 数值指标不能替代视觉检查：结论落地前需对 z MAE 最差的 case 复看渲染视频，确认是持续偏移还是瞬时穿透。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=EVAL_DIR)
    parser.add_argument("--generated-at", default="2026-09-04")
    parser.add_argument(
        "--cases-from",
        type=Path,
        help="TSV with a case_id column; restricts the report to those cases",
    )
    parser.add_argument("--out-prefix", default="object_z_diff")
    parser.add_argument("--scope-title", default="E178 full CEM：物体 z 高度与参考轨迹的差异")
    parser.add_argument("--scope-blurb", default="全部 {n} 个 full CEM case")
    parser.add_argument(
        "--compare-to",
        type=Path,
        help="by-case TSV of a wider run, rendered as a side-by-side comparison",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = read_tsv(MANIFEST)
    frozen_by_case = {row["case_id"]: row for row in read_tsv(CASE_METRICS)}
    if len(manifest_rows) != len(frozen_by_case):
        raise ValueError(
            f"row mismatch: manifest={len(manifest_rows)}, metrics={len(frozen_by_case)}"
        )

    extra_by_case: dict[str, dict[str, str]] = {}
    if args.cases_from:
        subset = [row for row in read_tsv(args.cases_from) if row.get("case_id")]
        extra_by_case = {row["case_id"]: row for row in subset}
        if len(extra_by_case) != len(subset):
            raise ValueError(f"{args.cases_from}: duplicate case_id rows")
        known = {row["case_id"] for row in manifest_rows}
        missing = sorted(set(extra_by_case) - known)
        if missing:
            raise KeyError(f"{args.cases_from}: case_ids absent from E178 manifest: {missing}")
        manifest_rows = [row for row in manifest_rows if row["case_id"] in extra_by_case]

    case_rows: list[dict[str, Any]] = []
    frame_rows: list[tuple[str, str, int, float, float, float, float]] = []
    for row in manifest_rows:
        frozen = frozen_by_case.get(row["case_id"])
        if frozen is None:
            raise KeyError(f"{row['case_id']}: absent from {CASE_METRICS.name}")
        record, series = compute_case(row, frozen, extra_by_case.get(row["case_id"]))
        case_rows.append(record)
        duration = float(record["duration_s"])
        frames = int(record["frames"])
        dt = duration / frames if math.isfinite(duration) and frames else 1.0 / 30.0
        for index in range(frames):
            frame_rows.append(
                (
                    record["case_id"],
                    record["object_key"],
                    index,
                    index * dt,
                    float(series[index, 0]),
                    float(series[index, 1]),
                    float(series[index, 2]),
                )
            )

    observed = {
        key: sum(row["object_key"] == key for row in case_rows) for key in EXPECTED_COUNTS
    }
    if args.cases_from:
        if len(case_rows) != len(extra_by_case):
            raise ValueError(
                f"subset size mismatch: requested={len(extra_by_case)}, computed={len(case_rows)}"
            )
    elif observed != EXPECTED_COUNTS:
        raise ValueError(f"unexpected object counts: expected={EXPECTED_COUNTS}, observed={observed}")
    if len(frame_rows) > MAX_FRAME_ROWS:
        raise ValueError(f"frame series too large: {len(frame_rows)} > {MAX_FRAME_ROWS}")

    order = {key: index for index, key in enumerate(OBJECT_ORDER)}
    case_rows.sort(key=lambda r: (order[str(r["object_key"])], str(r["case_id"])))
    object_rows = object_summary(case_rows)

    stem = args.out_prefix
    case_path = args.out_dir / f"e178_{stem}_by_case.tsv"
    object_path = args.out_dir / f"e178_{stem}_by_object.tsv"
    frame_path = args.out_dir / f"e178_{stem}_frame_series.tsv"
    report_path = args.out_dir / f"E178_{stem}_report.md"
    xlsx_path = args.out_dir / f"E178_{stem}.xlsx"
    summary_path = args.out_dir / f"e178_{stem}_summary.json"

    write_tsv(case_path, case_rows, CASE_FIELDS)
    write_tsv(object_path, object_rows, OBJECT_FIELDS)
    write_tsv(
        frame_path,
        [
            {
                "case_id": r[0],
                "object_key": r[1],
                "frame": r[2],
                "t_s": f"{r[3]:.6f}",
                "z_sim_m": f"{r[4]:.6f}",
                "z_ref_m": f"{r[5]:.6f}",
                "z_diff_cm": f"{r[6]:.6f}",
            }
            for r in frame_rows
        ],
        ["case_id", "object_key", "frame", "t_s", "z_sim_m", "z_ref_m", "z_diff_cm"],
    )
    compare_rows = read_tsv(args.compare_to) if args.compare_to else None
    if compare_rows:
        overlap = {str(r["case_id"]) for r in case_rows} - {
            str(r["case_id"]) for r in compare_rows
        }
        if overlap:
            raise KeyError(f"--compare-to missing cases present in this scope: {sorted(overlap)}")
    counts = ", ".join(f"{key}={observed[key]}" for key in OBJECT_ORDER if observed[key])
    if args.cases_from:
        consistency = (
            f"覆盖 `{len(case_rows)}` 个 case，与 `{args.cases_from.name}` 的行数一致"
            f"（{counts}）；每个 case_id 均在 E178 evaluated manifest 中找到。"
        )
    else:
        consistency = (
            f"覆盖 `{len(case_rows)}` 个 case，与 `summary.json` 的 `evaluated = 27` 一致；"
            f"by-object 计数 {dict(EXPECTED_COUNTS)} 已断言。"
        )
    scope = {
        "title": args.scope_title,
        "blurb": args.scope_blurb,
        "consistency": consistency,
    }
    report_path.write_text(
        render_report(case_rows, object_rows, args.generated_at, scope, compare_rows),
        encoding="utf-8",
    )

    total = object_rows[-1]
    meta = [
        ("Experiment", "E178 — semantic bucket PRG, full CEM"),
        ("Scope", args.scope_title),
        ("Metric", "object body world-z diff vs fixed kinematic reference"),
        ("Formula", "z_diff(t) = z_sim(t) - z_ref(t), 单位 cm；正值 = 重定向结果高于参考"),
        ("z_sim", "s6_downstream/cem/full/<case>_outdir/trajectory_mjwp_act.npz → FK → body `object` 的世界 z"),
        ("z_ref", "manifest `trajectory` (omnirt_v1 / ref_fk kinematic) 的 object freejoint 世界 z"),
        ("Frame window", "min(T_sim, T_ref)，逐 case 帧均值；by-object 为 case 等权宏平均"),
        ("Cases", f"{len(case_rows)}（{counts}）"),
        ("Overall z MAE (cm)", fmt(total["z_mae_cm_macro_mean"])),
        ("Overall z bias (cm)", signed(total["z_diff_mean_cm_macro_mean"])),
        ("Worst single frame (cm)", fmt(total["z_abs_max_cm_worst"])),
        (
            "Consistency",
            f"重算 3D object pos err 与冻结表差值 ≤ {FROZEN_3D_TOL_CM:g} cm；"
            f"{len(case_rows)} 个 scene XML sha256 全匹配",
        ),
        ("Caveat", "诊断指标，不改写 release gate；参考为 ref_fk kinematic 轨迹而非 CORE4D 原始 mocap；结论需配合视频复看"),
        ("Generated at", args.generated_at),
        ("Generator", "workspace/core4d/scripts/eval/reports/gen_E178_object_z_diff_report.py"),
    ]
    write_xlsx(xlsx_path, case_rows, object_rows, frame_rows, meta)

    summary = {
        "experiment_id": "E178",
        "mode": "full",
        "metric": "object_world_z_diff_vs_kinematic_reference",
        "unit": "cm",
        "sign_convention": "positive = retargeted object higher than reference",
        "formula": "z_diff(t) = (z_sim(t) - z_ref(t)) * 100",
        "case_aggregation": "equal-weight macro mean over cases",
        "scope": args.scope_title,
        "case_count": len(case_rows),
        "case_ids": [str(r["case_id"]) for r in case_rows],
        "frame_count": int(total["frame_count"]),
        "generated_at": args.generated_at,
        "frozen_3d_tolerance_cm": FROZEN_3D_TOL_CM,
        "inputs": {
            str(MANIFEST.relative_to(REPO)): sha256(MANIFEST),
            str(CASE_METRICS.relative_to(REPO)): sha256(CASE_METRICS),
            **({rel_label(args.cases_from): sha256(args.cases_from)} if args.cases_from else {}),
        },
        "generator": rel_label(Path(__file__)),
        "generator_sha256": sha256(Path(__file__)),
        "by_object": object_rows,
        "lift_correlation": {
            "x": "ref_obj_z_range_m",
            "pearson_r_vs_z_diff_mean_cm": float(
                np.corrcoef(
                    [float(r["ref_obj_z_range_m"]) for r in case_rows],
                    [float(r["z_diff_mean_cm"]) for r in case_rows],
                )[0, 1]
            ),
            "pearson_r_vs_z_mae_cm": float(
                np.corrcoef(
                    [float(r["ref_obj_z_range_m"]) for r in case_rows],
                    [float(r["z_mae_cm"]) for r in case_rows],
                )[0, 1]
            ),
            "undershoot_case_count": sum(
                1 for r in case_rows if float(r["z_diff_mean_cm"]) < 0.0
            ),
        },
        "outputs": {},
    }
    for path in (case_path, object_path, frame_path, report_path, xlsx_path):
        summary["outputs"][rel_label(path)] = sha256(path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(report_path)
    print(xlsx_path)


if __name__ == "__main__":
    main()
