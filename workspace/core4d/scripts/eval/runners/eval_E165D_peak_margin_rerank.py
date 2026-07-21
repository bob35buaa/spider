#!/usr/bin/env python3
"""Evaluate E165-D peak-margin CEM rerank results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval.core.core_metrics import EvalConfig, evaluate_sequence, person_idx_from_case  # noqa: E402
from eval_E156_clean8_gate_decay import (  # noqa: E402
    add_success_flags,
    contact_mask_for_case,
    kin_ref_for_scene,
    repo_path,
    write_json,
    write_tsv,
)


REPO = Path(__file__).resolve().parents[5]
VARIANTS = REPO / "workspace/core4d/scripts/experiments/E165/variants.tsv"
RESULT_ROOT = REPO / "workspace/core4d/results/E165/peak_margin_rerank"
ISAAC_PROBE_JSON = REPO / "workspace/core4d/results/E165/E1_onrails_probe/onrails_probe_scalars.json"
E163_EVAL_DIR = REPO / "workspace/core4d/results/E163/narrow_surface_band/eval/full"
E163_SUMMARY = E163_EVAL_DIR / "e163_method_summary.tsv"
E163_CASE_STATUS = E163_EVAL_DIR / "e163_case_status.tsv"
TARGET_CASES = ["box023_person2", "box021_029_p2", "box004_083_p2"]
REFERENCE_METHODS = [
    "OmniRetarget",
    "SPIDER+rubberhand",
    "+gateA",
    "surfaceBand releaseDecay",
    "E163 narrowSurfaceBand",
]
FULL_METHOD_ORDER = [*REFERENCE_METHODS, "E165D peakMargin025"]

METRIC_KEYS = [
    "success_tracked",
    "fall_flag",
    "hand_object_physics_contact_in_mask_frac",
    "hand_object_physics_contact_in_rl_mask_frac",
    "hand_object_physics_contact_3mm_in_mask_frac",
    "hand_object_physics_penetration_3mm_frame_frac",
    "hand_geom_penetration_2mm_frac",
    "track_joint_err_deg_mean",
    "track_eef_pos_err_cm_mean",
    "track_root_pos_err_cm_mean",
    "track_pelvis_z_err_terminal_m",
]

HEALTH_KEYS = {
    "cem_peak_margin_valid_frac": ("mean", "peak_margin_valid_frac"),
    "cem_peak_margin_selected_valid_frac": ("mean", "peak_margin_selected_valid_frac"),
    "cem_peak_margin_fallback_used": ("mean", "peak_margin_fallback_used"),
    "sample_peak_margin_ee_peak_mean": ("mean", "peak_margin_ee_peak_mean"),
    "sample_peak_margin_ee_peak_max": ("max", "peak_margin_ee_peak_max"),
    "sample_peak_margin_anchor_peak_mean": ("mean", "peak_margin_anchor_peak_mean"),
    "sample_peak_margin_anchor_peak_max": ("max", "peak_margin_anchor_peak_max"),
    "sample_peak_margin_ee_margin_mean": ("mean", "peak_margin_ee_margin_mean"),
    "sample_peak_margin_anchor_margin_mean": ("mean", "peak_margin_anchor_margin_mean"),
    "sample_peak_margin_violation_mean": ("mean", "peak_margin_violation_mean"),
    "cem_posture_gate_valid_frac": ("mean", "posture_gate_valid_frac"),
    "cem_posture_gate_selected_valid_frac": ("mean", "posture_gate_selected_valid_frac"),
    "cem_posture_gate_fallback_used": ("mean", "posture_gate_fallback_used"),
}


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO.resolve()))
    except Exception:
        return str(path)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def finite(value: Any, default: float = math.nan) -> float:
    try:
        val = float(value)
    except Exception:
        return default
    return val if math.isfinite(val) else default


def mean(values: list[Any]) -> float:
    vals = [finite(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return statistics.fmean(vals) if vals else math.nan


def tsv_value(value: str) -> Any:
    if value == "":
        return ""
    low = value.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        val = float(value)
    except Exception:
        return value
    return val if math.isfinite(val) else ""


def convert_tsv_row(row: dict[str, str]) -> dict[str, Any]:
    return {key: tsv_value(value) for key, value in row.items()}


def stage_paths(row: dict[str, str], stage: str) -> tuple[Path, Path, Path, Path]:
    variant = row["variant"]
    root = RESULT_ROOT / "cem" / stage
    outdir = root / f"{variant}_outdir_{stage}"
    return (
        root / f"{variant}.npz",
        outdir / "trajectory_mjwp_act.npz",
        outdir / "config_act.yaml",
        root / f"{variant}_{stage}.mp4",
    )


def load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None or not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def close_enough(value: Any, target: float, tol: float = 1e-7) -> bool:
    try:
        return abs(float(value) - target) <= tol
    except Exception:
        return False


def artifact_check(row: dict[str, str], stage: str) -> dict[str, Any]:
    root_npz, outdir_npz, config_act, video = stage_paths(row, stage)
    cfg = load_yaml(config_act)
    checks = {
        "variant": row["variant"],
        "short_case_id": row["short_case_id"],
        "root_npz_exists": root_npz.is_file(),
        "trajectory_mjwp_act_exists": outdir_npz.is_file(),
        "config_act_exists": config_act.is_file(),
        "mp4_exists": video.is_file(),
        "config_peak_margin_enabled_ok": cfg.get("cem_peak_margin_enabled") is True,
        "config_ee_threshold_ok": close_enough(cfg.get("cem_peak_margin_ee_threshold_m"), 0.25),
        "config_anchor_threshold_ok": close_enough(cfg.get("cem_peak_margin_anchor_threshold_m"), 0.25),
        "config_buffer_ok": close_enough(cfg.get("cem_peak_margin_buffer_m"), 0.03),
        "config_path": rel(config_act),
        "root_npz": rel(root_npz),
        "trajectory_mjwp_act": rel(outdir_npz),
        "video": rel(video),
    }
    bool_keys = [key for key in checks if key.endswith("_exists") or key.endswith("_ok")]
    checks["artifact_ok"] = all(bool(checks[key]) for key in bool_keys)
    return checks


def run_health(qpos_path: Path) -> dict[str, Any]:
    out = {dst: "" for _, dst in HEALTH_KEYS.values()}
    if not qpos_path.is_file():
        return out
    data = np.load(qpos_path, allow_pickle=True)
    for src, (agg, dst) in HEALTH_KEYS.items():
        if src not in data.files:
            continue
        arr = np.asarray(data[src], dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if not arr.size:
            continue
        if agg == "max":
            out[dst] = float(arr.max())
        else:
            out[dst] = float(arr.mean())
    out["has_peak_margin_health"] = all(
        key in data.files
        for key in (
            "cem_peak_margin_valid_frac",
            "sample_peak_margin_ee_peak_mean",
            "sample_peak_margin_anchor_peak_mean",
        )
    )
    return out


def evaluate_one(
    row: dict[str, str],
    qpos_path: Path,
    video_path: Path,
    method: str,
    source_exp: str,
    cfg: EvalConfig,
) -> dict[str, Any] | None:
    scene = repo_path(row["rubber_scene_act"])
    if not qpos_path.is_file() or not scene.is_file():
        return None
    eval_row = dict(row)
    eval_row["method"] = method
    item = evaluate_sequence(
        row=eval_row,
        method=method,
        hand_collision_variant_id="rubber_hull",
        qpos_path=qpos_path,
        scene_xml=scene,
        config=cfg,
        kin_ref_path=kin_ref_for_scene(scene),
        contact_mask_path=contact_mask_for_case(row["short_case_id"]),
        person_idx=person_idx_from_case(row["short_case_id"]),
    )
    add_success_flags(item, cfg)
    return {
        **item,
        "short_case_id": row["short_case_id"],
        "variant": row["variant"] if source_exp == "E165D" else row["base_e163_variant"],
        "method": method,
        "method_group": row["method_group"] if source_exp == "E165D" else "narrowSurfaceBand",
        "source_exp": source_exp,
        "split": row["split"],
        "result_npz": rel(qpos_path),
        "video": rel(video_path) if video_path.is_file() else "",
        **(run_health(qpos_path) if source_exp == "E165D" else {}),
    }


def summarize(method: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "method": method,
        "n_cases": len(rows),
        "success_tracked_cases": sum(1 for row in rows if row.get("success_tracked")),
        "fall_cases": sum(1 for row in rows if row.get("fall_flag")),
    }
    for key in METRIC_KEYS:
        if key == "success_tracked":
            out[f"{key}_mean"] = mean([1.0 if row.get(key) else 0.0 for row in rows])
        elif key == "fall_flag":
            out[f"{key}_mean"] = mean([1.0 if row.get(key) else 0.0 for row in rows])
        else:
            out[f"{key}_mean"] = mean([row.get(key) for row in rows])
    for key in [
        "peak_margin_valid_frac",
        "peak_margin_selected_valid_frac",
        "peak_margin_fallback_used",
        "peak_margin_ee_peak_mean",
        "peak_margin_ee_peak_max",
        "peak_margin_anchor_peak_mean",
        "peak_margin_anchor_peak_max",
        "peak_margin_violation_mean",
    ]:
        out[f"{key}_mean"] = mean([row.get(key) for row in rows])
    return out


def xlsx_value(value: Any) -> Any:
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def append_table(
    wb: Workbook,
    title: str,
    rows: list[dict[str, Any]],
    columns: list[tuple[str, str, str]],
    fill: str,
    *,
    hidden: bool = False,
) -> None:
    if len(wb.sheetnames) == 1 and wb.active.max_row == 1 and wb.active["A1"].value is None:
        ws = wb.active
        ws.delete_rows(1)
    else:
        ws = wb.create_sheet(title)
    ws.title = title
    ws.append([header for header, _, _ in columns])
    for row in rows:
        ws.append([xlsx_value(row.get(key, "")) for _, key, _ in columns])

    for cell in ws[1]:
        cell.font = Font(name="Arial", bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor=fill)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for row_cells in ws.iter_rows(min_row=2):
        for cell in row_cells:
            cell.font = Font(name="Arial", size=10)
            cell.alignment = Alignment(vertical="center", wrap_text=True)
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    for col_idx, (_, _, fmt) in enumerate(columns, start=1):
        letter = get_column_letter(col_idx)
        for cell in ws[letter][1:]:
            if fmt == "pct":
                cell.number_format = "0.0%"
            elif fmt == "pct_delta":
                cell.number_format = "0.0%;[Red]-0.0%"
            elif fmt == "num":
                cell.number_format = "0.00"
            elif fmt == "m":
                cell.number_format = "0.000"
        width = min(max(len(str(cell.value or "")) for cell in ws[letter]) + 2, 48)
        ws.column_dimensions[letter].width = max(width, 10)
    if hidden:
        ws.sheet_state = "hidden"


def case_sort_key(case_id: str) -> int:
    try:
        return TARGET_CASES.index(case_id)
    except ValueError:
        return len(TARGET_CASES)


def case_probe_key(short_case_id: str) -> str:
    if short_case_id.startswith("box023"):
        return "box023"
    if short_case_id.startswith("box021"):
        return "box021"
    if short_case_id.startswith("box004"):
        return "box004"
    return short_case_id.split("_", 1)[0]


def load_isaac_probe_rows() -> list[dict[str, Any]]:
    if not ISAAC_PROBE_JSON.is_file():
        return []
    data = json.loads(ISAAC_PROBE_JSON.read_text(encoding="utf-8"))
    cases = data.get("cases", {}) if isinstance(data, dict) else {}
    rows: list[dict[str, Any]] = []
    for short_case_id in TARGET_CASES:
        key = case_probe_key(short_case_id)
        item = cases.get(key, {})
        row = {"case": short_case_id, "probe_case": key}
        if isinstance(item, dict):
            row.update(item)
        rows.append(row)
    return rows


def method_sort_key(method: str) -> int:
    try:
        return FULL_METHOD_ORDER.index(method)
    except ValueError:
        return len(FULL_METHOD_ORDER)


def reference_summary_rows() -> list[dict[str, Any]]:
    if not E163_SUMMARY.is_file():
        return []
    rows = []
    for row in read_tsv(E163_SUMMARY):
        if row.get("方法") not in REFERENCE_METHODS:
            continue
        item = convert_tsv_row(row)
        item["method"] = item.get("方法", "")
        rows.append(item)
    return sorted(rows, key=lambda row: method_sort_key(str(row.get("方法", ""))))


def reference_case_rows() -> list[dict[str, Any]]:
    if not E163_CASE_STATUS.is_file():
        return []
    rows = []
    for row in read_tsv(E163_CASE_STATUS):
        if row.get("case") not in TARGET_CASES or row.get("方法") not in REFERENCE_METHODS:
            continue
        item = convert_tsv_row(row)
        item["method"] = item.get("方法", "")
        item["tracked"] = item.get("success_tracked", "")
        item["fall"] = item.get("fall_flag", "")
        rows.append(item)
    return sorted(rows, key=lambda row: (case_sort_key(str(row.get("case", ""))), method_sort_key(str(row.get("方法", "")))))


def rubberhand_by_case() -> dict[str, dict[str, Any]]:
    return {
        str(row.get("case")): row
        for row in reference_case_rows()
        if row.get("方法") == "SPIDER+rubberhand"
    }


def e163_narrow_by_case(metric_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("short_case_id")): row for row in metric_rows if row.get("source_exp") == "E163"}


def omni_by_case() -> dict[str, dict[str, Any]]:
    return {
        str(row.get("case")): row
        for row in reference_case_rows()
        if row.get("方法") == "OmniRetarget"
    }


def build_e165d_case_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rubber = rubberhand_by_case()
    e163 = e163_narrow_by_case(metric_rows)
    omni = omni_by_case()
    out: list[dict[str, Any]] = []
    delta_keys = [
        "hand_object_physics_contact_in_mask_frac",
        "hand_object_physics_contact_3mm_in_mask_frac",
        "hand_object_physics_contact_5mm_in_mask_frac",
        "hand_object_physics_penetration_3mm_frame_frac",
        "hand_object_physics_penetration_5mm_frame_frac",
        "hand_geom_penetration_2mm_frac",
        "hand_geom_penetration_5mm_frac",
        "hand_object_release_false_contact_3mm_frac",
        "track_eef_pos_err_cm_mean",
        "track_eef_ori_err_deg_mean",
        "track_joint_err_deg_mean",
        "track_obj_pos_err_cm_mean",
        "track_obj_ori_err_deg_mean",
        "track_root_pos_err_cm_mean",
        "track_root_ori_err_deg_mean",
        "track_pelvis_z_err_terminal_m",
    ]
    for row in metric_rows:
        if row.get("source_exp") != "E165D":
            continue
        case_id = str(row.get("short_case_id"))
        base = e163.get(case_id, {})
        omni_row = omni.get(case_id, {})
        rubber_row = rubber.get(case_id, {})
        item = dict(row)
        item["case"] = case_id
        item["方法"] = "E165D peakMargin025"
        item["method"] = "E165D peakMargin025"
        item["tracked"] = bool(row.get("success_tracked"))
        item["fall"] = bool(row.get("fall_flag"))
        item["Table4完整"] = all(
            math.isfinite(finite(row.get(key)))
            for key in (
                "track_joint_err_deg_mean",
                "track_eef_pos_err_cm_mean",
                "track_eef_ori_err_deg_mean",
                "track_obj_pos_err_cm_mean",
                "track_obj_ori_err_deg_mean",
            )
        )
        item["raw接触下限"] = rubber_row.get("raw接触下限", "")
        item["source/method_id"] = "gateA+surfaceBand-A2+postureRerankA+narrowSurfaceBandReleaseDecay+peakMargin025"
        rubber_raw = finite(rubber_row.get("hand_object_physics_contact_in_mask_frac"))
        if math.isfinite(rubber_raw):
            item["物理接触Δ_vs_rubberhand"] = finite(row.get("hand_object_physics_contact_in_mask_frac")) - rubber_raw
        for key in delta_keys:
            item[f"{key}_delta_vs_E163"] = finite(row.get(key)) - finite(base.get(key)) if base else math.nan
            item[f"{key}_delta_vs_OmniRetarget"] = finite(row.get(key)) - finite(omni_row.get(key)) if omni_row else math.nan
        raw_value = finite(row.get("hand_object_physics_contact_in_mask_frac"))
        raw_floor = finite(item.get("raw接触下限"))
        if not bool(row.get("success_tracked")):
            item["状态"] = "tracking失败"
        elif bool(row.get("fall_flag")):
            item["状态"] = "摔倒"
        elif math.isfinite(raw_floor) and raw_value < raw_floor:
            item["状态"] = "接触退化"
        else:
            item["状态"] = "通过"
        raw_delta = item.get("hand_object_physics_contact_in_mask_frac_delta_vs_E163", math.nan)
        clean_delta = item.get("hand_object_physics_contact_3mm_in_mask_frac_delta_vs_E163", math.nan)
        pen_delta = item.get("hand_object_physics_penetration_3mm_frame_frac_delta_vs_E163", math.nan)
        risks = []
        if math.isfinite(raw_delta) and raw_delta < -0.05:
            risks.append("vs_E163 raw接触回归")
        if math.isfinite(clean_delta) and clean_delta < -0.05:
            risks.append("vs_E163 clean3下降")
        if math.isfinite(pen_delta) and pen_delta > 0.05:
            risks.append("vs_E163 物理穿透上升")
        item["说明"] = ",".join(risks)
        out.append(item)
    return sorted(out, key=lambda row: case_sort_key(str(row.get("case", ""))))


def build_compare_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = reference_case_rows()
    rows.extend(build_e165d_case_rows(metric_rows))
    for item in rows:
        method = str(item.get("方法", item.get("method", "")))
        case_id = str(item.get("case", ""))
        if method != "E165D peakMargin025":
            for key in [
                "hand_object_physics_contact_in_mask_frac",
                "hand_object_physics_contact_3mm_in_mask_frac",
                "hand_object_physics_contact_5mm_in_mask_frac",
                "hand_object_physics_penetration_3mm_frame_frac",
                "hand_object_physics_penetration_5mm_frame_frac",
                "hand_geom_penetration_2mm_frac",
                "hand_geom_penetration_5mm_frac",
                "hand_object_release_false_contact_3mm_frac",
                "track_eef_pos_err_cm_mean",
                "track_eef_ori_err_deg_mean",
                "track_joint_err_deg_mean",
                "track_obj_pos_err_cm_mean",
                "track_obj_ori_err_deg_mean",
                "track_root_pos_err_cm_mean",
                "track_root_ori_err_deg_mean",
                "track_pelvis_z_err_terminal_m",
            ]:
                item.setdefault(f"{key}_delta_vs_E163", "")
                item.setdefault(f"{key}_delta_vs_OmniRetarget", "")
            item.setdefault("tracked", item.get("success_tracked", ""))
            item.setdefault("fall", item.get("fall_flag", ""))
        item["method"] = method
        item["case"] = case_id
    return sorted(rows, key=lambda row: (case_sort_key(str(row.get("case", ""))), method_sort_key(str(row.get("method", "")))))


def build_main_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = reference_summary_rows()
    e165d_rows = [row for row in metric_rows if row.get("source_exp") == "E165D"]
    if e165d_rows:
        case_rows = build_e165d_case_rows(metric_rows)
        item: dict[str, Any] = {
            "方法": "E165D peakMargin025",
            "method": "E165D peakMargin025",
            "case数": len(e165d_rows),
            "通过case": sum(1 for row in case_rows if row.get("状态") == "通过"),
            "失败case": ",".join(str(row.get("case")) for row in case_rows if row.get("状态") != "通过"),
            "接触退化case": ",".join(str(row.get("case")) for row in case_rows if row.get("状态") == "接触退化"),
            "说明": "E163 narrow 基础上新增 peakMargin025。",
        }
        for key in [
            "hand_object_physics_contact_in_mask_frac",
            "hand_object_physics_contact_in_rl_mask_frac",
            "hand_object_physics_contact_3mm_in_mask_frac",
            "hand_object_physics_contact_5mm_in_mask_frac",
            "hand_object_physics_penetration_3mm_frame_frac",
            "hand_object_physics_penetration_5mm_frame_frac",
            "hand_geom_penetration_2mm_frac",
            "hand_geom_penetration_5mm_frac",
            "hand_object_release_false_contact_3mm_frac",
            "hand_object_release_false_contact_5mm_frac",
            "track_joint_err_deg_mean",
            "track_eef_pos_err_cm_mean",
            "track_eef_ori_err_deg_mean",
            "track_obj_pos_err_cm_mean",
            "track_obj_ori_err_deg_mean",
            "track_root_pos_err_cm_mean",
            "track_root_ori_err_deg_mean",
        ]:
            item[f"{key}_mean"] = mean([row.get(key) for row in e165d_rows])
        rows.append(item)
    return sorted(rows, key=lambda row: method_sort_key(str(row.get("方法", ""))))


def write_workbook(
    eval_dir: Path,
    summary_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    artifact_rows: list[dict[str, Any]],
) -> Path:
    wb = Workbook()
    main_cols = [
        ("方法", "方法", "text"),
        ("case数", "case数", "int"),
        ("通过case", "通过case", "int"),
        ("失败case", "失败case", "text"),
        ("接触退化case", "接触退化case", "text"),
        ("物理接触(raw)", "hand_object_physics_contact_in_mask_frac_mean", "pct"),
        ("clean3接触", "hand_object_physics_contact_3mm_in_mask_frac_mean", "pct"),
        ("clean5接触", "hand_object_physics_contact_5mm_in_mask_frac_mean", "pct"),
        ("物理穿透3mm", "hand_object_physics_penetration_3mm_frame_frac_mean", "pct"),
        ("物理穿透5mm", "hand_object_physics_penetration_5mm_frame_frac_mean", "pct"),
        ("几何穿透2mm", "hand_geom_penetration_2mm_frac_mean", "pct"),
        ("几何穿透5mm", "hand_geom_penetration_5mm_frac_mean", "pct"),
        ("release误接触3mm", "hand_object_release_false_contact_3mm_frac_mean", "pct"),
        ("关节误差(°)", "track_joint_err_deg_mean_mean", "num"),
        ("EEF误差(cm)", "track_eef_pos_err_cm_mean_mean", "num"),
        ("EEF朝向误差(°)", "track_eef_ori_err_deg_mean_mean", "num"),
        ("物体位置误差(cm)", "track_obj_pos_err_cm_mean_mean", "num"),
        ("物体朝向误差(°)", "track_obj_ori_err_deg_mean_mean", "num"),
        ("root误差(cm)", "track_root_pos_err_cm_mean_mean", "num"),
        ("root朝向误差(°)", "track_root_ori_err_deg_mean_mean", "num"),
        ("说明", "说明", "text"),
    ]
    append_table(wb, "主表", build_main_rows(metric_rows), main_cols, "1F4E79")

    per_case_cols = [
        ("方法", "method", "text"),
        ("case", "case", "text"),
        ("状态", "状态", "text"),
        ("raw接触下限", "raw接触下限", "pct"),
        ("raw接触", "hand_object_physics_contact_in_mask_frac", "pct"),
        ("rawΔ_E165D_vs_E163", "hand_object_physics_contact_in_mask_frac_delta_vs_E163", "pct_delta"),
        ("rawΔ_E165D_vs_Omni", "hand_object_physics_contact_in_mask_frac_delta_vs_OmniRetarget", "pct_delta"),
        ("clean3接触", "hand_object_physics_contact_3mm_in_mask_frac", "pct"),
        ("clean3Δ_E165D_vs_E163", "hand_object_physics_contact_3mm_in_mask_frac_delta_vs_E163", "pct_delta"),
        ("clean3Δ_E165D_vs_Omni", "hand_object_physics_contact_3mm_in_mask_frac_delta_vs_OmniRetarget", "pct_delta"),
        ("clean5接触", "hand_object_physics_contact_5mm_in_mask_frac", "pct"),
        ("物理穿透3mm", "hand_object_physics_penetration_3mm_frame_frac", "pct"),
        ("物理穿透3mmΔ_E165D_vs_E163", "hand_object_physics_penetration_3mm_frame_frac_delta_vs_E163", "pct_delta"),
        ("物理穿透3mmΔ_E165D_vs_Omni", "hand_object_physics_penetration_3mm_frame_frac_delta_vs_OmniRetarget", "pct_delta"),
        ("物理穿透5mm", "hand_object_physics_penetration_5mm_frame_frac", "pct"),
        ("几何穿透2mm", "hand_geom_penetration_2mm_frac", "pct"),
        ("几何穿透2mmΔ_E165D_vs_E163", "hand_geom_penetration_2mm_frac_delta_vs_E163", "pct_delta"),
        ("几何穿透2mmΔ_E165D_vs_Omni", "hand_geom_penetration_2mm_frac_delta_vs_OmniRetarget", "pct_delta"),
        ("release误接触3mm", "hand_object_release_false_contact_3mm_frac", "pct"),
        ("tracked", "tracked", "text"),
        ("fall", "fall", "text"),
        ("Table4完整", "Table4完整", "text"),
        ("关节误差(°)", "track_joint_err_deg_mean", "num"),
        ("关节误差Δ_E165D_vs_E163", "track_joint_err_deg_mean_delta_vs_E163", "num"),
        ("关节误差Δ_E165D_vs_Omni", "track_joint_err_deg_mean_delta_vs_OmniRetarget", "num"),
        ("EEF误差(cm)", "track_eef_pos_err_cm_mean", "num"),
        ("EEF误差Δ_E165D_vs_E163", "track_eef_pos_err_cm_mean_delta_vs_E163", "num"),
        ("EEF误差Δ_E165D_vs_Omni", "track_eef_pos_err_cm_mean_delta_vs_OmniRetarget", "num"),
        ("EEF朝向误差(°)", "track_eef_ori_err_deg_mean", "num"),
        ("EEF朝向Δ_E165D_vs_E163", "track_eef_ori_err_deg_mean_delta_vs_E163", "num"),
        ("EEF朝向Δ_E165D_vs_Omni", "track_eef_ori_err_deg_mean_delta_vs_OmniRetarget", "num"),
        ("物体位置误差(cm)", "track_obj_pos_err_cm_mean", "num"),
        ("物体位置Δ_E165D_vs_E163", "track_obj_pos_err_cm_mean_delta_vs_E163", "num"),
        ("物体位置Δ_E165D_vs_Omni", "track_obj_pos_err_cm_mean_delta_vs_OmniRetarget", "num"),
        ("物体朝向误差(°)", "track_obj_ori_err_deg_mean", "num"),
        ("物体朝向Δ_E165D_vs_E163", "track_obj_ori_err_deg_mean_delta_vs_E163", "num"),
        ("物体朝向Δ_E165D_vs_Omni", "track_obj_ori_err_deg_mean_delta_vs_OmniRetarget", "num"),
        ("root误差(cm)", "track_root_pos_err_cm_mean", "num"),
        ("root误差Δ_E165D_vs_E163", "track_root_pos_err_cm_mean_delta_vs_E163", "num"),
        ("root误差Δ_E165D_vs_Omni", "track_root_pos_err_cm_mean_delta_vs_OmniRetarget", "num"),
        ("root朝向误差(°)", "track_root_ori_err_deg_mean", "num"),
        ("root朝向Δ_E165D_vs_E163", "track_root_ori_err_deg_mean_delta_vs_E163", "num"),
        ("root朝向Δ_E165D_vs_Omni", "track_root_ori_err_deg_mean_delta_vs_OmniRetarget", "num"),
        ("root-z terminal(m)", "track_pelvis_z_err_terminal_m", "m"),
        ("root-zΔ_E165D_vs_E163", "track_pelvis_z_err_terminal_m_delta_vs_E163", "m"),
        ("source/method_id", "source/method_id", "text"),
        ("说明", "说明", "text"),
    ]
    append_table(wb, "逐case", build_compare_rows(metric_rows), per_case_cols, "548235")

    health_cols = [
        ("case", "short_case_id", "text"),
        ("valid frac", "peak_margin_valid_frac", "pct"),
        ("selected valid frac", "peak_margin_selected_valid_frac", "pct"),
        ("fallback used", "peak_margin_fallback_used", "pct"),
        ("ee peak mean(m)", "peak_margin_ee_peak_mean", "m"),
        ("ee peak max(m)", "peak_margin_ee_peak_max", "m"),
        ("anchor peak mean(m)", "peak_margin_anchor_peak_mean", "m"),
        ("anchor peak max(m)", "peak_margin_anchor_peak_max", "m"),
        ("violation mean", "peak_margin_violation_mean", "num"),
        ("posture valid frac", "posture_gate_valid_frac", "pct"),
        ("posture fallback used", "posture_gate_fallback_used", "pct"),
        ("has health", "has_peak_margin_health", "text"),
    ]
    append_table(
        wb,
        "PeakMargin健康度",
        [row for row in metric_rows if row.get("source_exp") == "E165D"],
        health_cols,
        "7030A0",
    )

    isaac_cols = [
        ("case", "case", "text"),
        ("probe_case", "probe_case", "text"),
        ("frames", "n_frames", "int"),
        ("source contact frames", "n_source_contact", "int"),
        ("filtered recall either", "filtered_contact_recall_either", "pct"),
        ("filtered recall both", "filtered_contact_recall_both", "pct"),
        ("phantom force rate", "phantom_force_rate", "pct"),
        ("max init net force(N)", "max_init_net_force", "num"),
        ("max net force(N)", "max_net_force", "num"),
        ("mean filtered force contact(N)", "mean_filtered_force_contact", "num"),
        ("E163 staggered success", "staggered_success", "pct"),
        ("mode", "mode", "text"),
    ]
    append_table(wb, "Isaac on-rails", load_isaac_probe_rows(), isaac_cols, "C65911")

    artifact_cols = [
        ("variant", "variant", "text"),
        ("case", "short_case_id", "text"),
        ("artifact_ok", "artifact_ok", "text"),
        ("root_npz", "root_npz_exists", "text"),
        ("trajectory", "trajectory_mjwp_act_exists", "text"),
        ("config", "config_act_exists", "text"),
        ("mp4", "mp4_exists", "text"),
        ("peak enabled", "config_peak_margin_enabled_ok", "text"),
        ("ee threshold", "config_ee_threshold_ok", "text"),
        ("anchor threshold", "config_anchor_threshold_ok", "text"),
        ("buffer", "config_buffer_ok", "text"),
        ("config_path", "config_path", "text"),
    ]
    append_table(wb, "E165D产物检查", artifact_rows, artifact_cols, "8064A2")

    info_rows = [
        {"项": "范围", "说明": "仅比较本轮实际运行的 3 个 case：box023_person2、box021_029_p2、box004_083_p2。"},
        {"项": "baseline", "说明": "E165D 基于 E163 narrowSurfaceBand 配置栈重新跑 CEM；xlsx 中 delta 均为 E165D - E163。"},
        {"项": "E165D", "说明": "peakMargin025 接入 CEM sample elite selection，阈值 ee/anchor=0.25m，buffer=0.03m。"},
        {"项": "Isaac on-rails", "说明": "来自 scripts/eval/runners/eval_E165_isaac_onrails_probe.py；三标量用于诊断接触标签虚高、phantom force、初始穿透。"},
        {"项": "判读", "说明": "raw/clean 接触下降和穿透上升是风险信号；tracked/fall 不能单独证明 RL-safe。"},
    ]
    append_table(wb, "说明", info_rows, [("项", "项", "text"), ("说明", "说明", "text")], "7F6000")

    raw_cols = [(key, key, "text") for key in sorted({key for row in metric_rows for key in row})]
    append_table(wb, "原始metrics", metric_rows, raw_cols, "666666", hidden=True)

    out_path = eval_dir / "E165D_peak_margin_vs_E163_three_case_eval.xlsx"
    wb.save(out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", nargs="?", default="full", choices=["smoke", "full"])
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    rows = [row for row in read_tsv(VARIANTS) if row["short_case_id"] in TARGET_CASES]
    eval_dir = RESULT_ROOT / "eval" / args.stage
    eval_dir.mkdir(parents=True, exist_ok=True)
    cfg = EvalConfig()

    metric_rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []

    for row in rows:
        root_npz, outdir_npz, _, video = stage_paths(row, args.stage)
        artifact = artifact_check(row, args.stage)
        artifact_rows.append(artifact)
        qpos_path = outdir_npz if outdir_npz.is_file() else root_npz
        if not qpos_path.is_file():
            missing_rows.append({"variant": row["variant"], "short_case_id": row["short_case_id"], "reason": "missing_e165d_qpos"})
        else:
            item = evaluate_one(row, qpos_path, video, "E165D peakMargin025", "E165D", cfg)
            if item is None:
                missing_rows.append({"variant": row["variant"], "short_case_id": row["short_case_id"], "reason": "eval_e165d_failed"})
            else:
                item["artifact_ok"] = artifact["artifact_ok"]
                item["config_act"] = artifact["config_path"]
                metric_rows.append(item)

        baseline_qpos = repo_path(row["baseline_outdir_npz"])
        baseline_video = repo_path(row["baseline_video"])
        base_item = evaluate_one(row, baseline_qpos, baseline_video, "E163 narrowSurfaceBand", "E163", cfg)
        if base_item is not None:
            metric_rows.append(base_item)
        else:
            missing_rows.append({"variant": row["base_e163_variant"], "short_case_id": row["short_case_id"], "reason": "eval_e163_baseline_failed"})

    summary_rows = [
        summarize(method, [row for row in metric_rows if row.get("method") == method])
        for method in ["E163 narrowSurfaceBand", "E165D peakMargin025"]
    ]
    xlsx = write_workbook(eval_dir, summary_rows, metric_rows, artifact_rows)

    write_tsv(eval_dir / "e165d_method_metrics.tsv", metric_rows, sorted({k for row in metric_rows for k in row}))
    write_tsv(eval_dir / "e165d_artifact_checks.tsv", artifact_rows, sorted({k for row in artifact_rows for k in row}))
    write_tsv(eval_dir / "e165d_missing.tsv", missing_rows, sorted({k for row in missing_rows for k in row}) if missing_rows else ["variant", "short_case_id", "reason"])
    write_tsv(eval_dir / "e165d_method_summary.tsv", summary_rows, sorted({k for row in summary_rows for k in row}))
    write_json(
        eval_dir / "e165d_eval_summary.json",
        {
            "stage": args.stage,
            "metric_rows": len(metric_rows),
            "artifact_rows": len(artifact_rows),
            "missing": len(missing_rows),
            "e165d_rows": sum(1 for row in metric_rows if row.get("source_exp") == "E165D"),
            "e163_baseline_rows": sum(1 for row in metric_rows if row.get("source_exp") == "E163"),
            "all_artifacts_ok": all(bool(row.get("artifact_ok")) for row in artifact_rows),
            "xlsx": rel(xlsx),
            "summary": summary_rows,
        },
    )
    print(
        "E165D eval: "
        f"stage={args.stage} metric_rows={len(metric_rows)} "
        f"missing={len(missing_rows)} all_artifacts_ok={all(bool(row.get('artifact_ok')) for row in artifact_rows)} "
        f"xlsx={rel(xlsx)}"
    )
    if missing_rows and not args.allow_missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
