#!/usr/bin/env python3
"""Build E161 surfaceBand release ablation clean8 manifest and overrides."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[5]
SCRIPT_ROOT = REPO / "workspace/core4d/scripts/experiments/E161"
RESULT_ROOT = REPO / "workspace/core4d/results/E161/surface_release_ablation"
CEM_ROOT = RESULT_ROOT / "cem/full"
PREFLIGHT_ROOT = RESULT_ROOT / "preflight"
OVERRIDE_ROOT = REPO / "examples/config/override"
E156_VARIANTS = REPO / "workspace/core4d/scripts/experiments/E156/variants.tsv"
E160_CEM_ROOT = REPO / "workspace/core4d/results/E160/posture_rerank/cem/full"

CLEAN8_CASES = [
    "box021_035_p1",
    "box021_035_p2",
    "box021_029_p2",
    "box004_083_p1",
    "box004_083_p2",
    "box023_person2",
    "box004_082_p1",
    "box026_139_p1",
]
E160_REUSE_M0 = {"box021_029_p2", "box004_083_p2", "box023_person2"}

METHODS = [
    ("postureRerankA", "gateA+surfaceBand-A2+postureRerankA"),
    ("releaseDecay", "gateA+surfaceBand-A2+postureRerankA+surfaceBandReleaseDecay"),
    ("strictMask", "gateA+surfaceBand-A2+postureRerankA+surfaceBandStrictMask"),
]

SPLITS = {
    ("postureRerankA", "box021_035_p1"): "local-gpu0",
    ("postureRerankA", "box004_082_p1"): "local-gpu0",
    ("releaseDecay", "box021_029_p2"): "local-gpu0",
    ("releaseDecay", "box023_person2"): "local-gpu0",
    ("releaseDecay", "box026_139_p1"): "local-gpu0",
    ("strictMask", "box021_035_p2"): "local-gpu0",
    ("strictMask", "box004_083_p2"): "local-gpu0",
    ("postureRerankA", "box021_035_p2"): "remote-gpu0",
    ("postureRerankA", "box026_139_p1"): "remote-gpu0",
    ("releaseDecay", "box021_035_p1"): "remote-gpu0",
    ("releaseDecay", "box004_083_p1"): "remote-gpu0",
    ("releaseDecay", "box004_082_p1"): "remote-gpu0",
    ("strictMask", "box021_029_p2"): "remote-gpu0",
    ("strictMask", "box023_person2"): "remote-gpu0",
    ("postureRerankA", "box004_083_p1"): "remote-gpu1",
    ("releaseDecay", "box021_035_p2"): "remote-gpu1",
    ("releaseDecay", "box004_083_p2"): "remote-gpu1",
    ("strictMask", "box021_035_p1"): "remote-gpu1",
    ("strictMask", "box004_083_p1"): "remote-gpu1",
    ("strictMask", "box004_082_p1"): "remote-gpu1",
    ("strictMask", "box026_139_p1"): "remote-gpu1",
}

GATE_MIN_SDF_M = -0.010
GATE_MAX_VIOL = 0.10
GATE_HARD_FLOOR_M = -0.020
SURFACE_REW_SCALE = 1.5
SURFACE_PENALTY_SCALE = 0.0
SURFACE_WIDTH_M = 0.030
SURFACE_MIN_SDF_M = -0.001
SURFACE_SIGMA = 0.015
SURFACE_PEN_TOL_M = 0.003
SURFACE_DECAY_FRAC = 0.15
POSTURE_MEAN_Z_ERR_M = 0.10
POSTURE_TERMINAL_Z_ERR_M = 0.12
POSTURE_MAX_Z_DROP_M = 0.18
POSTURE_TERMINAL_FRAC = 0.15
POSTURE_MIN_VALID_FRAC = 0.05
POSTURE_FALLBACK_LAMBDA = 5.0

FIELDS = [
    "ordinal",
    "variant",
    "short_case_id",
    "method",
    "method_display",
    "method_group",
    "run_status",
    "source_exp",
    "split",
    "e156_gate_variant",
    "e148_variant",
    "e143_variant",
    "e109_case_id",
    "case_id",
    "object_key",
    "object_category",
    "person_idx",
    "derived_task",
    "target_scene",
    "trajectory",
    "base_scene_act",
    "rubber_scene_act",
    "scene_name",
    "override",
    "object_asset",
    "mask_path",
    "baseline_variant",
    "baseline_npz",
    "baseline_outdir_npz",
    "baseline_video",
    "cem_hand_gate_min_sdf_m",
    "cem_hand_gate_max_violation_pct",
    "cem_hand_gate_hard_floor_m",
    "surface_band_rew_scale",
    "surface_band_penalty_scale",
    "surface_band_width_m",
    "surface_band_min_sdf_m",
    "surface_band_sigma",
    "surface_band_penetration_tol_m",
    "surface_band_gate_source",
    "surface_band_decay_frac",
    "cem_posture_gate_mean_z_err_m",
    "cem_posture_gate_terminal_z_err_m",
    "cem_posture_gate_max_z_drop_m",
    "cem_posture_gate_terminal_frac",
    "cem_posture_gate_min_valid_frac",
    "cem_posture_gate_fallback_lambda",
    "result_npz",
    "outdir_npz",
    "video",
    "expected_quality",
    "remote_sync_key",
]

PREFLIGHT_FIELDS = [
    "short_case_id",
    "method_group",
    "variant",
    "run_status",
    "split",
    "override_exists",
    "task_dir_exists",
    "object_asset_exists",
    "mask_exists",
    "baseline_npz_exists",
    "baseline_outdir_exists",
    "baseline_video_exists",
    "result_npz_exists",
    "outdir_npz_exists",
    "video_exists",
    "ok",
]


def rel(path: str | Path) -> str:
    if not path:
        return ""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO.resolve()))
    except Exception:
        return str(path)


def repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO / p


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field, "") for field in fields})


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def variant_for(case_id: str, group: str) -> str:
    return f"E161_{case_id}_{group}"


def result_paths(case_id: str, group: str) -> tuple[str, str, str]:
    if group == "postureRerankA" and case_id in E160_REUSE_M0:
        variant = f"E160_{case_id}_gateA_postureRerankA"
        return (
            rel(E160_CEM_ROOT / f"{variant}.npz"),
            rel(E160_CEM_ROOT / f"{variant}_outdir_full/trajectory_mjwp_act.npz"),
            rel(E160_CEM_ROOT / f"{variant}_full.mp4"),
        )
    variant = variant_for(case_id, group)
    return (
        rel(CEM_ROOT / f"{variant}.npz"),
        rel(CEM_ROOT / f"{variant}_outdir_full/trajectory_mjwp_act.npz"),
        rel(CEM_ROOT / f"{variant}_full.mp4"),
    )


def gate_source_for(group: str) -> str:
    return "contact_mask_strict_current" if group == "strictMask" else "contact_mask"


def decay_frac_for(group: str) -> float:
    return SURFACE_DECAY_FRAC if group == "releaseDecay" else 0.0


def write_override(row: dict[str, Any], base_override: str) -> str:
    path = OVERRIDE_ROOT / f"core4d_{row['variant']}.yaml"
    base_stem = Path(base_override).stem
    lines = [
        "# @package _global_",
        "# Auto-generated by workspace/core4d/scripts/experiments/E161/build_surface_release_ablation_manifest.py.",
        f"# E161 {row['method_group']} case={row['short_case_id']}.",
        "defaults:",
        f"  - {base_stem}",
        "  - _self_",
        "",
        f"task: {row['derived_task']}",
        f"scene_name: {row['scene_name']}",
        "video_camera: auto",
        "",
        "cem_hand_gate_enabled: true",
        'cem_hand_gate_geom_names: ["lh", "rh"]',
        "cem_hand_gate_geom_ids: []",
        f"cem_hand_gate_min_sdf_m: {GATE_MIN_SDF_M:.6f}",
        f"cem_hand_gate_max_violation_pct: {GATE_MAX_VIOL:.6f}",
        f"cem_hand_gate_hard_floor_m: {GATE_HARD_FLOOR_M:.6f}",
        "",
        f"surface_band_rew_scale: {SURFACE_REW_SCALE:.6f}",
        f"surface_band_penalty_scale: {SURFACE_PENALTY_SCALE:.6f}",
        f"surface_band_width_m: {SURFACE_WIDTH_M:.6f}",
        f"surface_band_min_sdf_m: {SURFACE_MIN_SDF_M:.6f}",
        f"surface_band_sigma: {SURFACE_SIGMA:.6f}",
        f"surface_band_penetration_tol_m: {SURFACE_PEN_TOL_M:.6f}",
        f"surface_band_gate_source: {row['surface_band_gate_source']}",
        f"surface_band_decay_frac: {float(row['surface_band_decay_frac']):.6f}",
        "surface_band_start_eval_time: 0.0",
        "surface_band_end_eval_time: 0.0",
        'surface_band_geom_names: ["lh", "rh"]',
        "surface_band_geom_ids: []",
        "",
        "cem_posture_gate_enabled: true",
        f"cem_posture_gate_mean_z_err_m: {POSTURE_MEAN_Z_ERR_M:.6f}",
        f"cem_posture_gate_terminal_z_err_m: {POSTURE_TERMINAL_Z_ERR_M:.6f}",
        f"cem_posture_gate_max_z_drop_m: {POSTURE_MAX_Z_DROP_M:.6f}",
        f"cem_posture_gate_terminal_frac: {POSTURE_TERMINAL_FRAC:.6f}",
        f"cem_posture_gate_min_valid_frac: {POSTURE_MIN_VALID_FRAC:.6f}",
        f"cem_posture_gate_fallback_lambda: {POSTURE_FALLBACK_LAMBDA:.6f}",
        "",
        "hand_support_rew_scale: 0.0",
        "hand_support_geom_ids: []",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rel(path)


def build_row(case_id: str, group: str, method: str, gate_row: dict[str, str], ordinal: int) -> dict[str, Any]:
    variant = variant_for(case_id, group)
    run_status = "to_run"
    source_exp = "E161"
    split = "reuse" if group == "postureRerankA" and case_id in E160_REUSE_M0 else SPLITS[(group, case_id)]
    if group == "postureRerankA" and case_id in E160_REUSE_M0:
        run_status = "reuse_e160"
        source_exp = "E160"
    result_npz, outdir_npz, video = result_paths(case_id, group)
    row: dict[str, Any] = {
        "ordinal": ordinal,
        "variant": variant,
        "short_case_id": case_id,
        "method": method,
        "method_display": method,
        "method_group": group,
        "run_status": run_status,
        "source_exp": source_exp,
        "split": split,
        "e156_gate_variant": gate_row["variant"],
        "e148_variant": gate_row["e148_variant"],
        "e143_variant": gate_row["e143_variant"],
        "e109_case_id": gate_row["e109_case_id"],
        "case_id": gate_row["case_id"],
        "object_key": gate_row["object_key"],
        "object_category": gate_row["object_category"],
        "person_idx": gate_row["person_idx"],
        "derived_task": gate_row["derived_task"],
        "target_scene": gate_row["target_scene"],
        "trajectory": gate_row["trajectory"],
        "base_scene_act": gate_row["base_scene_act"],
        "rubber_scene_act": gate_row["rubber_scene_act"],
        "scene_name": gate_row["scene_name"],
        "override": "",
        "object_asset": gate_row["object_asset"],
        "mask_path": gate_row["mask_path"],
        "baseline_variant": gate_row["baseline_variant"],
        "baseline_npz": gate_row["baseline_npz"],
        "baseline_outdir_npz": gate_row["baseline_outdir_npz"],
        "baseline_video": gate_row["baseline_video"],
        "cem_hand_gate_min_sdf_m": f"{GATE_MIN_SDF_M:.6f}",
        "cem_hand_gate_max_violation_pct": f"{GATE_MAX_VIOL:.6f}",
        "cem_hand_gate_hard_floor_m": f"{GATE_HARD_FLOOR_M:.6f}",
        "surface_band_rew_scale": f"{SURFACE_REW_SCALE:.6f}",
        "surface_band_penalty_scale": f"{SURFACE_PENALTY_SCALE:.6f}",
        "surface_band_width_m": f"{SURFACE_WIDTH_M:.6f}",
        "surface_band_min_sdf_m": f"{SURFACE_MIN_SDF_M:.6f}",
        "surface_band_sigma": f"{SURFACE_SIGMA:.6f}",
        "surface_band_penetration_tol_m": f"{SURFACE_PEN_TOL_M:.6f}",
        "surface_band_gate_source": gate_source_for(group),
        "surface_band_decay_frac": f"{decay_frac_for(group):.6f}",
        "cem_posture_gate_mean_z_err_m": f"{POSTURE_MEAN_Z_ERR_M:.6f}",
        "cem_posture_gate_terminal_z_err_m": f"{POSTURE_TERMINAL_Z_ERR_M:.6f}",
        "cem_posture_gate_max_z_drop_m": f"{POSTURE_MAX_Z_DROP_M:.6f}",
        "cem_posture_gate_terminal_frac": f"{POSTURE_TERMINAL_FRAC:.6f}",
        "cem_posture_gate_min_valid_frac": f"{POSTURE_MIN_VALID_FRAC:.6f}",
        "cem_posture_gate_fallback_lambda": f"{POSTURE_FALLBACK_LAMBDA:.6f}",
        "result_npz": result_npz,
        "outdir_npz": outdir_npz,
        "video": video,
        "expected_quality": gate_row["expected_quality"],
        "remote_sync_key": gate_row["remote_sync_key"],
    }
    row["override"] = write_override(row, gate_row["override"])
    return row


def preflight_row(row: dict[str, Any]) -> dict[str, Any]:
    task_dir = repo_path(row["rubber_scene_act"]).parent
    item = {
        "short_case_id": row["short_case_id"],
        "method_group": row["method_group"],
        "variant": row["variant"],
        "run_status": row["run_status"],
        "split": row["split"],
        "override_exists": repo_path(row["override"]).is_file(),
        "task_dir_exists": task_dir.is_dir(),
        "object_asset_exists": repo_path(row["object_asset"]).is_file(),
        "mask_exists": repo_path(row["mask_path"]).is_file(),
        "baseline_npz_exists": repo_path(row["baseline_npz"]).is_file(),
        "baseline_outdir_exists": repo_path(row["baseline_outdir_npz"]).is_file(),
        "baseline_video_exists": repo_path(row["baseline_video"]).is_file(),
        "result_npz_exists": repo_path(row["result_npz"]).is_file(),
        "outdir_npz_exists": repo_path(row["outdir_npz"]).is_file(),
        "video_exists": repo_path(row["video"]).is_file(),
    }
    required = [
        "override_exists",
        "task_dir_exists",
        "object_asset_exists",
        "mask_exists",
        "baseline_npz_exists",
        "baseline_outdir_exists",
        "baseline_video_exists",
    ]
    if row["run_status"] == "reuse_e160":
        required += ["result_npz_exists", "outdir_npz_exists", "video_exists"]
    item["ok"] = all(bool(item[key]) for key in required)
    return item


def build() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    e156_rows = read_tsv(E156_VARIANTS)
    by_gate = {(row["short_case_id"], row["method"]): row for row in e156_rows}
    missing = [case for case in CLEAN8_CASES if (case, "+gateA") not in by_gate]
    if missing:
        raise RuntimeError(f"E156 +gateA rows missing clean8 cases: {missing}")

    rows: list[dict[str, Any]] = []
    ordinal = 1
    for group, method in METHODS:
        for case in CLEAN8_CASES:
            rows.append(build_row(case, group, method, by_gate[(case, "+gateA")], ordinal))
            ordinal += 1
    preflight = [preflight_row(row) for row in rows]
    to_run = [row for row in rows if row["run_status"] == "to_run"]
    summary = {
        "experiment": "E161",
        "git_head": git_head(),
        "source_e156_variants": rel(E156_VARIANTS),
        "clean8_cases": CLEAN8_CASES,
        "method_rows": len(rows),
        "reuse_e160": sum(1 for row in rows if row["run_status"] == "reuse_e160"),
        "to_run_m0": sum(1 for row in rows if row["method_group"] == "postureRerankA" and row["run_status"] == "to_run"),
        "to_run_decay": sum(1 for row in rows if row["method_group"] == "releaseDecay"),
        "to_run_strict": sum(1 for row in rows if row["method_group"] == "strictMask"),
        "to_run_total": len(to_run),
        "preflight_rows": len(preflight),
        "preflight_ok": all(bool(row["ok"]) for row in preflight),
        "split_counts": {
            split: sum(1 for row in to_run if row["split"] == split)
            for split in ["local-gpu0", "remote-gpu0", "remote-gpu1"]
        },
    }
    return rows, preflight, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    rows, preflight, summary = build()
    write_tsv(SCRIPT_ROOT / "variants.tsv", rows, FIELDS)
    write_tsv(PREFLIGHT_ROOT / "e161_surface_release_ablation_preflight.tsv", preflight, PREFLIGHT_FIELDS)
    write_json(PREFLIGHT_ROOT / "e161_surface_release_ablation_preflight_summary.json", summary)
    print(
        "E161 manifest: "
        f"rows={len(rows)} reuse_e160={summary['reuse_e160']} "
        f"to_run_total={summary['to_run_total']} preflight_ok={summary['preflight_ok']} "
        f"split_counts={summary['split_counts']}"
    )
    if not summary["preflight_ok"]:
        for row in preflight:
            if not row["ok"]:
                print(row)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
