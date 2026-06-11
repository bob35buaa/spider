#!/usr/bin/env python3
"""Build E156 clean8 gate/decay benchmark manifest and overrides."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[5]
SCRIPT_ROOT = REPO / "workspace/core4d/scripts/experiments/E156"
LEGACY_SCRIPT_ROOT = REPO / "workspace/core4d/scripts/E156"
RESULT_ROOT = REPO / "workspace/core4d/results/E156/clean8_gate_decay"
CEM_ROOT = RESULT_ROOT / "cem/full"
PREFLIGHT_ROOT = RESULT_ROOT / "preflight"
OVERRIDE_ROOT = REPO / "examples/config/override"
E148_VARIANTS = REPO / "workspace/core4d/scripts/experiments/E148/variants.tsv"
E148_VARIANTS_FALLBACK = REPO / "workspace/core4d/scripts/E148/variants.tsv"
E155_CEM_ROOT = REPO / "workspace/core4d/results/E155/cem/full"

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

E155_REUSE_DECAY = {"box021_029_p2", "box004_083_p2", "box023_person2"}

METHODS = ["spider-rubberhand", "+gateA", "E155_decay"]
GATE_MIN_SDF_M = -0.010
GATE_MAX_VIOL = 0.10
GATE_HARD_FLOOR_M = -0.020

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
    "hand_support_scale",
    "contact_hdmi_mask_carry_union",
    "hand_support_decay_frac",
    "result_npz",
    "outdir_npz",
    "video",
    "expected_quality",
    "remote_sync_key",
]

PREFLIGHT_FIELDS = [
    "short_case_id",
    "method",
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


def short_case_id(row: dict[str, str]) -> str:
    variant = row["e143_variant"]
    if not variant.startswith("E143_") or not variant.endswith("_raw_mask_ref_fk"):
        raise ValueError(f"unexpected E143 variant: {variant}")
    return variant.removeprefix("E143_").removesuffix("_raw_mask_ref_fk")


def split_for(case_id: str, method: str) -> str:
    if method == "spider-rubberhand":
        return "reuse"
    if case_id in {"box021_035_p1", "box021_035_p2"}:
        return "local-gpu0"
    if case_id == "box021_029_p2":
        return "local-gpu0" if method == "+gateA" else "reuse"
    if case_id in {"box004_082_p1", "box004_083_p1"}:
        return "remote-gpu0"
    if case_id in {"box004_083_p2", "box023_person2"}:
        return "remote-gpu1" if method == "+gateA" else "reuse"
    if case_id == "box026_139_p1":
        return "remote-gpu1"
    raise ValueError(case_id)


def result_paths(variant: str) -> tuple[str, str, str]:
    return (
        rel(CEM_ROOT / f"{variant}.npz"),
        rel(CEM_ROOT / f"{variant}_outdir_full/trajectory_mjwp_act.npz"),
        rel(CEM_ROOT / f"{variant}_full.mp4"),
    )


def e155_decay_paths(case_id: str) -> tuple[str, str, str]:
    variant = f"E155_{case_id}_decay"
    return (
        rel(E155_CEM_ROOT / f"{variant}.npz"),
        rel(E155_CEM_ROOT / f"{variant}_outdir_full/trajectory_mjwp_act.npz"),
        rel(E155_CEM_ROOT / f"{variant}_full.mp4"),
    )


def rubber_task_inputs(e148: dict[str, str]) -> dict[str, str]:
    task_dir = repo_path(e148["rubber_scene_act"]).parent
    return {
        "derived_task": task_dir.name,
        "target_scene": rel(task_dir / "scene.xml"),
        "trajectory": rel(task_dir / "0/trajectory_kinematic.npz"),
        "base_scene_act": rel(task_dir / "scene_act.xml"),
    }


def write_gate_override(row: dict[str, Any], base_override: str) -> str:
    path = OVERRIDE_ROOT / f"core4d_{row['variant']}.yaml"
    base_stem = Path(base_override).stem
    lines = [
        "# @package _global_",
        "# Auto-generated by workspace/core4d/scripts/experiments/E156/build_clean8_gate_decay_manifest.py.",
        f"# E156 +gateA case={row['short_case_id']}.",
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
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rel(path)


def write_decay_override(row: dict[str, Any], base_override: str) -> str:
    path = OVERRIDE_ROOT / f"core4d_{row['variant']}.yaml"
    base_stem = Path(base_override).stem
    lines = [
        "# @package _global_",
        "# Auto-generated by workspace/core4d/scripts/experiments/E156/build_clean8_gate_decay_manifest.py.",
        f"# E156 E155_decay case={row['short_case_id']}.",
        "defaults:",
        f"  - {base_stem}",
        "  - _self_",
        "",
        f"task: {row['derived_task']}",
        f"scene_name: {row['scene_name']}",
        "video_camera: auto",
        "",
        "contact_hdmi_dynamic_target: true",
        "contact_hdmi_target_source: ref_fk",
        'contact_hdmi_target_path: ""',
        "contact_hdmi_target_time_axis: auto",
        "contact_hdmi_target_uses_eef_offset: true",
        "contact_hdmi_gain: 5.0",
        "",
        "hand_support_rew_scale: 3.0",
        "hand_support_sigma: 0.015",
        "hand_support_margin_m: 0.01",
        "hand_support_gate_source: contact_mask",
        "hand_support_start_eval_time: 0.0",
        "hand_support_end_eval_time: 0.0",
        'hand_support_geom_names: ["lh", "rh"]',
        "hand_support_geom_ids: []",
        "hand_object_deep_penalty_scale: 0.0",
        "hand_object_deep_penalty_geom_ids: []",
        "",
        "cem_hand_gate_enabled: true",
        'cem_hand_gate_geom_names: ["lh", "rh"]',
        "cem_hand_gate_geom_ids: []",
        f"cem_hand_gate_min_sdf_m: {GATE_MIN_SDF_M:.6f}",
        f"cem_hand_gate_max_violation_pct: {GATE_MAX_VIOL:.6f}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rel(path)


def base_row(e148: dict[str, str], method: str, ordinal: int) -> dict[str, Any]:
    case_id = short_case_id(e148)
    task_inputs = rubber_task_inputs(e148)
    if method == "spider-rubberhand":
        variant = f"E156_{case_id}_spider_rubberhand"
        run_status = "reuse_e148"
        source_exp = e148.get("reuse_source_exp") or "E148"
        result_npz, outdir_npz, video = e148["rubber_npz"], e148["rubber_outdir_npz"], e148["rubber_video"]
        override = e148["override"]
        method_group = "baseline"
        hand_support_scale = "0.0"
        carry_union = "false"
        decay_frac = "0.0"
    elif method == "+gateA":
        variant = f"E156_{case_id}_gateA"
        run_status = "to_run"
        source_exp = "E156"
        result_npz, outdir_npz, video = result_paths(variant)
        override = ""
        method_group = "gateA"
        hand_support_scale = "0.0"
        carry_union = "false"
        decay_frac = "0.0"
    elif method == "E155_decay":
        variant = f"E156_{case_id}_decay"
        source_exp = "E155" if case_id in E155_REUSE_DECAY else "E156"
        run_status = "reuse_e155" if case_id in E155_REUSE_DECAY else "to_run"
        result_npz, outdir_npz, video = e155_decay_paths(case_id) if case_id in E155_REUSE_DECAY else result_paths(variant)
        override = ""
        method_group = "decay"
        hand_support_scale = "3.0"
        carry_union = "true"
        decay_frac = "0.15"
    else:
        raise ValueError(method)

    row: dict[str, Any] = {
        "ordinal": ordinal,
        "variant": variant,
        "short_case_id": case_id,
        "method": method,
        "method_display": method,
        "method_group": method_group,
        "run_status": run_status,
        "source_exp": source_exp,
        "split": split_for(case_id, method),
        "e148_variant": e148["variant"],
        "e143_variant": e148["e143_variant"],
        "e109_case_id": e148["e109_case_id"],
        "case_id": e148["case_id"],
        "object_key": e148["object_key"],
        "object_category": e148["object_category"],
        "person_idx": e148["person_idx"],
        "derived_task": task_inputs["derived_task"],
        "target_scene": task_inputs["target_scene"],
        "trajectory": task_inputs["trajectory"],
        "base_scene_act": task_inputs["base_scene_act"],
        "rubber_scene_act": e148["rubber_scene_act"],
        "scene_name": e148["scene_name"],
        "override": override,
        "object_asset": e148["object_asset"],
        "mask_path": e148["mask_path"],
        "baseline_variant": f"E156_{case_id}_spider_rubberhand",
        "baseline_npz": e148["rubber_npz"],
        "baseline_outdir_npz": e148["rubber_outdir_npz"],
        "baseline_video": e148["rubber_video"],
        "cem_hand_gate_min_sdf_m": "" if method == "spider-rubberhand" else f"{GATE_MIN_SDF_M:.6f}",
        "cem_hand_gate_max_violation_pct": "" if method == "spider-rubberhand" else f"{GATE_MAX_VIOL:.6f}",
        "cem_hand_gate_hard_floor_m": "" if method == "spider-rubberhand" else f"{GATE_HARD_FLOOR_M:.6f}",
        "hand_support_scale": hand_support_scale,
        "contact_hdmi_mask_carry_union": carry_union,
        "hand_support_decay_frac": decay_frac,
        "result_npz": result_npz,
        "outdir_npz": outdir_npz,
        "video": video,
        "expected_quality": e148["expected_quality"],
        "remote_sync_key": e148["remote_sync_key"],
    }
    if method == "+gateA":
        row["override"] = write_gate_override(row, e148["override"])
    if method == "E155_decay" and run_status == "to_run":
        row["override"] = write_decay_override(row, e148["override"])
    if method == "E155_decay" and run_status == "reuse_e155":
        row["override"] = "E155_reuse_runtime_cli"
    return row


def preflight_row(row: dict[str, Any]) -> dict[str, Any]:
    task_dir = repo_path(row["rubber_scene_act"]).parent
    override_exists = row["override"] == "E155_reuse_runtime_cli" or repo_path(row["override"]).is_file()
    result_npz_exists = repo_path(row["result_npz"]).is_file()
    outdir_npz_exists = repo_path(row["outdir_npz"]).is_file()
    video_exists = repo_path(row["video"]).is_file()
    item = {
        "short_case_id": row["short_case_id"],
        "method": row["method"],
        "variant": row["variant"],
        "run_status": row["run_status"],
        "split": row["split"],
        "override_exists": override_exists,
        "task_dir_exists": task_dir.is_dir(),
        "object_asset_exists": repo_path(row["object_asset"]).is_file(),
        "mask_exists": repo_path(row["mask_path"]).is_file(),
        "baseline_npz_exists": repo_path(row["baseline_npz"]).is_file(),
        "baseline_outdir_exists": repo_path(row["baseline_outdir_npz"]).is_file(),
        "baseline_video_exists": repo_path(row["baseline_video"]).is_file(),
        "result_npz_exists": result_npz_exists,
        "outdir_npz_exists": outdir_npz_exists,
        "video_exists": video_exists,
    }
    if row["run_status"].startswith("reuse"):
        item["ok"] = all(
            bool(item[k])
            for k in (
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
            )
        )
    else:
        item["ok"] = all(
            bool(item[k])
            for k in (
                "override_exists",
                "task_dir_exists",
                "object_asset_exists",
                "mask_exists",
                "baseline_npz_exists",
                "baseline_outdir_exists",
                "baseline_video_exists",
            )
        )
    return item


def build() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    e148_path = E148_VARIANTS if E148_VARIANTS.is_file() else E148_VARIANTS_FALLBACK
    e148_rows = read_tsv(e148_path)
    by_short = {short_case_id(row): row for row in e148_rows}
    missing_cases = [case for case in CLEAN8_CASES if case not in by_short]
    if missing_cases:
        raise RuntimeError(f"E148 variants missing clean8 cases: {missing_cases}")

    rows: list[dict[str, Any]] = []
    ordinal = 1
    for case_id in CLEAN8_CASES:
        for method in METHODS:
            rows.append(base_row(by_short[case_id], method, ordinal))
            ordinal += 1

    preflight = [preflight_row(row) for row in rows]
    summary = {
        "experiment": "E156",
        "git_head": git_head(),
        "source_e148_variants": rel(e148_path),
        "clean8_cases": CLEAN8_CASES,
        "method_rows": len(rows),
        "preflight_rows": len(preflight),
        "run_status_counts": {status: sum(1 for r in rows if r["run_status"] == status) for status in sorted({r["run_status"] for r in rows})},
        "split_counts": {split: sum(1 for r in rows if r["split"] == split) for split in sorted({r["split"] for r in rows})},
        "preflight_ok": all(bool(r["ok"]) for r in preflight),
        "to_run_count": sum(1 for r in rows if r["run_status"] == "to_run"),
        "gate_min_sdf_m": GATE_MIN_SDF_M,
        "gate_max_viol": GATE_MAX_VIOL,
        "gate_hard_floor_m": GATE_HARD_FLOOR_M,
    }
    return rows, preflight, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-copy", action="store_true", help="also copy variants.tsv to workspace/core4d/scripts/E156")
    args = parser.parse_args()

    rows, preflight, summary = build()
    write_tsv(SCRIPT_ROOT / "variants.tsv", rows, FIELDS)
    write_tsv(PREFLIGHT_ROOT / "e156_clean8_gate_decay_preflight.tsv", preflight, PREFLIGHT_FIELDS)
    write_json(PREFLIGHT_ROOT / "e156_clean8_gate_decay_preflight_summary.json", summary)
    if args.legacy_copy:
        LEGACY_SCRIPT_ROOT.mkdir(parents=True, exist_ok=True)
        write_tsv(LEGACY_SCRIPT_ROOT / "variants.tsv", rows, FIELDS)
    print(
        "E156 manifest: "
        f"rows={len(rows)} to_run={summary['to_run_count']} "
        f"preflight_ok={summary['preflight_ok']} "
        f"run_status={summary['run_status_counts']}"
    )
    if not summary["preflight_ok"]:
        bad = [r for r in preflight if not r["ok"]]
        print("Bad preflight rows:")
        for row in bad:
            print(row)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
