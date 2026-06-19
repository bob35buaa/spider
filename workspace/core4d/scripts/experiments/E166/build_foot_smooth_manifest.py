#!/usr/bin/env python3
"""Build E166 foot/smoothness retarget manifest and CEM overrides."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[5]
SCRIPT_ROOT = REPO / "workspace/core4d/scripts/experiments/E166"
RESULT_ROOT = REPO / "workspace/core4d/results/E166/foot_smooth_retarget"
CEM_ROOT = RESULT_ROOT / "cem"
POST_ROOT = RESULT_ROOT / "postprocess"
PREFLIGHT_ROOT = RESULT_ROOT / "preflight"
OVERRIDE_ROOT = REPO / "examples/config/override"
E163_VARIANTS = REPO / "workspace/core4d/scripts/experiments/E163/variants.tsv"

TARGET_CASES = [
    "box021_035_p2",
    "box004_082_p1",
    "box004_083_p2",
]

CASE_SPLITS = {
    "box021_035_p2": "local-gpu0",
    "box004_082_p1": "remote-gpu0",
    "box004_083_p2": "remote-gpu1",
}

ARMS = ["baseline", "B1", "B2", "A", "A_B2_postSmooth", "AplusB"]
ARM_METHOD = {
    "baseline": "E163_narrowSurfaceBand",
    "B1": "E166_B1_cemSmooth",
    "B2": "E166_B2_smoothHandoff",
    "A": "E166_A_footConstraints",
    "A_B2_postSmooth": "E166_A_then_B2_postSmooth",
    "AplusB": "E166_AplusB_footSmoothFull",
}
ARM_KIND = {
    "baseline": "reuse_e163",
    "B1": "cem",
    "B2": "postprocess",
    "A": "cem",
    "A_B2_postSmooth": "postprocess",
    "AplusB": "cem",
}

SMOOTH_ACCEL_WEIGHT = 0.0005
SMOOTH_JERK_WEIGHT = 0.00002
ANKLE_WEIGHT = 2.0
FOOT_SLIP_WEIGHT = 0.2
FOOT_GROUND_WEIGHT = 2.0
FOOT_CONTACT_HEIGHT_M = 0.05
B2_WINDOW = 7
B2_POLYORDER = 2

FIELDS = [
    "ordinal",
    "variant",
    "short_case_id",
    "arm",
    "arm_kind",
    "method",
    "method_display",
    "run_status",
    "source_exp",
    "split",
    "base_e163_variant",
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
    "base_override",
    "object_asset",
    "mask_path",
    "baseline_npz",
    "baseline_outdir_npz",
    "baseline_video",
    "cem_smooth_enabled",
    "cem_smooth_accel_weight",
    "cem_smooth_jerk_weight",
    "local_frame_ankle_weight",
    "foot_slip_enabled",
    "foot_slip_weight",
    "foot_slip_contact_height_m",
    "foot_ground_enabled",
    "foot_ground_weight",
    "b2_window",
    "b2_polyorder",
    "result_npz",
    "outdir_npz",
    "video",
    "postprocess_source_npz",
    "postprocess_output_npz",
    "remote_sync_key",
]

PREFLIGHT_FIELDS = [
    "short_case_id",
    "arm",
    "variant",
    "run_status",
    "split",
    "override_exists",
    "base_override_exists",
    "task_dir_exists",
    "object_asset_exists",
    "mask_exists",
    "e163_result_npz_exists",
    "e163_outdir_exists",
    "e163_video_exists",
    "result_npz_exists",
    "outdir_npz_exists",
    "video_exists",
    "postprocess_source_exists",
    "postprocess_output_exists",
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
            writer.writerow(
                {field: "" if row.get(field) is None else row.get(field, "") for field in fields}
            )


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def variant_for(case_id: str, arm: str) -> str:
    return f"E166_{case_id}_{arm}"


def cem_result_paths(case_id: str, arm: str, stage: str = "full") -> tuple[str, str, str]:
    variant = variant_for(case_id, arm)
    root = CEM_ROOT / stage
    return (
        rel(root / f"{variant}.npz"),
        rel(root / f"{variant}_outdir_{stage}/trajectory_mjwp_act.npz"),
        rel(root / f"{variant}_{stage}.mp4"),
    )


def postprocess_output_path(case_id: str, arm: str, stage: str = "full") -> str:
    return rel(POST_ROOT / stage / f"{variant_for(case_id, arm)}.npz")


def artifact_complete(row: dict[str, Any]) -> bool:
    if row["arm_kind"] == "reuse_e163":
        return repo_path(row["baseline_npz"]).is_file()
    if row["arm_kind"] == "postprocess":
        return repo_path(row["postprocess_output_npz"]).is_file()
    return (
        repo_path(row["result_npz"]).is_file()
        and repo_path(row["outdir_npz"]).is_file()
        and repo_path(row["video"]).is_file()
    )


def write_override(row: dict[str, Any]) -> str:
    if row["arm_kind"] != "cem":
        return ""
    path = OVERRIDE_ROOT / f"core4d_{row['variant']}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    base_stem = Path(row["base_override"]).stem
    lines = [
        "# @package _global_",
        "# Auto-generated by workspace/core4d/scripts/experiments/E166/build_foot_smooth_manifest.py.",
        f"# E166 {row['arm']} case={row['short_case_id']}.",
        "defaults:",
        f"  - {base_stem}",
        "  - _self_",
        "",
    ]
    if row["arm"] in {"B1", "AplusB"}:
        lines.extend(
            [
                "cem_smooth_enabled: true",
                'cem_smooth_body_names: ["left_ankle_roll_link", "right_ankle_roll_link"]',
                "cem_smooth_body_ids: []",
                f"cem_smooth_accel_weight: {SMOOTH_ACCEL_WEIGHT:.8f}",
                f"cem_smooth_jerk_weight: {SMOOTH_JERK_WEIGHT:.8f}",
                "",
            ]
        )
    if row["arm"] in {"A", "AplusB"}:
        lines.extend(
            [
                "local_frame_ankle_ids: [7, 13]",
                f"local_frame_ankle_weight: {ANKLE_WEIGHT:.6f}",
                "foot_slip_enabled: true",
                f"foot_slip_weight: {FOOT_SLIP_WEIGHT:.6f}",
                f"foot_slip_contact_height_m: {FOOT_CONTACT_HEIGHT_M:.6f}",
                "foot_ground_enabled: true",
                f"foot_ground_weight: {FOOT_GROUND_WEIGHT:.6f}",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rel(path)


def build_row(case_id: str, arm: str, e163_row: dict[str, str], ordinal: int) -> dict[str, Any]:
    arm_kind = ARM_KIND[arm]
    result_npz, outdir_npz, video = ("", "", "")
    if arm_kind == "cem":
        result_npz, outdir_npz, video = cem_result_paths(case_id, arm)
    elif arm_kind == "reuse_e163":
        result_npz = e163_row["result_npz"]
        outdir_npz = e163_row["outdir_npz"]
        video = e163_row["video"]

    postprocess_output = postprocess_output_path(case_id, arm) if arm_kind == "postprocess" else ""
    postprocess_source = ""
    if arm == "B2":
        postprocess_source = e163_row["result_npz"]
    elif arm == "A_B2_postSmooth":
        postprocess_source = cem_result_paths(case_id, "A")[0]
    row: dict[str, Any] = {
        "ordinal": ordinal,
        "variant": variant_for(case_id, arm),
        "short_case_id": case_id,
        "arm": arm,
        "arm_kind": arm_kind,
        "method": ARM_METHOD[arm],
        "method_display": ARM_METHOD[arm],
        "run_status": "",
        "source_exp": "E166",
        "split": CASE_SPLITS[case_id] if arm_kind == "cem" else "local-cpu",
        "base_e163_variant": e163_row["variant"],
        "case_id": e163_row["case_id"],
        "object_key": e163_row["object_key"],
        "object_category": e163_row["object_category"],
        "person_idx": e163_row["person_idx"],
        "derived_task": e163_row["derived_task"],
        "target_scene": e163_row["target_scene"],
        "trajectory": e163_row["trajectory"],
        "base_scene_act": e163_row["base_scene_act"],
        "rubber_scene_act": e163_row["rubber_scene_act"],
        "scene_name": e163_row["scene_name"],
        "override": "",
        "base_override": e163_row["override"],
        "object_asset": e163_row["object_asset"],
        "mask_path": e163_row["mask_path"],
        "baseline_npz": e163_row["result_npz"],
        "baseline_outdir_npz": e163_row["outdir_npz"],
        "baseline_video": e163_row["video"],
        "cem_smooth_enabled": str(arm in {"B1", "AplusB"}).lower(),
        "cem_smooth_accel_weight": f"{SMOOTH_ACCEL_WEIGHT:.8f}" if arm in {"B1", "AplusB"} else "0.0",
        "cem_smooth_jerk_weight": f"{SMOOTH_JERK_WEIGHT:.8f}" if arm in {"B1", "AplusB"} else "0.0",
        "local_frame_ankle_weight": f"{ANKLE_WEIGHT:.6f}" if arm in {"A", "AplusB"} else "1.0",
        "foot_slip_enabled": str(arm in {"A", "AplusB"}).lower(),
        "foot_slip_weight": f"{FOOT_SLIP_WEIGHT:.6f}" if arm in {"A", "AplusB"} else "0.0",
        "foot_slip_contact_height_m": f"{FOOT_CONTACT_HEIGHT_M:.6f}",
        "foot_ground_enabled": str(arm in {"A", "AplusB"}).lower(),
        "foot_ground_weight": f"{FOOT_GROUND_WEIGHT:.6f}" if arm in {"A", "AplusB"} else "0.0",
        "b2_window": B2_WINDOW if arm_kind == "postprocess" else "",
        "b2_polyorder": B2_POLYORDER if arm_kind == "postprocess" else "",
        "result_npz": result_npz,
        "outdir_npz": outdir_npz,
        "video": video,
        "postprocess_source_npz": postprocess_source,
        "postprocess_output_npz": postprocess_output,
        "remote_sync_key": e163_row.get("remote_sync_key", ""),
    }
    row["override"] = write_override(row)
    if artifact_complete(row):
        row["run_status"] = "reuse_e163_full" if arm == "baseline" else "reuse_existing"
    elif arm_kind == "postprocess":
        row["run_status"] = "postprocess_to_run"
    elif arm_kind == "cem":
        row["run_status"] = "to_run"
    else:
        row["run_status"] = "missing_baseline"
    return row


def preflight_row(row: dict[str, Any]) -> dict[str, Any]:
    task_dir = repo_path(row["rubber_scene_act"]).parent
    item = {
        "short_case_id": row["short_case_id"],
        "arm": row["arm"],
        "variant": row["variant"],
        "run_status": row["run_status"],
        "split": row["split"],
        "override_exists": (not row["override"]) or repo_path(row["override"]).is_file(),
        "base_override_exists": repo_path(row["base_override"]).is_file(),
        "task_dir_exists": task_dir.is_dir(),
        "object_asset_exists": repo_path(row["object_asset"]).is_file(),
        "mask_exists": repo_path(row["mask_path"]).is_file(),
        "e163_result_npz_exists": repo_path(row["baseline_npz"]).is_file(),
        "e163_outdir_exists": repo_path(row["baseline_outdir_npz"]).is_file(),
        "e163_video_exists": repo_path(row["baseline_video"]).is_file(),
        "result_npz_exists": bool(row["result_npz"]) and repo_path(row["result_npz"]).is_file(),
        "outdir_npz_exists": bool(row["outdir_npz"]) and repo_path(row["outdir_npz"]).is_file(),
        "video_exists": bool(row["video"]) and repo_path(row["video"]).is_file(),
        "postprocess_source_exists": (not row["postprocess_source_npz"])
        or repo_path(row["postprocess_source_npz"]).is_file(),
        "postprocess_output_exists": bool(row["postprocess_output_npz"])
        and repo_path(row["postprocess_output_npz"]).is_file(),
    }
    required = [
        "override_exists",
        "base_override_exists",
        "task_dir_exists",
        "object_asset_exists",
        "mask_exists",
        "e163_result_npz_exists",
        "e163_outdir_exists",
        "e163_video_exists",
        "postprocess_source_exists",
    ]
    item["ok"] = all(bool(item[key]) for key in required)
    return item


def build() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    e163_rows = read_tsv(E163_VARIANTS)
    by_case = {row["short_case_id"]: row for row in e163_rows}
    missing = [case for case in TARGET_CASES if case not in by_case]
    if missing:
        raise RuntimeError(f"E163 rows missing target cases: {missing}")

    rows = []
    ordinal = 1
    for case in TARGET_CASES:
        for arm in ARMS:
            rows.append(build_row(case, arm, by_case[case], ordinal))
            ordinal += 1
    preflight = [preflight_row(row) for row in rows]
    to_run_rows = [row for row in rows if row["run_status"] == "to_run"]
    post_rows = [row for row in rows if row["run_status"] == "postprocess_to_run"]
    summary = {
        "experiment": "E166",
        "git_head": git_head(),
        "source_e163_variants": rel(E163_VARIANTS),
        "target_cases": TARGET_CASES,
        "arms": ARMS,
        "rows": len(rows),
        "cem_to_run_total": len(to_run_rows),
        "postprocess_to_run_total": len(post_rows),
        "preflight_rows": len(preflight),
        "preflight_ok": all(bool(row["ok"]) for row in preflight),
        "split_counts": {
            split: sum(1 for row in to_run_rows if row["split"] == split)
            for split in ["local-gpu0", "remote-gpu0", "remote-gpu1"]
        },
        "smooth_weights": {
            "accel": SMOOTH_ACCEL_WEIGHT,
            "jerk": SMOOTH_JERK_WEIGHT,
        },
        "foot_weights": {
            "ankle": ANKLE_WEIGHT,
            "slip": FOOT_SLIP_WEIGHT,
            "ground": FOOT_GROUND_WEIGHT,
            "contact_height_m": FOOT_CONTACT_HEIGHT_M,
        },
    }
    return rows, preflight, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    rows, preflight, summary = build()
    write_tsv(SCRIPT_ROOT / "variants.tsv", rows, FIELDS)
    write_tsv(PREFLIGHT_ROOT / "e166_foot_smooth_preflight.tsv", preflight, PREFLIGHT_FIELDS)
    write_json(PREFLIGHT_ROOT / "e166_foot_smooth_preflight_summary.json", summary)
    print(
        "E166 manifest: "
        f"rows={len(rows)} cem_to_run={summary['cem_to_run_total']} "
        f"postprocess_to_run={summary['postprocess_to_run_total']} "
        f"preflight_ok={summary['preflight_ok']} split_counts={summary['split_counts']}"
    )
    if not summary["preflight_ok"]:
        for row in preflight:
            if not row["ok"]:
                print(row)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
