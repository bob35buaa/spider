#!/usr/bin/env python3
"""Evaluate E167 CEM/postprocess artifacts with SPIDER contact/tracking metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_E166_foot_smooth_retarget as e166  # noqa: E402
from eval.core.core_metrics import EvalConfig, evaluate_sequence, person_idx_from_case  # noqa: E402
from eval_E156_clean8_gate_decay import add_success_flags, contact_mask_for_case, kin_ref_for_scene  # noqa: E402


REPO = Path(__file__).resolve().parents[5]
VARIANTS = REPO / "workspace/core4d/scripts/experiments/E167/variants.tsv"
RESULT_ROOT = REPO / "workspace/core4d/results/E167/holosoma_zonly"
EVAL_ROOT = RESULT_ROOT / "eval/cem_metrics"
ARM_ORDER = ["baseline", "E167A", "E167A_B1", "E167A_B2"]
CASE_ORDER = [
    "box023_person2",
    "box021_029_p2",
    "box021_035_p1",
    "box021_035_p2",
    "box004_082_p1",
    "box004_083_p1",
    "box004_083_p2",
]
XY_TOL = 1e-6


def rel(path: str | Path) -> str:
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


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    e166.write_tsv(path, rows, fields)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None or not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def close(value: Any, target: float, tol: float = 1e-8) -> bool:
    try:
        return abs(float(value) - target) <= tol
    except Exception:
        return False


def baseline_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        case = row["short_case_id"]
        if case in seen:
            continue
        seen.add(case)
        base = dict(row)
        base.update(
            {
                "variant": row["base_e163_variant"],
                "arm": "baseline",
                "arm_kind": "baseline",
                "method": "E163_narrowSurfaceBand",
                "method_display": "E163_narrowSurfaceBand",
                "split": "baseline",
                "result_npz": row["baseline_npz"],
                "outdir_npz": row["baseline_outdir_npz"],
                "video": row["baseline_video"],
                "postprocess_source_npz": "",
                "postprocess_output_npz": "",
            }
        )
        out.append(base)
    return out


def qpos_path(row: dict[str, str]) -> Path:
    if row["arm_kind"] == "postprocess":
        return repo_path(row["postprocess_output_npz"])
    return repo_path(row["outdir_npz"] or row["result_npz"])


def video_path(row: dict[str, str]) -> Path:
    return repo_path(row["video"]) if row.get("video") else Path("")


def config_path(row: dict[str, str]) -> Path:
    if row["arm_kind"] != "cem":
        return Path("")
    outdir = Path(row["outdir_npz"])
    return repo_path(outdir.parent / "config_act.yaml") if outdir.name == "trajectory_mjwp_act.npz" else Path("")


def artifact_check(row: dict[str, str]) -> dict[str, Any]:
    qpath = qpos_path(row)
    cfg_path = config_path(row)
    cfg = load_yaml(cfg_path)
    item: dict[str, Any] = {
        "variant": row["variant"],
        "short_case_id": row["short_case_id"],
        "arm": row["arm"],
        "arm_kind": row["arm_kind"],
        "qpos_exists": qpath.is_file(),
        "video_exists": row["arm_kind"] in {"baseline", "postprocess"} or video_path(row).is_file(),
        "config_exists": row["arm_kind"] != "cem" or cfg_path.is_file(),
        "qpos_path": rel(qpath),
        "video": rel(video_path(row)) if video_path(row) else "",
        "config_act": rel(cfg_path) if cfg_path else "",
    }
    if row["arm_kind"] == "cem":
        item.update(
            {
                "config_e167_body_z_enabled_ok": boolish(cfg.get("e167_body_z_enabled")),
                "config_e167_ground_z_enabled_ok": boolish(cfg.get("e167_ground_z_enabled")),
                "config_foot_slip_disabled_ok": not boolish(cfg.get("foot_slip_enabled")),
                "config_foot_ground_disabled_ok": not boolish(cfg.get("foot_ground_enabled")),
                "config_ankle_weight_neutral_ok": close(cfg.get("local_frame_ankle_weight"), 1.0),
                "config_b1_smooth_enabled_ok": row["arm"] != "E167A_B1" or boolish(cfg.get("cem_smooth_enabled")),
                "config_b1_axis_z_ok": row["arm"] != "E167A_B1" or str(cfg.get("cem_smooth_axis", "")).lower() == "z",
                "config_a_smooth_disabled_ok": row["arm"] != "E167A" or not boolish(cfg.get("cem_smooth_enabled")),
            }
        )
    if row["arm_kind"] == "postprocess":
        report = qpath.with_suffix(qpath.suffix + ".zonly_smooth_report.json")
        xy = math.inf
        if report.is_file():
            xy = float(json.loads(report.read_text(encoding="utf-8")).get("xy_max_abs_delta", math.inf))
        item.update(
            {
                "zonly_report": rel(report),
                "zonly_report_exists": report.is_file(),
                "xy_max_abs_delta": xy if math.isfinite(xy) else "",
                "b2_xy_invariance_ok": math.isfinite(xy) and xy <= XY_TOL,
            }
        )
    bool_keys = [key for key in item if key.endswith("_exists") or key.endswith("_ok")]
    item["artifact_ok"] = all(bool(item[key]) for key in bool_keys)
    return item


def evaluate_one(row: dict[str, str], cfg: EvalConfig) -> dict[str, Any] | None:
    qpath = qpos_path(row)
    scene = repo_path(row["rubber_scene_act"])
    if not qpath.is_file() or not scene.is_file():
        return None
    item = evaluate_sequence(
        row=row,
        method=row["method"],
        hand_collision_variant_id="rubber_hull",
        qpos_path=qpath,
        scene_xml=scene,
        config=cfg,
        kin_ref_path=kin_ref_for_scene(scene),
        contact_mask_path=contact_mask_for_case(row["short_case_id"]),
        person_idx=person_idx_from_case(row["short_case_id"]),
    )
    add_success_flags(item, cfg)
    item.update(
        {
            "short_case_id": row["short_case_id"],
            "variant": row["variant"],
            "arm": row["arm"],
            "arm_kind": row["arm_kind"],
            "method": row["method"],
            "source_exp": "E167",
            "split": row["split"],
            "result_npz": rel(qpath),
            "video": rel(video_path(row)) if video_path(row).is_file() else "",
        }
    )
    item.update(e166.run_health(qpath, scene, cfg))
    return item


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for arm in ARM_ORDER:
        arm_rows = [row for row in rows if row.get("arm") == arm]
        if not arm_rows:
            continue
        item: dict[str, Any] = {
            "arm": arm,
            "n_cases": len(arm_rows),
            "pass_cases": sum(1 for row in arm_rows if row.get("spider_gate_pass")),
            "tracked_cases": sum(1 for row in arm_rows if row.get("success_tracked")),
            "fall_cases": sum(1 for row in arm_rows if row.get("fall_flag")),
            "failed_cases": ",".join(str(row.get("short_case_id")) for row in arm_rows if not row.get("spider_gate_pass")),
        }
        for key in e166.METRIC_KEYS:
            if key in {"success_tracked", "fall_flag"}:
                item[f"{key}_mean"] = e166.mean([1.0 if row.get(key) else 0.0 for row in arm_rows])
            else:
                item[f"{key}_mean"] = e166.mean([row.get(key) for row in arm_rows])
                item[f"{key}_delta_vs_baseline_mean"] = e166.mean(
                    [row.get(f"{key}_delta_vs_baseline") for row in arm_rows if row.get("arm") != "baseline"]
                )
        out.append(item)
    return out


def write_summary_md(path: Path, summary_rows: list[dict[str, Any]], metric_rows: list[dict[str, Any]], missing: list[dict[str, Any]]) -> None:
    lines = [
        "# E167 Holosoma z-only SPIDER metrics",
        "",
        "| arm | n | pass | tracked | fall | raw contact | clean3 contact | pen3 | qpos jerk p95 | track jerk p95 | failed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        lines.append(
            "| {arm} | {n_cases} | {pass_cases} | {tracked_cases} | {fall_cases} | {raw:.3f} | {clean:.3f} | {pen:.3f} | {qjerk:.1f} | {tjerk:.1f} | {failed} |".format(
                arm=row.get("arm", ""),
                n_cases=row.get("n_cases", ""),
                pass_cases=row.get("pass_cases", ""),
                tracked_cases=row.get("tracked_cases", ""),
                fall_cases=row.get("fall_cases", ""),
                raw=e166.finite(row.get("hand_object_physics_contact_in_mask_frac_mean")),
                clean=e166.finite(row.get("hand_object_physics_contact_3mm_in_mask_frac_mean")),
                pen=e166.finite(row.get("hand_object_physics_penetration_3mm_frame_frac_mean")),
                qjerk=e166.finite(row.get("qpos_jerk_l2_p95_mean")),
                tjerk=e166.finite(row.get("trackbody_jerk_p95_mean")),
                failed=row.get("failed_cases", ""),
            )
        )
    if missing:
        lines.extend(["", "## Missing", ""])
        for row in missing:
            lines.append(f"- {row['short_case_id']} {row['arm']}: {row['reason']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(stage: str, allow_missing: bool) -> dict[str, Any]:
    rows = baseline_rows(read_tsv(VARIANTS)) + read_tsv(VARIANTS)
    cfg = EvalConfig()
    eval_dir = EVAL_ROOT / stage
    metric_rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for row in rows:
        artifact = artifact_check(row)
        artifact_rows.append(artifact)
        item = evaluate_one(row, cfg)
        if item is None:
            missing_rows.append(
                {
                    "short_case_id": row["short_case_id"],
                    "arm": row["arm"],
                    "variant": row["variant"],
                    "reason": "missing_qpos_or_scene",
                    "qpos_path": rel(qpos_path(row)),
                }
            )
            continue
        item.update(artifact)
        metric_rows.append(item)
    e166.add_baseline_deltas(metric_rows)
    summary_rows = summarize(metric_rows)
    metric_fields = [
        "short_case_id",
        "arm",
        "variant",
        "method",
        "status",
        "spider_gate_pass",
        "success_tracked",
        "fall_flag",
        "contact_no_regression",
        "penetration_no_regression",
        "artifact_ok",
        *e166.METRIC_KEYS,
        *[f"{key}_delta_vs_baseline" for key in e166.METRIC_KEYS if key not in {"success_tracked", "fall_flag"}],
        *e166.HEALTH_AGGS.keys(),
        "has_smooth_health",
        "has_foot_health",
        "qpos_path",
        "config_act",
        "video",
    ]
    write_tsv(eval_dir / "e167_arm_metrics.tsv", metric_rows, metric_fields)
    write_tsv(eval_dir / "e167_arm_summary.tsv", summary_rows)
    write_tsv(eval_dir / "e167_artifact_check.tsv", artifact_rows)
    write_tsv(eval_dir / "e167_missing.tsv", missing_rows)
    write_summary_md(eval_dir / "summary.md", summary_rows, metric_rows, missing_rows)
    payload = {
        "stage": stage,
        "rows_expected": len(rows),
        "rows_evaluated": len(metric_rows),
        "missing_rows": len(missing_rows),
        "allow_missing": allow_missing,
        "all_artifacts_ok_for_evaluated": all(bool(row.get("artifact_ok")) for row in metric_rows),
        "summary": summary_rows,
    }
    write_json(eval_dir / "summary.json", payload)
    if missing_rows and not allow_missing:
        raise SystemExit(f"E167 metrics missing {len(missing_rows)} rows; see {rel(eval_dir / 'e167_missing.tsv')}")
    print(
        "E167 metrics: "
        f"stage={stage} expected={len(rows)} evaluated={len(metric_rows)} "
        f"missing={len(missing_rows)} out={rel(eval_dir)}"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", nargs="?", default="full")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    evaluate(args.stage, args.allow_missing)


if __name__ == "__main__":
    main()
