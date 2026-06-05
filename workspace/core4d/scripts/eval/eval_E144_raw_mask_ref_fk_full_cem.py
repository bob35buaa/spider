#!/usr/bin/env python3
"""Evaluate E144 raw_mask_ref_fk full-CEM rows and build downstream evidence inputs."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any


THIS = Path(__file__).resolve()
REPO = THIS.parents[4]
TASK_ROOT = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
VARIANTS_TSV = REPO / "workspace/core4d/scripts/E144/variants.tsv"
E090_EVAL = REPO / "workspace/core4d/scripts/eval/eval_E090.py"
E105_EVAL = REPO / "workspace/core4d/scripts/eval/eval_E105_box026_clean_cem.py"
OUT_ROOT = REPO / "workspace/core4d/results/E144/eval"

FIELDS = [
    "ordinal",
    "variant",
    "case_id",
    "object_key",
    "object_category",
    "person",
    "person_idx",
    "split",
    "ablation",
    "retarget_variant_id",
    "target_variant_id",
    "derived_task",
    "target_scene",
    "trajectory",
    "scene_act",
    "mask_path",
    "mask_kind",
    "override",
    "handoff_decision",
    "candidate_decision",
    "raw_contact_threshold_label",
    "run_status",
]

METRIC_FIELDS = [
    *FIELDS,
    "stage",
    "cem_status",
    "downstream_decision",
    "downstream_failure_mode",
    "hand_object_contact_physics_frac",
    "hand_near_5cm_frac",
    "hand_near_10cm_frac",
    "hand_object_penetration_frac",
    "hand_object_deep_penetration_2cm_frac",
    "leg_box_interference_frac",
    "lowerbody_strict_pass",
    "pelvis_min_m",
    "obj_err_mean_m",
    "obj_err_max_m",
    "work_status",
    "stage_pass",
    "root_npz_path",
    "video_path",
    "outdir_npz_path",
    "legobj_timeseries_csv",
]

EVIDENCE_FIELDS = [
    "case_id",
    "object_key",
    "object_name",
    "date",
    "seq",
    "person",
    "person_idx",
    "retarget_variant_id",
    "target_variant_id",
    "candidate_decision",
    "handoff_decision",
    "cem_status",
    "rl_status",
    "downstream_decision",
    "downstream_failure_mode",
    "downstream_notes",
    "cem_run_id",
    "cem_result_npz",
    "cem_video",
    "cem_metrics_ref",
    "rl_run_id",
    "rl_checkpoint",
    "rl_metrics_ref",
    "rl_video",
]


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO.resolve()))
    except ValueError:
        return str(path)


def repo_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else REPO / path


def repo_path_from_env(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    path = Path(raw) if raw else default
    return path if path.is_absolute() else REPO / path


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.8g}"
    return str(value)


def read_tsv(path: Path, fields: list[str] | None = None) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        lines = [line for line in f if line.strip() and not line.startswith("#")]
        if not lines:
            return []
        if fields is None:
            return list(csv.DictReader(lines, delimiter="\t"))
        return list(csv.DictReader(lines, fieldnames=fields, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field, "")) for field in fields})


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expected_paths(stage: str, row: dict[str, str], results_dir: Path) -> dict[str, Path]:
    variant = row["variant"]
    return {
        "root": results_dir / f"{variant}.npz",
        "video": results_dir / f"{variant}_{stage}.mp4",
        "outdir": results_dir / f"{variant}_outdir_{stage}/trajectory_mjwp_act.npz",
    }


def missing_outputs(stage: str, rows: list[dict[str, str]], results_dir: Path) -> list[dict[str, str]]:
    missing = []
    for row in rows:
        paths = expected_paths(stage, row, results_dir)
        absent = [rel(path) for path in paths.values() if not path.is_file()]
        if absent:
            missing.append({"variant": row["variant"], "case_id": row["case_id"], "missing": ";".join(absent)})
    return missing


def compute_metrics(modules: tuple[Any, Any], row: dict[str, str], stage: str, results_dir: Path) -> dict[str, Any] | None:
    paths = expected_paths(stage, row, results_dir)
    outdir_npz = paths["outdir"]
    if not outdir_npz.is_file():
        return None
    eval_e090, eval_e105 = modules
    scene = TASK_ROOT / row["derived_task"] / "scene_act.xml"
    metrics = eval_e090.compute_sim_metrics(outdir_npz, scene)
    metrics.update(eval_e105.replay_metrics(outdir_npz, scene))
    metrics.update(eval_e105.leg_object_metrics(row["variant"], outdir_npz, scene, results_dir))

    ts_path = repo_path(str(metrics["legobj_timeseries_csv"]))
    hand_sdf = []
    with ts_path.open("r", encoding="utf-8", newline="") as f:
        for item in csv.DictReader(f):
            hand_sdf.append(float(item["hand_box_sdf_min_m"]))
    hand_near_5 = sum(value <= 0.05 for value in hand_sdf) / len(hand_sdf) if hand_sdf else 0.0
    hand_near_10 = sum(value <= 0.10 for value in hand_sdf) / len(hand_sdf) if hand_sdf else 0.0
    hand_pen = sum(value < 0.0 for value in hand_sdf) / len(hand_sdf) if hand_sdf else 0.0
    hand_deep = sum(value < -0.02 for value in hand_sdf) / len(hand_sdf) if hand_sdf else 0.0
    work_status, stage_pass = eval_e105.status(stage, metrics)
    lowerbody_pass = bool(metrics["lowerbody_strict_pass"])
    cem_status = "pass" if work_status == "WORK" and lowerbody_pass else "fail"
    failure_mode = "" if cem_status == "pass" else ("lowerbody_interference" if not lowerbody_pass else "cem_work_status_fail")
    return {
        **row,
        "stage": stage,
        "cem_status": cem_status,
        "downstream_decision": "DOWNSTREAM_CEM_PASS" if cem_status == "pass" else "DOWNSTREAM_CEM_FAIL",
        "downstream_failure_mode": failure_mode,
        "hand_object_contact_physics_frac": metrics.get("hand_object_contact_physics_frac"),
        "hand_near_5cm_frac": hand_near_5,
        "hand_near_10cm_frac": hand_near_10,
        "hand_object_penetration_frac": hand_pen,
        "hand_object_deep_penetration_2cm_frac": hand_deep,
        "leg_box_interference_frac": metrics.get("leg_box_interference_frac"),
        "lowerbody_strict_pass": lowerbody_pass,
        "pelvis_min_m": metrics.get("pelvis_min_m"),
        "obj_err_mean_m": metrics.get("obj_err_mean_m"),
        "obj_err_max_m": metrics.get("obj_err_max_m"),
        "work_status": work_status,
        "stage_pass": stage_pass,
        "root_npz_path": rel(paths["root"]),
        "video_path": rel(paths["video"]),
        "outdir_npz_path": rel(outdir_npz),
        "legobj_timeseries_csv": metrics.get("legobj_timeseries_csv", ""),
    }


def evidence_rows(metric_rows: list[dict[str, Any]], summary_path: Path) -> list[dict[str, Any]]:
    rows = []
    for row in metric_rows:
        parts = row["case_id"].split("_")
        date = row.get("date", "")
        seq = row.get("seq", "")
        if not date and len(parts) >= 3:
            date = parts[1] if parts[0].startswith("bucket") else ""
        if not seq and len(parts) >= 3:
            seq = parts[2] if parts[0].startswith("bucket") else ""
        notes = (
            f"work_status={row.get('work_status', '')}; lowerbody_strict_pass={row.get('lowerbody_strict_pass', '')}; "
            f"contact={float(row.get('hand_object_contact_physics_frac') or 0.0) * 100:.1f}%; "
            f"leg_interference={float(row.get('leg_box_interference_frac') or 0.0) * 100:.1f}%"
        )
        rows.append(
            {
                "case_id": row["case_id"],
                "object_key": row.get("object_key", ""),
                "object_name": row.get("object_key", ""),
                "date": date,
                "seq": seq,
                "person": row.get("person", ""),
                "person_idx": row.get("person_idx", ""),
                "retarget_variant_id": row.get("retarget_variant_id", ""),
                "target_variant_id": row.get("target_variant_id", ""),
                "candidate_decision": row.get("candidate_decision", ""),
                "handoff_decision": row.get("handoff_decision", ""),
                "cem_status": row.get("cem_status", ""),
                "rl_status": "not_run",
                "downstream_decision": row.get("downstream_decision", ""),
                "downstream_failure_mode": row.get("downstream_failure_mode", ""),
                "downstream_notes": notes,
                "cem_run_id": row.get("variant", ""),
                "cem_result_npz": row.get("root_npz_path", ""),
                "cem_video": row.get("video_path", ""),
                "cem_metrics_ref": rel(summary_path),
                "rl_run_id": "",
                "rl_checkpoint": "",
                "rl_metrics_ref": "",
                "rl_video": "",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["smoke", "full"], default="full")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    results_dir = repo_path_from_env("RESULTS", args.results_dir or REPO / f"workspace/core4d/results/E144/cem/{args.stage}")
    out_dir = args.out_dir or OUT_ROOT / args.stage
    out_dir = out_dir if out_dir.is_absolute() else REPO / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = read_tsv(VARIANTS_TSV)
    missing = missing_outputs(args.stage, variants, results_dir)
    if missing and not args.allow_missing:
        print("E144 eval is deferred until all outputs exist. Missing:")
        for row in missing:
            print(f"- {row['variant']}: {row['missing']}")
        raise SystemExit(1)

    modules = (load_module("eval_E090_for_E144", E090_EVAL), load_module("eval_E105_for_E144", E105_EVAL))
    metric_rows = []
    for row in variants:
        metrics = compute_metrics(modules, row, args.stage, results_dir)
        if metrics is not None:
            metric_rows.append(metrics)

    write_tsv(out_dir / "e144_raw_mask_ref_fk_full_cem_metrics.tsv", metric_rows, METRIC_FIELDS)
    if missing:
        write_tsv(out_dir / "e144_missing_outputs.tsv", missing, ["variant", "case_id", "missing"])
    evidence = evidence_rows(metric_rows, out_dir / "e144_raw_mask_ref_fk_full_cem_metrics.tsv")
    write_tsv(out_dir / "e144_downstream_evidence_input.tsv", evidence, EVIDENCE_FIELDS)

    rl_candidates = []
    for row in metric_rows:
        rl_candidates.append(
            {
                **{field: row.get(field, "") for field in FIELDS},
                "cem_status": row.get("cem_status", ""),
                "downstream_decision": row.get("downstream_decision", ""),
                "rl_export_decision": "RL_EXPORT_READY" if row.get("cem_status") == "pass" else "SKIP_CEM_FAIL",
                "skip_reason": "" if row.get("cem_status") == "pass" else row.get("downstream_failure_mode", "cem_fail"),
                "cem_result_npz": row.get("root_npz_path", ""),
                "cem_video": row.get("video_path", ""),
            }
        )
    rl_fields = [*FIELDS, "cem_status", "downstream_decision", "rl_export_decision", "skip_reason", "cem_result_npz", "cem_video"]
    write_tsv(out_dir / "e144_rl_export_candidates.tsv", rl_candidates, rl_fields)

    summary = {
        "stage": args.stage,
        "manifest_rows": len(variants),
        "evaluated_rows": len(metric_rows),
        "missing_rows": len(missing),
        "cem_status_counts": dict(Counter(row.get("cem_status", "") for row in metric_rows)),
        "downstream_decision_counts": dict(Counter(row.get("downstream_decision", "") for row in metric_rows)),
        "rl_export_decision_counts": dict(Counter(row.get("rl_export_decision", "") for row in rl_candidates)),
        "metrics_tsv": rel(out_dir / "e144_raw_mask_ref_fk_full_cem_metrics.tsv"),
        "downstream_evidence_input_tsv": rel(out_dir / "e144_downstream_evidence_input.tsv"),
        "rl_export_candidates_tsv": rel(out_dir / "e144_rl_export_candidates.tsv"),
    }
    (out_dir / "e144_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# E144 Raw-Mask Ref-FK Full-CEM Eval Summary",
        "",
        f"- stage: `{args.stage}`",
        f"- manifest rows: `{len(variants)}`",
        f"- evaluated rows: `{len(metric_rows)}`",
        f"- missing rows: `{len(missing)}`",
        "",
        "## CEM Status",
        "",
        "| status | count |",
        "|---|---:|",
    ]
    for key, count in summary["cem_status_counts"].items():
        lines.append(f"| `{key}` | {count} |")
    lines.extend(["", "## RL Export Candidates", "", "| decision | count |", "|---|---:|"])
    for key, count in summary["rl_export_decision_counts"].items():
        lines.append(f"| `{key}` | {count} |")
    if missing:
        lines.extend(["", "## Missing Outputs", ""])
        for row in missing:
            lines.append(f"- `{row['variant']}`: {row['missing']}")
    (out_dir / "e144_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
