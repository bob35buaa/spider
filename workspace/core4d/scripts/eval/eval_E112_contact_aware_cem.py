#!/usr/bin/env python3
"""Evaluate E112 contact-aware CEM Phase A outputs."""

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
TASK_ROOT = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
VARIANTS_TSV = REPO / "workspace/core4d/scripts/E112/variants.tsv"
E090_EVAL = REPO / "workspace/core4d/scripts/eval/eval_E090.py"
E105_EVAL = REPO / "workspace/core4d/scripts/eval/eval_E105_box026_clean_cem.py"

FIELDS = [
    "ordinal",
    "variant",
    "source_task",
    "derived_task",
    "person_idx",
    "split",
    "case_id",
    "object_key",
    "ablation",
    "mask_path",
    "source_override",
]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def repo_path_from_env(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    path = Path(raw) if raw else default
    return path if path.is_absolute() else REPO / path


def read_variants() -> list[dict[str, str]]:
    with VARIANTS_TSV.open("r", encoding="utf-8", newline="") as f:
        lines = (line for line in f if line.strip() and not line.startswith("#"))
        return list(csv.DictReader(lines, fieldnames=FIELDS, delimiter="\t"))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expected_outputs(stage: str, row: dict[str, str], results_dir: Path) -> list[Path]:
    variant = row["variant"]
    return [
        results_dir / f"{variant}.npz",
        results_dir / f"{variant}_{stage}.mp4",
        results_dir / f"{variant}_outdir_{stage}/trajectory_mjwp_act.npz",
    ]


def fail_if_missing(stage: str, rows: list[dict[str, str]], results_dir: Path, allow_missing: bool) -> list[dict[str, str]]:
    missing_rows: list[dict[str, str]] = []
    for row in rows:
        missing = [path for path in expected_outputs(stage, row, results_dir) if not path.is_file()]
        if missing:
            missing_rows.append(
                {
                    "variant": row["variant"],
                    "case_id": row["case_id"],
                    "ablation": row["ablation"],
                    "missing": ";".join(rel(path) for path in missing),
                }
            )
    if missing_rows and not allow_missing:
        print("E112 eval is deferred until all CEM outputs exist. Missing:")
        for row in missing_rows:
            print(f"- {row['variant']}: {row['missing']}")
        raise SystemExit(1)
    return missing_rows


def evaluate_rollouts(stage: str, rows: list[dict[str, str]], results_dir: Path) -> list[dict[str, Any]]:
    eval_e090 = load_module("eval_E090_for_E112", E090_EVAL)
    eval_e105 = load_module("eval_E105_for_E112", E105_EVAL)
    out: list[dict[str, Any]] = []
    for row in rows:
        variant = row["variant"]
        npz = results_dir / f"{variant}_outdir_{stage}/trajectory_mjwp_act.npz"
        if not npz.is_file():
            continue
        scene = TASK_ROOT / row["derived_task"] / "scene_act.xml"
        metrics = eval_e090.compute_sim_metrics(npz, scene)
        metrics.update(eval_e105.replay_metrics(npz, scene))
        metrics.update(eval_e105.leg_object_metrics(variant, npz, scene, results_dir))
        work_status, stage_pass = eval_e105.status(stage, metrics)
        lowerbody_pass = bool(metrics["lowerbody_strict_pass"])
        strict_status = "WORK" if work_status == "WORK" and lowerbody_pass else "FAIL"
        out.append(
            {
                **row,
                **metrics,
                "stage": stage,
                "npz_path": rel(npz),
                "root_npz_path": rel(results_dir / f"{variant}.npz"),
                "video_path": rel(results_dir / f"{variant}_{stage}.mp4"),
                "scene_xml": rel(scene),
                "work_status": work_status,
                "work_status_lowerbody_strict": strict_status,
                "stage_pass": stage_pass,
                "advance_to_phase_b_candidate": bool(stage == "full" and strict_status == "WORK"),
            }
        )
    return out


def add_baseline_deltas(rows: list[dict[str, Any]]) -> None:
    by_case = {(row["case_id"], row["ablation"]): row for row in rows}
    delta_fields = [
        "contact_frac_either",
        "obj_err_mean_m",
        "obj_err_max_m",
        "pelvis_min_m",
        "leg_box_interference_frac",
        "hand_geom_deep_penetration_2cm",
        "hand_object_physics_contact",
    ]
    for row in rows:
        base = by_case.get((row["case_id"], "baseline_ref_fk"))
        for field in delta_fields:
            if base is None or field not in row or field not in base:
                row[f"delta_vs_baseline_{field}"] = ""
                continue
            try:
                row[f"delta_vs_baseline_{field}"] = float(row[field]) - float(base[field])
            except (TypeError, ValueError):
                row[f"delta_vs_baseline_{field}"] = ""


def write_outputs(
    stage: str,
    rows: list[dict[str, Any]],
    missing: list[dict[str, str]],
    results_dir: Path,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    add_baseline_deltas(rows)
    (results_dir / f"{stage}_eval_summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if rows:
        fields = sorted({key for row in rows for key in row})
        with (results_dir / f"{stage}_eval_summary.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    if missing:
        fields = ["variant", "case_id", "ablation", "missing"]
        with (results_dir / f"{stage}_missing_outputs.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(missing)

    lines = [
        f"# E112 Contact-Aware CEM {stage} Summary",
        "",
        f"- evaluated variants: `{len(rows)}`",
        f"- missing variants: `{len(missing)}`",
        f"- variants file: `{rel(VARIANTS_TSV)}`",
        "",
        "| case | ablation | contact | delta contact | obj mean | pelvis min | leg int | strict | phase-B |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in sorted(rows, key=lambda item: (item["case_id"], item["ablation"])):
        delta_contact = row.get("delta_vs_baseline_contact_frac_either", "")
        delta_s = "" if delta_contact == "" else f"{float(delta_contact) * 100:+.1f}%"
        lines.append(
            f"| `{row['case_id']}` | `{row['ablation']}` | "
            f"{float(row['contact_frac_either']) * 100:.1f}% | {delta_s} | "
            f"{float(row['obj_err_mean_m']):.3f}m | {float(row['pelvis_min_m']):.3f}m | "
            f"{float(row['leg_box_interference_frac']) * 100:.1f}% | "
            f"`{row['work_status_lowerbody_strict']}` | "
            f"{'YES' if row['advance_to_phase_b_candidate'] else 'NO'} |"
        )
    if missing:
        lines.extend(["", "## Missing Outputs", ""])
        for row in missing:
            lines.append(f"- `{row['variant']}`: {row['missing']}")
    (results_dir / f"{stage}_eval_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="full", choices=["smoke", "full"])
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=None)
    args = parser.parse_args()

    results_dir = repo_path_from_env(
        "RESULTS",
        args.results_dir or REPO / f"workspace/core4d/results/E112/cem/{args.stage}",
    )
    rows = read_variants()
    missing = fail_if_missing(args.stage, rows, results_dir, args.allow_missing)
    evaluated = evaluate_rollouts(args.stage, rows, results_dir)
    write_outputs(args.stage, evaluated, missing, results_dir)
    print(f"wrote {rel(results_dir / f'{args.stage}_eval_summary.md')}")


if __name__ == "__main__":
    main()
