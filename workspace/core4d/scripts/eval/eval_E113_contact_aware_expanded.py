#!/usr/bin/env python3
"""Evaluate E113 hold-band expanded-workset outputs against baseline cache."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
from typing import Any


THIS = Path(__file__).resolve()
REPO = THIS.parents[4]
TASK_ROOT = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
VARIANTS_TSV = REPO / "workspace/core4d/scripts/E113/variants.tsv"
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
    "e109_case_id",
    "object_key",
    "ablation",
    "mask_path",
    "source_override",
    "baseline_npz_path",
    "phase_scope",
    "holdout_reason",
]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def repo_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO / p


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
        baseline = repo_path(row["baseline_npz_path"])
        if not baseline.is_file():
            missing.append(baseline)
        if missing:
            missing_rows.append(
                {
                    "variant": row["variant"],
                    "case_id": row["case_id"],
                    "missing": ";".join(rel(path) for path in missing),
                }
            )
    if missing_rows and not allow_missing:
        print("E113 eval is deferred until all CEM outputs exist. Missing:")
        for row in missing_rows:
            print(f"- {row['variant']}: {row['missing']}")
        raise SystemExit(1)
    return missing_rows


def compute_metrics(
    modules: tuple[Any, Any],
    npz: Path,
    scene: Path,
    tag: str,
    results_dir: Path,
    stage: str,
) -> dict[str, Any]:
    eval_e090, eval_e105 = modules
    metrics = eval_e090.compute_sim_metrics(npz, scene)
    metrics.update(eval_e105.replay_metrics(npz, scene))
    metrics.update(eval_e105.leg_object_metrics(tag, npz, scene, results_dir))
    metrics["hand_object_physics_contact"] = metrics["hand_object_contact_physics_frac"]
    ts_path = repo_path(str(metrics["legobj_timeseries_csv"]))
    with ts_path.open("r", encoding="utf-8", newline="") as f:
        hand_sdf = [float(row["hand_box_sdf_min_m"]) for row in csv.DictReader(f)]
    metrics["hand_geom_penetration_frac"] = sum(value < 0.0 for value in hand_sdf) / len(hand_sdf)
    metrics["hand_geom_deep_penetration_2cm"] = sum(value < -0.02 for value in hand_sdf) / len(hand_sdf)
    work_status, stage_pass = eval_e105.status(stage, metrics)
    lowerbody_pass = bool(metrics["lowerbody_strict_pass"])
    strict_status = "WORK" if work_status == "WORK" and lowerbody_pass else "FAIL"
    return {
        **metrics,
        "work_status": work_status,
        "work_status_lowerbody_strict": strict_status,
        "stage_pass": stage_pass,
    }


def evaluate(stage: str, rows: list[dict[str, str]], results_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    modules = (
        load_module("eval_E090_for_E113", E090_EVAL),
        load_module("eval_E105_for_E113", E105_EVAL),
    )
    method_rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for row in rows:
        scene = TASK_ROOT / row["derived_task"] / "scene_act.xml"
        hold_npz = results_dir / f"{row['variant']}_outdir_{stage}/trajectory_mjwp_act.npz"
        baseline_npz = repo_path(row["baseline_npz_path"])
        if not hold_npz.is_file() or not baseline_npz.is_file():
            continue

        baseline_metrics = compute_metrics(modules, baseline_npz, scene, f"{row['variant']}_baseline", results_dir, stage)
        hold_metrics = compute_metrics(modules, hold_npz, scene, row["variant"], results_dir, stage)
        base_out = {
            **row,
            **baseline_metrics,
            "stage": stage,
            "method": "baseline_ref_fk_cache",
            "npz_path": rel(baseline_npz),
            "root_npz_path": rel(baseline_npz),
            "video_path": "",
            "scene_xml": rel(scene),
        }
        hold_out = {
            **row,
            **hold_metrics,
            "stage": stage,
            "method": "hold_band",
            "npz_path": rel(hold_npz),
            "root_npz_path": rel(results_dir / f"{row['variant']}.npz"),
            "video_path": rel(results_dir / f"{row['variant']}_{stage}.mp4"),
            "scene_xml": rel(scene),
        }
        method_rows.extend([base_out, hold_out])

        def delta(field: str) -> float:
            return float(hold_metrics[field]) - float(baseline_metrics[field])

        contact_delta = delta("contact_frac_either")
        deep_delta = delta("hand_geom_deep_penetration_2cm")
        physics_delta = delta("hand_object_physics_contact")
        pelvis_delta = delta("pelvis_min_m")
        leg_delta = delta("leg_box_interference_frac")
        obj_delta = delta("obj_err_mean_m")
        contact_ok = contact_delta >= 0.08 or physics_delta >= 0.08
        penetration_ok = deep_delta <= 0.03
        pelvis_ok = pelvis_delta >= -0.05 and not bool(hold_metrics.get("replay_gate_fail", False))
        lowerbody_ok = bool(hold_metrics["lowerbody_strict_pass"]) and float(hold_metrics["leg_box_interference_frac"]) <= 0.05
        object_ok = float(hold_metrics["obj_err_mean_m"]) <= max(float(baseline_metrics["obj_err_mean_m"]) + 0.01, 0.025)
        strict_work = hold_metrics["work_status_lowerbody_strict"] == "WORK"
        release = (
            row["phase_scope"] == "phaseA_release_candidate"
            and strict_work
            and contact_ok
            and penetration_ok
            and pelvis_ok
            and lowerbody_ok
            and object_ok
        )
        if release:
            decision = "release_candidate"
        elif contact_ok and not lowerbody_ok:
            decision = "contact_good_lowerbody_fail"
        elif contact_ok and not penetration_ok:
            decision = "contact_good_penetration_fail"
        elif not contact_ok:
            decision = "contact_not_improved"
        else:
            decision = "review"
        decisions.append(
            {
                **row,
                "stage": stage,
                "baseline_npz_path": rel(baseline_npz),
                "hold_npz_path": rel(hold_npz),
                "hold_video_path": rel(results_dir / f"{row['variant']}_{stage}.mp4"),
                "baseline_contact_frac_either": baseline_metrics["contact_frac_either"],
                "hold_contact_frac_either": hold_metrics["contact_frac_either"],
                "delta_contact_frac_either": contact_delta,
                "baseline_physics_contact": baseline_metrics["hand_object_physics_contact"],
                "hold_physics_contact": hold_metrics["hand_object_physics_contact"],
                "delta_physics_contact": physics_delta,
                "baseline_deep_penetration_2cm": baseline_metrics["hand_geom_deep_penetration_2cm"],
                "hold_deep_penetration_2cm": hold_metrics["hand_geom_deep_penetration_2cm"],
                "delta_deep_penetration_2cm": deep_delta,
                "baseline_pelvis_min_m": baseline_metrics["pelvis_min_m"],
                "hold_pelvis_min_m": hold_metrics["pelvis_min_m"],
                "delta_pelvis_min_m": pelvis_delta,
                "baseline_leg_interference_frac": baseline_metrics["leg_box_interference_frac"],
                "hold_leg_interference_frac": hold_metrics["leg_box_interference_frac"],
                "delta_leg_interference_frac": leg_delta,
                "baseline_obj_err_mean_m": baseline_metrics["obj_err_mean_m"],
                "hold_obj_err_mean_m": hold_metrics["obj_err_mean_m"],
                "delta_obj_err_mean_m": obj_delta,
                "hold_work_status": hold_metrics["work_status"],
                "hold_strict_status": hold_metrics["work_status_lowerbody_strict"],
                "contact_ok": contact_ok,
                "penetration_ok": penetration_ok,
                "pelvis_ok": pelvis_ok,
                "lowerbody_ok": lowerbody_ok,
                "object_ok": object_ok,
                "phase_b_release_candidate": release,
                "pareto_decision": decision,
            }
        )
    return method_rows, decisions


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    delimiter = "\t" if path.suffix == ".tsv" else ","
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    stage: str,
    method_rows: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    missing: list[dict[str, str]],
    results_dir: Path,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{stage}_eval_summary.json").write_text(
        json.dumps({"method_rows": method_rows, "decisions": decisions}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_table(results_dir / f"{stage}_method_metrics.csv", method_rows)
    write_table(results_dir / "pareto_decisions.tsv", decisions)
    write_table(results_dir / "release_candidates.tsv", [row for row in decisions if row["phase_b_release_candidate"]])
    write_table(
        results_dir / "contact_good_lowerbody_fail.tsv",
        [row for row in decisions if row["pareto_decision"] == "contact_good_lowerbody_fail"],
    )
    if missing:
        write_table(results_dir / f"{stage}_missing_outputs.csv", missing)

    lines = [
        f"# E113 Contact-Aware Expanded {stage} Summary",
        "",
        f"- evaluated cases: `{len(decisions)}`",
        f"- method rows: `{len(method_rows)}`",
        f"- missing variants: `{len(missing)}`",
        f"- variants file: `{rel(VARIANTS_TSV)}`",
        "",
        "| case | object | phase | contact base->hold | delta | deep pen delta | pelvis delta | leg hold | strict | decision |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in sorted(decisions, key=lambda item: (item["object_key"], item["case_id"])):
        lines.append(
            f"| `{row['case_id']}` | `{row['object_key']}` | `{row['phase_scope']}` | "
            f"{float(row['baseline_contact_frac_either']) * 100:.1f}% -> {float(row['hold_contact_frac_either']) * 100:.1f}% | "
            f"{float(row['delta_contact_frac_either']) * 100:+.1f}% | "
            f"{float(row['delta_deep_penetration_2cm']) * 100:+.1f}% | "
            f"{float(row['delta_pelvis_min_m']):+.3f}m | "
            f"{float(row['hold_leg_interference_frac']) * 100:.1f}% | "
            f"`{row['hold_strict_status']}` | `{row['pareto_decision']}` |"
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
        args.results_dir or REPO / f"workspace/core4d/results/E113/cem/{args.stage}",
    )
    rows = read_variants()
    missing = fail_if_missing(args.stage, rows, results_dir, args.allow_missing)
    method_rows, decisions = evaluate(args.stage, rows, results_dir)
    write_outputs(args.stage, method_rows, decisions, missing, results_dir)
    print(f"wrote {rel(results_dir / f'{args.stage}_eval_summary.md')}")


if __name__ == "__main__":
    main()
