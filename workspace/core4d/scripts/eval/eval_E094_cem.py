#!/usr/bin/env python3
"""Evaluate E094 adaptive-support CEM rollouts."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
TASK_ROOT = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
DEFAULT_VARIANTS = REPO / "workspace/core4d/scripts/E094/variants.tsv"
E090_EVAL = REPO / "workspace/core4d/scripts/eval/eval_E090.py"

FIELDS = [
    "route",
    "case_id",
    "variant",
    "source_task",
    "derived_task",
    "person_idx",
    "split",
    "object",
    "role",
    "note",
    "target_case_id",
    "target_npz",
]


def repo_path_from_env(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    path = Path(raw) if raw else default
    return path if path.is_absolute() else REPO / path


def load_e090_eval():
    spec = importlib.util.spec_from_file_location("eval_E090_for_E094", E090_EVAL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {E090_EVAL}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_variants(path: Path) -> dict[str, dict[str, str]]:
    rows = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(
            (line for line in f if line.strip() and not line.startswith("#")),
            delimiter="\t",
            fieldnames=FIELDS,
        )
        for row in reader:
            if row["route"] == "cem_hbproj":
                rows[row["variant"]] = row
    return rows


def status(stage: str, metrics: dict[str, object]) -> tuple[str, bool]:
    T = int(metrics["T"])
    obj_mean = float(metrics["obj_err_mean_m"])
    obj_max = float(metrics["obj_err_max_m"])
    pelvis = float(metrics["pelvis_min_m"])
    head = float(metrics["head_pen_frac"])
    upper = float(metrics["upper_pen_frac"])
    lh_floor = float(metrics["handL_floor_lt_5cm_frac"])
    rh_floor = float(metrics["handR_floor_lt_5cm_frac"])
    contact = float(metrics["contact_frac_either"])
    if stage == "smoke":
        passed = T >= 80 and obj_mean <= 0.15 and pelvis >= 0.45 and head <= 0.10 and upper <= 0.10 and lh_floor <= 0.10 and rh_floor <= 0.10
        if passed:
            return "PASS", True
        review = T >= 80 and obj_mean <= 0.20 and pelvis >= 0.35 and head <= 0.15 and upper <= 0.15 and max(lh_floor, rh_floor) <= 0.15
        return ("REVIEW+", False) if review else ("FAIL", False)
    work = T >= 80 and obj_mean <= 0.10 and obj_max <= 0.30 and pelvis >= 0.55 and head <= 0.05 and upper <= 0.05 and lh_floor <= 0.05 and rh_floor <= 0.05 and contact >= 0.30
    if work:
        return "WORK", True
    review = T >= 80 and obj_mean <= 0.12 and obj_max <= 0.35 and pelvis >= 0.50 and head <= 0.05 and upper <= 0.05 and lh_floor <= 0.05 and rh_floor <= 0.05
    return ("REVIEW+", False) if review else ("FAIL", False)


def print_table(results: dict[str, dict[str, object]]) -> None:
    print("=" * 128)
    print(f"{'variant':<34} {'case':<3} {'T':>4} {'cont':>7} {'obj_mean':>9} {'obj_max':>8} {'pelv':>7} {'head':>7} {'upper':>7} {'LH':>7} {'RH':>7} {'status':>8}")
    print("-" * 128)
    for variant, row in results.items():
        print(
            f"{variant:<34} {row['case_id']:<3} {int(row['T']):>4} "
            f"{float(row['contact_frac_either']) * 100:>6.1f}% "
            f"{float(row['obj_err_mean_m']):>8.3f}m "
            f"{float(row['obj_err_max_m']):>7.3f}m "
            f"{float(row['pelvis_min_m']):>6.3f}m "
            f"{float(row['head_pen_frac']) * 100:>6.1f}% "
            f"{float(row['upper_pen_frac']) * 100:>6.1f}% "
            f"{float(row['handL_floor_lt_5cm_frac']) * 100:>6.1f}% "
            f"{float(row['handR_floor_lt_5cm_frac']) * 100:>6.1f}% "
            f"{row['work_status']:>8}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["smoke", "full"], default="full")
    parser.add_argument("variants", nargs="*")
    args = parser.parse_args()

    eval_e090 = load_e090_eval()
    results_dir = repo_path_from_env("RESULTS", REPO / f"workspace/core4d/results/E094/cem/{args.stage}")
    variants_file = repo_path_from_env("VARIANTS_FILE", DEFAULT_VARIANTS)
    variants = read_variants(variants_file)
    selected = args.variants or list(variants)

    results: dict[str, dict[str, object]] = {}
    for variant in selected:
        row = variants.get(variant)
        if row is None:
            print(f"[SKIP] unknown E094 variant {variant}")
            continue
        npz = results_dir / f"{variant}_outdir_{args.stage}/trajectory_mjwp_act.npz"
        scene = TASK_ROOT / row["derived_task"] / "scene_act.xml"
        if not npz.is_file():
            print(f"[SKIP] missing rollout {npz}")
            continue
        metrics = eval_e090.compute_sim_metrics(npz, scene)
        work_status, stage_pass = status(args.stage, metrics)
        metrics.update(
            {
                "route": "cem_hbproj",
                "stage": args.stage,
                "variant": variant,
                "case_id": row["case_id"],
                "object": row["object"],
                "source_task": row["source_task"],
                "derived_task": row["derived_task"],
                "target_case_id": row["target_case_id"],
                "target_npz": row["target_npz"],
                "split": row["split"],
                "role": row["role"],
                "npz_path": str(npz.relative_to(REPO)),
                "scene_xml": str(scene.relative_to(REPO)),
                "work_status": work_status,
                "stage_pass": stage_pass,
                "advance_to_rl": bool(args.stage == "full" and work_status == "WORK"),
            }
        )
        results[variant] = metrics

    if not results:
        raise SystemExit("No E094 rollout results found.")

    results_dir.mkdir(parents=True, exist_ok=True)
    out_json = results_dir / f"{args.stage}_eval_summary.json"
    out_json.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    keys = sorted({key for row in results.values() for key in row})
    out_csv = results_dir / f"{args.stage}_eval_summary.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results.values())
    out_md = results_dir / f"{args.stage}_eval_summary.md"
    lines = [
        f"# E094 adaptive-support CEM {args.stage} summary",
        "",
        "| variant | case | T | contact | obj_mean | obj_max | pelvis | head | upper | LH floor | RH floor | status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for variant, row in results.items():
        lines.append(
            f"| `{variant}` | {row['case_id']} | {row['T']} | "
            f"{float(row['contact_frac_either']) * 100:.1f}% | "
            f"{float(row['obj_err_mean_m']):.3f}m | "
            f"{float(row['obj_err_max_m']):.3f}m | "
            f"{float(row['pelvis_min_m']):.3f}m | "
            f"{float(row['head_pen_frac']) * 100:.1f}% | "
            f"{float(row['upper_pen_frac']) * 100:.1f}% | "
            f"{float(row['handL_floor_lt_5cm_frac']) * 100:.1f}% | "
            f"{float(row['handR_floor_lt_5cm_frac']) * 100:.1f}% | "
            f"{row['work_status']} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print_table(results)
    print(f"wrote {out_json.relative_to(REPO)}")
    print(f"wrote {out_csv.relative_to(REPO)}")
    print(f"wrote {out_md.relative_to(REPO)}")


if __name__ == "__main__":
    main()
