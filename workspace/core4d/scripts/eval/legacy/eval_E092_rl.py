#!/usr/bin/env python3
"""Evaluate E092 Stage B/C training rollouts.

The current repository-visible training entry for E092 is the existing MJWP
stack. This evaluator keeps the route names `rl_omni` and `rl_spider` from the
plan while applying the same safety/object/pelvis metrics used by E090/E091.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
TASK_ROOT = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
BASE_VARIANTS = REPO / "workspace/core4d/scripts/E092/variants.tsv"
SPIDER_VARIANTS = REPO / "workspace/core4d/scripts/E092/variants_rl_spider.tsv"
E090_EVAL = REPO / "workspace/core4d/scripts/eval/eval_E090.py"
EVAL_SUMMARY = REPO / "workspace/core4d/results/E092/eval_summary.json"

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
]

ROUTE_DIR = {
    "rl_omni": "rl_from_omni",
    "rl_spider": "rl_from_spider",
}


def _repo_path_from_env(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    path = Path(raw) if raw else default
    return path if path.is_absolute() else REPO / path


def _load_e090_eval():
    spec = importlib.util.spec_from_file_location("eval_E090_for_E092_rl", E090_EVAL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {E090_EVAL}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_variants(path: Path, route: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(
            (line for line in f if line.strip() and not line.startswith("#")),
            delimiter="\t",
            fieldnames=FIELDS,
        )
        for row in reader:
            if row["route"] == route:
                out[row["variant"]] = row
    return out


def _status(stage: str, metrics: dict[str, object]) -> tuple[str, bool]:
    T = int(metrics["T"])
    obj_mean = float(metrics["obj_err_mean_m"])
    pelvis = float(metrics["pelvis_min_m"])
    head = float(metrics["head_pen_frac"])
    upper = float(metrics["upper_pen_frac"])
    lh_floor = float(metrics["handL_floor_lt_5cm_frac"])
    rh_floor = float(metrics["handR_floor_lt_5cm_frac"])
    if stage == "smoke":
        passed = (
            T >= 80
            and obj_mean <= 0.15
            and pelvis >= 0.45
            and head <= 0.10
            and upper <= 0.10
            and lh_floor <= 0.10
            and rh_floor <= 0.10
        )
        return ("PASS", True) if passed else ("FAIL", False)
    passed = (
        T >= 80
        and obj_mean <= 0.10
        and pelvis >= 0.55
        and head <= 0.05
        and upper <= 0.05
        and lh_floor <= 0.05
        and rh_floor <= 0.05
    )
    return ("PASS", True) if passed else ("FAIL", False)


def _print_table(route: str, stage: str, results: dict[str, dict[str, object]]) -> None:
    print("=" * 120)
    print(f"E092 {route} {stage}")
    print(
        f"{'variant':<36} {'case':<3} {'T':>4} {'cont':>7} {'obj_mean':>9} "
        f"{'pelv':>7} {'head':>7} {'upper':>7} {'LH':>7} {'RH':>7} {'status':>7}"
    )
    print("-" * 120)
    for variant, row in results.items():
        print(
            f"{variant:<36} {row['case_id']:<3} {int(row['T']):>4} "
            f"{float(row['contact_frac_either']) * 100:>6.1f}% "
            f"{float(row['obj_err_mean_m']):>8.3f}m "
            f"{float(row['pelvis_min_m']):>6.3f}m "
            f"{float(row['head_pen_frac']) * 100:>6.1f}% "
            f"{float(row['upper_pen_frac']) * 100:>6.1f}% "
            f"{float(row['handL_floor_lt_5cm_frac']) * 100:>6.1f}% "
            f"{float(row['handR_floor_lt_5cm_frac']) * 100:>6.1f}% "
            f"{row['rl_status']:>7}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", choices=["rl_omni", "rl_spider"], required=True)
    parser.add_argument("--stage", choices=["smoke", "main"], default="smoke")
    parser.add_argument("variants", nargs="*")
    args = parser.parse_args()

    eval_e090 = _load_e090_eval()
    variants_file_default = SPIDER_VARIANTS if args.route == "rl_spider" else BASE_VARIANTS
    variants_file = _repo_path_from_env("VARIANTS_FILE", variants_file_default)
    variants = _read_variants(variants_file, args.route)
    selected = args.variants or list(variants)
    route_dir = ROUTE_DIR[args.route]
    default_results = REPO / f"workspace/core4d/results/E092/{route_dir}/{args.stage}"
    results_dir = _repo_path_from_env("RESULTS", default_results)

    results: dict[str, dict[str, object]] = {}
    for variant in selected:
        row = variants.get(variant)
        if row is None:
            print(f"[SKIP] unknown E092 {args.route} variant {variant}")
            continue
        npz = results_dir / f"{variant}_outdir_{args.stage}/trajectory_mjwp_act.npz"
        scene = TASK_ROOT / row["derived_task"] / "scene_act.xml"
        if not npz.is_file():
            print(f"[SKIP] missing rollout {npz}")
            continue
        metrics = eval_e090.compute_sim_metrics(npz, scene)
        rl_status, stage_pass = _status(args.stage, metrics)
        metrics.update(
            {
                "route": args.route,
                "stage": args.stage,
                "variant": variant,
                "case_id": row["case_id"],
                "object": row["object"],
                "source_task": row["source_task"],
                "derived_task": row["derived_task"],
                "split": row["split"],
                "role": row["role"],
                "npz_path": str(npz.relative_to(REPO)),
                "scene_xml": str(scene.relative_to(REPO)),
                "rl_status": rl_status,
                "stage_pass": stage_pass,
            }
        )
        results[variant] = metrics

    if not results:
        raise SystemExit(f"No E092 {args.route} {args.stage} rollout results found.")

    results_dir.mkdir(parents=True, exist_ok=True)
    out_json = results_dir / f"{args.stage}_eval_summary.json"
    out_json.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    keys = sorted({key for row in results.values() for key in row})
    out_csv = results_dir / f"{args.stage}_eval_summary.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results.values())
    out_md = results_dir / f"{args.stage}_eval_summary.md"
    lines = [
        f"# E092 {args.route} {args.stage} summary",
        "",
        "| variant | case | T | contact | obj_mean | pelvis | head | upper | LH floor | RH floor | status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for variant, row in results.items():
        lines.append(
            f"| `{variant}` | {row['case_id']} | {row['T']} | "
            f"{float(row['contact_frac_either']) * 100:.1f}% | "
            f"{float(row['obj_err_mean_m']):.3f}m | "
            f"{float(row['pelvis_min_m']):.3f}m | "
            f"{float(row['head_pen_frac']) * 100:.1f}% | "
            f"{float(row['upper_pen_frac']) * 100:.1f}% | "
            f"{float(row['handL_floor_lt_5cm_frac']) * 100:.1f}% | "
            f"{float(row['handR_floor_lt_5cm_frac']) * 100:.1f}% | "
            f"{row['rl_status']} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = json.loads(EVAL_SUMMARY.read_text(encoding="utf-8")) if EVAL_SUMMARY.is_file() else {}
    summary[f"{args.route}_{args.stage}"] = {
        "num_results": len(results),
        "num_pass": sum(bool(row["stage_pass"]) for row in results.values()),
        "results": results,
    }
    EVAL_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    EVAL_SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    _print_table(args.route, args.stage, results)
    print(f"wrote {out_json.relative_to(REPO)}")
    print(f"wrote {out_csv.relative_to(REPO)}")
    print(f"wrote {out_md.relative_to(REPO)}")
    print(f"updated {EVAL_SUMMARY.relative_to(REPO)}")


if __name__ == "__main__":
    main()
