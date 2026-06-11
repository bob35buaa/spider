#!/usr/bin/env python3
"""Evaluate E091 minimal SPIDER smoke rollouts."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
TASK_ROOT = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
DEFAULT_VARIANTS = REPO / "workspace/core4d/scripts/E091/variants_smoke.tsv"
EVAL_SUMMARY = REPO / "workspace/core4d/results/E091/eval_summary.json"
E090_EVAL = REPO / "workspace/core4d/scripts/eval/eval_E090.py"


def _repo_path_from_env(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    path = Path(raw) if raw else default
    return path if path.is_absolute() else REPO / path


def _load_e090_eval():
    spec = importlib.util.spec_from_file_location("eval_E090_for_E091", E090_EVAL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {E090_EVAL}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_variants(path: Path) -> dict[str, dict[str, str]]:
    fieldnames = [
        "variant",
        "source_task",
        "derived_task",
        "mask_source_dir",
        "mask_slug",
        "person_idx",
        "split",
        "role",
    ]
    out: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(
            (line for line in f if line.strip() and not line.startswith("#")),
            delimiter="\t",
            fieldnames=fieldnames,
        )
        for row in reader:
            out[row["variant"]] = row
    return out


def _print_table(results: dict[str, dict[str, object]]) -> None:
    print("=" * 112)
    print(
        f"{'variant':<38} {'T':>4} {'cont':>7} {'obj_mean':>9} {'pelv_min':>9} "
        f"{'head':>7} {'upper':>7} {'LH_fl':>7} {'RH_fl':>7} {'pass':>6}"
    )
    print("-" * 112)
    for variant, m in results.items():
        print(
            f"{variant:<38} {int(m['T']):>4} "
            f"{float(m['contact_frac_either']) * 100:>6.1f}% "
            f"{float(m['obj_err_mean_m']):>8.3f}m "
            f"{float(m['pelvis_min_m']):>8.3f}m "
            f"{float(m['head_pen_frac']) * 100:>6.1f}% "
            f"{float(m['upper_pen_frac']) * 100:>6.1f}% "
            f"{float(m['handL_floor_lt_5cm_frac']) * 100:>6.1f}% "
            f"{float(m['handR_floor_lt_5cm_frac']) * 100:>6.1f}% "
            f"{str(m['stage_pass']):>6}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["smoke"], default="smoke")
    parser.add_argument("variants", nargs="*")
    args = parser.parse_args()

    eval_e090 = _load_e090_eval()
    default_results = REPO / f"workspace/core4d/results/E091/{args.stage}"
    results_dir = _repo_path_from_env("RESULTS", default_results)
    variants_file = _repo_path_from_env("VARIANTS_FILE", DEFAULT_VARIANTS)
    variants = _read_variants(variants_file)
    selected = args.variants or list(variants)

    results: dict[str, dict[str, object]] = {}
    for variant in selected:
        if variant not in variants:
            print(f"[SKIP] unknown E091 variant {variant}")
            continue
        row = variants[variant]
        npz = results_dir / f"{variant}_outdir_{args.stage}/trajectory_mjwp_act.npz"
        scene = TASK_ROOT / row["derived_task"] / "scene_act.xml"
        if not npz.is_file():
            print(f"[SKIP] missing rollout {npz}")
            continue
        metrics = eval_e090.compute_sim_metrics(npz, scene)
        metrics.update(
            {
                "variant": variant,
                "source_task": row["source_task"],
                "derived_task": row["derived_task"],
                "npz_path": str(npz.relative_to(REPO)),
                "scene_xml": str(scene.relative_to(REPO)),
                "stage": args.stage,
            }
        )
        metrics["smoke_collision_pass"] = eval_e090._passes_stage(args.stage, metrics)
        metrics["pelvis_ok_for_smoke"] = float(metrics["pelvis_min_m"]) >= 0.55
        metrics["pelvis_collapse_warning"] = not bool(metrics["pelvis_ok_for_smoke"])
        metrics["stage_pass"] = bool(metrics["smoke_collision_pass"] and metrics["pelvis_ok_for_smoke"])
        metrics["smoke_pass"] = metrics["stage_pass"]
        results[variant] = metrics

    if not results:
        raise SystemExit("No E091 smoke rollout results found.")

    results_dir.mkdir(parents=True, exist_ok=True)
    out_json = results_dir / f"{args.stage}_eval_summary.json"
    out_json.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    keys = sorted({key for row in results.values() for key in row})
    out_csv = results_dir / f"{args.stage}_eval_summary.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results.values())

    summary = json.loads(EVAL_SUMMARY.read_text(encoding="utf-8")) if EVAL_SUMMARY.is_file() else {}
    summary["E091_SMOKE"] = {
        "results": results,
        "num_results": len(results),
        "num_pass": sum(bool(row["stage_pass"]) for row in results.values()),
    }
    EVAL_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    EVAL_SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    _print_table(results)
    print(f"wrote {out_json.relative_to(REPO)}")
    print(f"wrote {out_csv.relative_to(REPO)}")
    print(f"updated {EVAL_SUMMARY.relative_to(REPO)}")


if __name__ == "__main__":
    main()
