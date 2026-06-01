#!/usr/bin/env python3
"""Evaluate E106 Box026 clean ref-FK candidate batch after all CEM runs finish."""

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
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/E106"))

from e106_common import RESULTS_ROOT, TASK_ROOT, VARIANTS_TSV, read_variants, rel  # noqa: E402


E090_EVAL = REPO / "workspace/core4d/scripts/eval/eval_E090.py"
E105_EVAL = REPO / "workspace/core4d/scripts/eval/eval_E105_box026_clean_cem.py"
FAILURES_TSV = RESULTS_ROOT / "preprocess_failures.tsv"


def repo_path_from_env(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    path = Path(raw) if raw else default
    return path if path.is_absolute() else REPO / path


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_failures() -> list[dict[str, str]]:
    if not FAILURES_TSV.is_file():
        return []
    with FAILURES_TSV.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def expected_outputs(stage: str, variants: list[dict[str, str]], results_dir: Path) -> list[Path]:
    out = []
    for row in variants:
        out.append(results_dir / f"{row['variant']}.npz")
        out.append(results_dir / f"{row['variant']}_{stage}.mp4")
        out.append(results_dir / f"{row['variant']}_outdir_{stage}/trajectory_mjwp_act.npz")
    return out


def selected_rows(variants: list[dict[str, str]], wanted: list[str]) -> list[dict[str, str]]:
    if not wanted:
        return variants
    want = set(wanted)
    return [row for row in variants if row["variant"] in want]


def fail_if_missing_outputs(stage: str, rows: list[dict[str, str]], results_dir: Path) -> None:
    missing = [path for path in expected_outputs(stage, rows, results_dir) if not path.is_file()]
    if missing:
        print("E106 eval is deferred until all selected CEM outputs exist. Missing:")
        for path in missing[:120]:
            print(f"- {rel(path)}")
        if len(missing) > 120:
            print(f"- ... {len(missing) - 120} more")
        raise SystemExit(1)


def review_status(variant: str) -> str:
    review = RESULTS_ROOT / "pre_cem_visual_review" / variant / "REVIEW.md"
    if not review.is_file():
        return "MISSING"
    for line in review.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("Status:"):
            return line.split(":", 1)[1].strip()
    return "UNKNOWN"


def evaluate_rollouts(stage: str, rows: list[dict[str, str]], results_dir: Path) -> dict[str, dict[str, Any]]:
    eval_e090 = load_module("eval_E090_for_E106", E090_EVAL)
    eval_e105 = load_module("eval_E105_for_E106", E105_EVAL)
    results: dict[str, dict[str, Any]] = {}
    for row in rows:
        variant = row["variant"]
        npz = results_dir / f"{variant}_outdir_{stage}/trajectory_mjwp_act.npz"
        scene = TASK_ROOT / row["derived_task"] / "scene_act.xml"
        metrics = eval_e090.compute_sim_metrics(npz, scene)
        metrics.update(eval_e105.replay_metrics(npz, scene))
        metrics.update(eval_e105.leg_object_metrics(variant, npz, scene, results_dir))
        work_status, stage_pass = eval_e105.status(stage, metrics)
        lowerbody_pass = bool(metrics["lowerbody_strict_pass"])
        strict_work_status = "WORK" if work_status == "WORK" and lowerbody_pass else "FAIL"
        metrics.update(
            {
                "variant": variant,
                "ordinal": row.get("ordinal", ""),
                "source_task": row["source_task"],
                "derived_task": row["derived_task"],
                "split": row.get("split", ""),
                "route": "ref_fk_clean",
                "stage": stage,
                "npz_path": rel(npz),
                "root_npz_path": rel(results_dir / f"{variant}.npz"),
                "video_path": rel(results_dir / f"{variant}_{stage}.mp4"),
                "scene_xml": rel(scene),
                "pre_cem_review_status": review_status(variant),
                "work_status": work_status,
                "work_status_lowerbody_strict": strict_work_status,
                "stage_pass": stage_pass,
                "advance_to_rl": bool(stage == "full" and work_status == "WORK"),
                "advance_to_rl_lowerbody_strict": bool(stage == "full" and strict_work_status == "WORK"),
            }
        )
        results[variant] = metrics
    return results


def write_eval_outputs(
    stage: str,
    results: dict[str, dict[str, Any]],
    failures: list[dict[str, str]],
    results_dir: Path,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    out_json = results_dir / f"{stage}_eval_summary.json"
    out_json.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    keys = sorted({key for row in results.values() for key in row})
    out_csv = results_dir / f"{stage}_eval_summary.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results.values())

    out_reject_csv = results_dir / "preprocess_rejects.csv"
    if failures:
        fields = sorted({key for row in failures for key in row})
        with out_reject_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(failures)

    out_md = results_dir / f"{stage}_eval_summary.md"
    lines = [
        f"# E106 Box026 30-candidate clean ref-FK {stage} summary",
        "",
        f"- runnable CEM variants: `{len(results)}`",
        f"- preprocess rejects: `{len(failures)}`",
        f"- variants file: `{rel(VARIANTS_TSV)}`",
        "",
        "Status definitions reuse E105 upper-body/replay gates plus E026/E081 lower-body strict proxy.",
        "",
        "| variant | split | T | contact | obj mean | obj max | pelvis min | replay | upper status | leg interference | lower strict | RL strict |",
        "|---|---|---:|---:|---:|---:|---:|---|---|---:|---|---|",
    ]
    for variant, row in results.items():
        lines.append(
            f"| `{variant}` | `{row['split']}` | {row['T']} | "
            f"{float(row['contact_frac_either']) * 100:.1f}% | "
            f"{float(row['obj_err_mean_m']):.3f}m | "
            f"{float(row['obj_err_max_m']):.3f}m | "
            f"{float(row['pelvis_min_m']):.3f}m | "
            f"{'PASS' if row['replay_gate_pass'] else 'FAIL'} | "
            f"{row['work_status']} | "
            f"{float(row['leg_box_interference_frac']) * 100:.1f}% | "
            f"{'PASS' if row['lowerbody_strict_pass'] else 'FAIL'} | "
            f"{'YES' if row['advance_to_rl_lowerbody_strict'] else 'NO'} |"
        )

    lines.extend(
        [
            "",
            "Replay gate thresholds: pelvis_end_z>=0.55m, pelvis_tilt_end<=75deg, lie_on_box_frac<0.30.",
            "",
            "| variant | pelvis end | tilt end | lie on box | replay fail flags |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for variant, row in results.items():
        flags = ",".join(
            name
            for name in ["gate_pelvis_low", "gate_pelvis_tilt", "gate_lie_on_box"]
            if bool(row.get(name))
        )
        lines.append(
            f"| `{variant}` | {float(row['pelvis_end_z_m']):.3f}m | "
            f"{float(row['pelvis_tilt_end_deg']):.1f}deg | "
            f"{float(row['lie_on_box_frac']) * 100:.1f}% | {flags or 'none'} |"
        )

    lines.extend(
        [
            "",
            "Lower-body strict proxy follows E026/E081: leg_box_interference_frac <= 5%.",
            "",
            "| variant | leg interference | leg contact | min leg SDF | argmin geom | object-floor contact |",
            "|---|---:|---:|---:|---|---:|",
        ]
    )
    for variant, row in results.items():
        lines.append(
            f"| `{variant}` | {float(row['leg_box_interference_frac']) * 100:.1f}% | "
            f"{float(row['leg_object_contact_frac']) * 100:.1f}% | "
            f"{float(row['leg_box_sdf_min_m']):.3f}m | "
            f"`{row['leg_box_sdf_argmin']}` | "
            f"{float(row['object_floor_contact_frac']) * 100:.1f}% |"
        )

    if failures:
        lines.extend(
            [
                "",
                "Preprocess rejects are not CEM failures; they are upstream OmniRetarget/data-construction failures.",
                "",
                "| source task | variant | stage | reason |",
                "|---|---|---|---|",
            ]
        )
        for row in failures:
            lines.append(
                f"| `{row.get('source_task', '')}` | `{row.get('variant', '')}` | "
                f"{row.get('stage', '')} | {row.get('reason', '')} |"
            )

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {rel(out_json)}")
    print(f"wrote {rel(out_csv)}")
    if failures:
        print(f"wrote {rel(out_reject_csv)}")
    print(f"wrote {rel(out_md)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="full", choices=["smoke", "full"])
    parser.add_argument("variants", nargs="*")
    args = parser.parse_args()

    rows = selected_rows(read_variants(), args.variants)
    results_dir = repo_path_from_env("RESULTS", RESULTS_ROOT / "cem" / args.stage)
    fail_if_missing_outputs(args.stage, rows, results_dir)
    results = evaluate_rollouts(args.stage, rows, results_dir)
    if not results:
        raise SystemExit("No E106 rollout results found.")
    write_eval_outputs(args.stage, results, read_failures(), results_dir)


if __name__ == "__main__":
    main()
