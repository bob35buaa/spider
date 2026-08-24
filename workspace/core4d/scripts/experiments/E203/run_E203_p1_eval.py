#!/usr/bin/env python3
"""E203 P1 evaluation: metrics + 14-gate over the core4d_v2 (v2 human-motion)
orig retarget+CEM outputs, plus a same-case comparison against the v1-motion PRG
baselines (E170/E172/E173).

The CEM output for each core4d_v2 task lives at
``example_datasets/processed/core4d_v2/unitree_g1/humanoid_object/<task>/0/
trajectory_mjwp_act.npz`` — a single npz that already carries both the executed
qpos AND the per-iteration CEM leg-gate arrays, so it serves as both
``outdir_npz`` and ``result_npz`` for the frozen E173 evaluator.

Pipeline (all frozen contracts reused, nothing re-simulated):
  1. enumerate P1 tasks (box001/box024) that have trajectory_mjwp_act.npz,
  2. build a manifest.tsv (E173-shaped rows -> the E173 evaluator),
  3. evaluate_row (eval.core.core_metrics via eval_E173_boxes) -> full metric set
     + the 6 numeric release gates + leg-gate health,
  4. apply the E201 funnel_config 14-gate (4 hard + 10 banded wide/narrow) and the
     three-tier layer (orig-only families auto-accept when all banded pass narrow),
  5. join by physical case_id to the v1-motion PRG baselines and emit per-case
     deltas + a matched-subset summary,
  6. snapshot the exact PRG scene XMLs used (reproducibility safeguard 2).

Outputs -> workspace/core4d/results/E203/s6_downstream/eval_P1/
  e203_case_metrics.tsv       full per-case metrics + 6 gates + 14-gate columns
  e203_funnel_rollout.tsv     compact per-case 14-gate view (funnel_config order)
  e203_vs_prg_comparison.tsv  per-case E203(v2) vs prior PRG(v1) deltas
  manifest.tsv                the evaluated manifest (player + audit)
  summary.json                counts / thresholds / gate tallies / comparison
  E203_P1_eval_report.md      human-readable summary
  scene_snapshot/             frozen PRG scene XMLs + manifest.txt

Usage:
    MUJOCO_GL=osmesa .venv/bin/python \
        workspace/core4d/scripts/experiments/E203/run_E203_p1_eval.py
    # optional: --object-keys box001,box024  --out-dir <path>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
import statistics
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[5]
EVAL_DIR = REPO / "workspace/core4d/scripts/eval"
sys.path.insert(0, str(EVAL_DIR.parent))  # so `import eval.core.core_metrics` resolves
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(EVAL_DIR / "runners"))
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/experiments/E201"))

import funnel_config as FC  # noqa: E402
from eval.core.core_metrics import EVAL_METRIC_STANDARD_ID, METRIC_FIELDS, EvalConfig  # noqa: E402
from eval.core.motion_health import HEALTH_AGGS, METRIC_KEYS  # noqa: E402


def _load_e173_module():
    """Load the frozen E173 evaluator (evaluate_row) as a module."""
    path = EVAL_DIR / "runners/eval_E173_boxes.py"
    spec = importlib.util.spec_from_file_location("eval_E173_boxes", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


E173 = _load_e173_module()

DATASET = "core4d_v2"
TASK_ROOT = REPO / f"example_datasets/processed/{DATASET}/unitree_g1/humanoid_object"
PRG_SCENE = "scene_act_E199_rubberHull_PRG.xml"
DEFAULT_OUT = REPO / "workspace/core4d/results/E203/s6_downstream/eval_P1"

# Prior v1-motion PRG baselines to compare against, keyed by physical case_id.
# (exp id -> case_metrics filename; E172's file is mislabeled e171_).
PRIOR_PRG = {
    "E170": "e170_case_metrics.tsv",
    "E172": "e171_case_metrics.tsv",
    "E173": "e173_case_metrics.tsv",
}

# Metrics compared E203(v2) vs prior PRG(v1); direction = which way is better.
COMPARE_METRICS = {
    "track_root_pos_err_cm_mean": "lower",
    "track_root_ori_err_deg_mean": "lower",
    "track_eef_pos_err_cm_mean": "lower",
    "track_eef_ori_err_deg_mean": "lower",
    "track_obj_pos_err_cm_mean": "lower",
    "track_obj_ori_err_deg_mean": "lower",
    "body_z_err_p95_m": "lower",
    "hand_object_physics_contact_in_mask_frac": "higher",
    "hand_object_physics_penetration_3mm_frame_frac": "lower",
    "hand_object_release_false_contact_3mm_frac": "lower",
    "leg_penetration_frac": "lower",
    "ankle_jerk_p95": "lower",
    "obj_speed_max": "lower",
}

# funnel banded gate name -> the review-player GATE_FIELDS *_gate_pass flag.
BANDED_TO_GATEFLAG = {
    "root_pos": "root_pos_gate_pass",
    "root_ori": "root_ori_gate_pass",
    "eef_pos": "hand_pos_gate_pass",
    "eef_ori": "hand_ori_gate_pass",
    "obj_pos": "object_pos_gate_pass",
    "obj_ori": "object_ori_gate_pass",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO.resolve()))
    except ValueError:
        return str(path)


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def serial(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serial(row.get(key, "")) for key in fields})


# --- manifest construction ---------------------------------------------------

def physical_case_id(task: str) -> str:
    return re.sub(r"^dcv3_omnirt_v[12]_ref_fk_", "", task)


def object_key_of(case_id: str) -> str:
    return case_id.split("_", 1)[0]


def retarget_variant_of(task: str) -> str:
    return "omnirt_v2" if "omnirt_v2" in task else "omnirt_v1"


def contact_mask_path(task: str) -> Path:
    variant = retarget_variant_of(task)
    root = REPO / f"workspace/core4d/results/E203/s3_retarget/{variant}/ref_fk/results/{variant}_ref_fk"
    return root / "contact_masks" / task / "raw_contact_mask_3cm.npz"


def enumerate_tasks(object_keys: list[str]) -> list[str]:
    tasks = []
    for task_dir in sorted(TASK_ROOT.glob("dcv3_omnirt_v*_ref_fk_*")):
        if not (task_dir / "0/trajectory_mjwp_act.npz").is_file():
            continue
        name = task_dir.name
        if object_keys and not any(f"_{key}_" in name for key in object_keys):
            continue
        tasks.append(name)
    return tasks


def build_manifest_row(task: str, ordinal: int) -> dict[str, Any] | None:
    task_dir = TASK_ROOT / task
    act = task_dir / "0/trajectory_mjwp_act.npz"
    scene = task_dir / PRG_SCENE
    if not scene.is_file():
        scene = task_dir / "scene_act.xml"
    trajectory = task_dir / "0/trajectory_kinematic.npz"
    mask = contact_mask_path(task)
    config_act = task_dir / "0/config_act.yaml"
    video = task_dir / "0/visualization_mjwp_act.mp4"
    missing = [str(p) for p in (act, scene, trajectory, mask) if not p.is_file()]
    cid = physical_case_id(task)
    row = {
        "ordinal": ordinal,
        "case_id": cid,
        "object_key": object_key_of(cid),
        "target_task": task,
        "retarget_variant_id": retarget_variant_of(task),
        "hand_collision_variant_id": "rubber_hull",
        "spider_method_id": "E203_core4d_v2_orig_PRG",
        "source_person": "person1" if task.endswith("_p1") else "person2",
        "variant": f"E203_{cid}_PRG",
        "status": "run_complete_pending_eval",
        "result_npz": str(act),
        "outdir_npz": str(act),
        "scene_act": str(scene),
        "trajectory": str(trajectory),
        "contact_mask": str(mask),
        "config_act": str(config_act),
        "video": str(video),
        "missing_artifacts": ",".join(missing),
    }
    return row


# --- 14-gate (E201 funnel_config) --------------------------------------------

def apply_funnel(item: dict[str, Any]) -> None:
    """Attach the 14-gate result + layer to a metric row (orig-only family)."""
    hard_ok, hard_failed = FC.hard_gate_result(item)
    wide_ok, wide_failed = FC.banded_gate_result(item, "wide")
    narrow_ok, narrow_failed = FC.banded_gate_result(item, "narrow")
    if not hard_ok or not wide_ok:
        layer = "L1_reject"
    elif not narrow_ok:
        layer = "L2_review"
    else:
        # P1 is orig-only (no aug family arms) -> a lone narrow-passer is L3_auto.
        layer = "L3_auto"
    item["funnel_hard_pass"] = hard_ok
    item["funnel_hard_failed"] = ",".join(hard_failed)
    item["funnel_wide_pass"] = wide_ok
    item["funnel_wide_failed"] = ",".join(wide_failed)
    item["funnel_narrow_pass"] = narrow_ok
    item["funnel_narrow_failed"] = ",".join(narrow_failed)
    item["funnel_layer"] = layer
    item["funnel_14gate_pass"] = narrow_ok and hard_ok  # strict: all 14 at narrow
    # Populate the review-player tracking gate flags from the narrow banded result.
    narrow_failed_set = set(narrow_failed)
    for gate_name, flag in BANDED_TO_GATEFLAG.items():
        item[flag] = gate_name not in narrow_failed_set


# --- prior-PRG comparison ----------------------------------------------------

def load_prior_prg() -> dict[str, dict[str, Any]]:
    """case_id -> {source_exp, <metric>: value, numeric_release_pass}."""
    lookup: dict[str, dict[str, Any]] = {}
    for exp, filename in PRIOR_PRG.items():
        path = REPO / f"workspace/core4d/results/{exp}/s6_downstream/eval/full/{filename}"
        for row in read_tsv(path):
            cid = (row.get("case_id") or "").strip()
            if not cid or cid in lookup:
                continue  # first exp wins; P1 objects are disjoint across E170/172/173
            entry = {"source_exp": exp, "numeric_release_pass": row.get("numeric_release_pass", "")}
            for metric in COMPARE_METRICS:
                entry[metric] = finite(row.get(metric))
            lookup[cid] = entry
    return lookup


def build_comparison(metrics: list[dict[str, Any]], prior: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    matched: list[dict[str, Any]] = []
    for item in metrics:
        cid = item["case_id"]
        base = prior.get(cid)
        row: dict[str, Any] = {
            "case_id": cid,
            "object_key": item.get("object_key", ""),
            "e203_retarget_variant": item.get("retarget_variant_id", ""),
            "prior_exp": base["source_exp"] if base else "",
            "matched": bool(base),
            "e203_numeric_release_pass": item.get("numeric_release_pass"),
            "prior_numeric_release_pass": base.get("numeric_release_pass") if base else "",
            "e203_funnel_layer": item.get("funnel_layer", ""),
        }
        for metric, direction in COMPARE_METRICS.items():
            cur = finite(item.get(metric))
            old = finite(base.get(metric)) if base else math.nan
            delta = cur - old if math.isfinite(cur) and math.isfinite(old) else math.nan
            improved = ""
            if math.isfinite(delta):
                improved = (delta < 0) if direction == "lower" else (delta > 0)
            row[f"e203_{metric}"] = cur
            row[f"prior_{metric}"] = old
            row[f"delta_{metric}"] = delta
            row[f"improved_{metric}"] = improved
        rows.append(row)
        if base:
            matched.append(row)

    summary: dict[str, Any] = {
        "e203_cases": len(metrics),
        "matched_cases": len(matched),
        "unmatched_cases": len(metrics) - len(matched),
        "prior_sources": dict(Counter(r["prior_exp"] for r in matched)),
        "per_metric": {},
    }
    for metric, direction in COMPARE_METRICS.items():
        deltas = [finite(r[f"delta_{metric}"]) for r in matched]
        deltas = [d for d in deltas if math.isfinite(d)]
        improved = sum(1 for r in matched if r[f"improved_{metric}"] is True)
        summary["per_metric"][metric] = {
            "direction": direction,
            "n": len(deltas),
            "e203_mean": _mean([finite(r[f"e203_{metric}"]) for r in matched]),
            "prior_mean": _mean([finite(r[f"prior_{metric}"]) for r in matched]),
            "mean_delta": statistics.fmean(deltas) if deltas else math.nan,
            "improved": improved,
            "regressed": len(deltas) - improved,
        }
    return rows, summary


def _mean(values: list[float]) -> float:
    clean = [v for v in values if math.isfinite(v)]
    return statistics.fmean(clean) if clean else math.nan


# --- scene snapshot (reproducibility safeguard 2) ----------------------------

def snapshot_scenes(tasks: list[str], out_dir: Path) -> None:
    snap = out_dir / "scene_snapshot"
    snap.mkdir(parents=True, exist_ok=True)
    try:
        head = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO).decode().strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO).decode().strip()
    except Exception:
        head, branch = "unknown", "unknown"
    lines = [f"# E203 P1 eval scene snapshot", f"# git HEAD: {head} ({branch})", ""]
    for task in tasks:
        task_dir = TASK_ROOT / task
        for name in (PRG_SCENE, "scene_act.xml", "scene.xml", "scene_act_meta.json", "task_info.json"):
            src = task_dir / name
            if not src.is_file():
                continue
            dst_dir = snap / task
            dst_dir.mkdir(parents=True, exist_ok=True)
            (dst_dir / name).write_bytes(src.read_bytes())
            lines.append(f"{task}/{name}\t{src.stat().st_size}\t{sha256(src)}")
    (snap / "manifest.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- report ------------------------------------------------------------------

def write_report(out_dir: Path, payload: dict[str, Any], metrics: list[dict[str, Any]]) -> None:
    counts = payload["counts"]
    layers = Counter(m["funnel_layer"] for m in metrics)
    cmp = payload["comparison"]
    lines = [
        "# E203 P1 evaluation — v2 human-motion orig retarget + CEM",
        "",
        f"Generated: {payload['generated_at']}",
        f"Metric standard: `{payload['metric_standard_id']}`",
        "",
        "## Coverage",
        "",
        f"- Evaluated: **{counts['evaluated']}** / {counts['manifest_rows']} manifest rows "
        f"(not-ready {counts['not_ready']}, errors {counts['errors']})",
        f"- Retarget variants: {payload['retarget_variant_counts']}",
        f"- Objects: {payload['object_counts']}",
        "",
        "## Numeric release gate (6-gate, E173 contract)",
        "",
        f"- Pass: **{counts['numeric_pass']}** / {counts['evaluated']} "
        f"({counts['numeric_pass'] / counts['evaluated']:.0%})" if counts['evaluated'] else "- (no cases)",
        f"- Failure modes: {payload['numeric_failure_counts']}",
        "",
        "## 14-gate funnel (E201 funnel_config: 4 hard + 10 banded)",
        "",
        f"- 14-gate strict pass (all narrow): **{sum(1 for m in metrics if m['funnel_14gate_pass'])}** / {len(metrics)}",
        f"- Layers: L1_reject={layers.get('L1_reject', 0)}  L2_review={layers.get('L2_review', 0)}  "
        f"L3_auto={layers.get('L3_auto', 0)}",
        "",
        "## Comparison vs prior v1-motion PRG (E170/E172/E173), same physical case",
        "",
        f"- Matched: **{cmp['matched_cases']}** / {cmp['e203_cases']} "
        f"(unmatched {cmp['unmatched_cases']}); prior sources {cmp['prior_sources']}",
        "",
        "| metric | dir | E203 v2 mean | prior v1 mean | mean Δ | improved/n |",
        "|---|---|---|---|---|---|",
    ]
    for metric, stat in cmp["per_metric"].items():
        lines.append(
            f"| {metric} | {stat['direction']} | {stat['e203_mean']:.4g} | {stat['prior_mean']:.4g} | "
            f"{stat['mean_delta']:+.4g} | {stat['improved']}/{stat['n']} |"
        )
    lines += [
        "",
        "Δ is E203 − prior; **improved** counts cases that moved in the better direction.",
        "Same physical box001/box024 cases; the only intended difference is the retarget",
        "reference motion (E203 = CORE4D v2 human motion, prior = v1). Identical PRG",
        "reward/leg-gate contract and evaluator.",
        "",
    ]
    (out_dir / "E203_P1_eval_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- main --------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-keys", default="box001,box024",
                        help="comma-separated object keys to include (empty = all present)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--no-snapshot", action="store_true")
    args = parser.parse_args()

    FC.assert_monotonic()
    out_dir = args.out_dir
    keys = [k.strip().lower() for k in args.object_keys.split(",") if k.strip()]
    tasks = enumerate_tasks(keys)
    if not tasks:
        print("no P1 tasks with trajectory_mjwp_act.npz", file=sys.stderr)
        return 2
    print(f"[E203-P1] {len(tasks)} tasks (objects={keys or 'all'})", flush=True)

    manifest_rows = [build_manifest_row(t, i) for i, t in enumerate(tasks, 1)]
    ready = [r for r in manifest_rows if not r["missing_artifacts"]]
    not_ready = [r for r in manifest_rows if r["missing_artifacts"]]

    cfg = EvalConfig()
    metrics: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for index, row in enumerate(ready, 1):
        print(f"[{index}/{len(ready)}] {row['case_id']} ({row['retarget_variant_id']})", flush=True)
        try:
            item = E173.evaluate_row(row, cfg, {}, {})
        except Exception as exc:  # noqa: BLE001
            errors.append({"case_id": row["case_id"], "error": f"{type(exc).__name__}: {exc}"})
            continue
        # ensure player/comparison identity columns survive
        item["object_key"] = row["object_key"]
        item["retarget_variant_id"] = row["retarget_variant_id"]
        item["target_task"] = row["target_task"]
        apply_funnel(item)
        metrics.append(item)

    if not args.no_snapshot:
        snapshot_scenes([r["target_task"] for r in ready], out_dir)

    prior = load_prior_prg()
    comparison_rows, comparison_summary = build_comparison(metrics, prior)

    numeric_failures = Counter(mode for m in metrics for mode in m["numeric_failure_modes"].split(",") if mode)
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "metric_standard_id": EVAL_METRIC_STANDARD_ID,
        "dataset": DATASET,
        "phase": "P1",
        "counts": {
            "manifest_rows": len(manifest_rows),
            "evaluated": len(metrics),
            "not_ready": len(not_ready),
            "errors": len(errors),
            "numeric_pass": sum(bool(m["numeric_release_pass"]) for m in metrics),
        },
        "retarget_variant_counts": dict(Counter(m["retarget_variant_id"] for m in metrics)),
        "object_counts": dict(Counter(m["object_key"] for m in metrics)),
        "numeric_failure_counts": dict(numeric_failures),
        "funnel_14gate": {
            "strict_pass": sum(1 for m in metrics if m["funnel_14gate_pass"]),
            "layers": dict(Counter(m["funnel_layer"] for m in metrics)),
        },
        "thresholds": {
            "numeric_6gate": {
                "body_z_err_p95_m_max": E173.BODY_Z_MAX,
                "raw_contact_min": E173.CONTACT_MIN,
                "release_false_3mm_max": E173.RELEASE_MAX,
                "hand_penetration_3mm_max": E173.HAND_PEN_MAX,
                "leg_penetration_max": E173.LEG_PEN_MAX,
            },
            "funnel_hard": [[n, f, o, t] for n, f, o, t in FC.HARD_GATES],
            "funnel_banded_narrow_wide": [[n, f, o, nn, ww] for n, f, o, nn, ww in FC.BANDED_GATES],
        },
        "comparison": comparison_summary,
    }

    # --- write outputs -------------------------------------------------------
    funnel_cols = [
        "funnel_layer", "funnel_14gate_pass", "funnel_hard_pass", "funnel_hard_failed",
        "funnel_wide_pass", "funnel_wide_failed", "funnel_narrow_pass", "funnel_narrow_failed",
    ]
    priority = [
        "case_id", "object_key", "retarget_variant_id", "variant", "target_task",
        "numeric_release_pass", "numeric_failure_modes", *funnel_cols,
        "fall_gate_pass", "body_z_gate_pass", "contact_gate_pass", "release_gate_pass",
        "hand_penetration_gate_pass", "lower_body_gate_pass",
        *BANDED_TO_GATEFLAG.values(),
        "leg_gate_health_pass", *COMPARE_METRICS,
    ]
    fields: list[str] = []
    for key in [*priority, *METRIC_FIELDS, *METRIC_KEYS, *HEALTH_AGGS.keys()]:
        if key not in fields:
            fields.append(key)
    for row in metrics:
        for key in row:
            if key not in fields:
                fields.append(key)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(out_dir / "e203_case_metrics.tsv", metrics, fields)
    write_tsv(out_dir / "manifest.tsv", manifest_rows)
    if not_ready:
        write_tsv(out_dir / "e203_not_ready.tsv", not_ready)
    if errors:
        write_tsv(out_dir / "e203_evaluation_errors.tsv", errors)

    # compact 14-gate rollout (funnel_config field order, self-contained)
    gate_val_cols = ["fall_flag", "body_z_err_p95_m"]
    gate_val_cols += [f for _n, f, _o, _t in FC.HARD_GATES if f not in ("fall_flag", "body_z_err_p95_m")]
    gate_val_cols += [f for _n, f, *_ in FC.BANDED_GATES]
    rollout_cols = ["object_key", "case_id", "retarget_variant_id", "funnel_layer", *gate_val_cols,
                    "funnel_hard_pass", "funnel_hard_failed", "funnel_wide_pass", "funnel_wide_failed",
                    "funnel_narrow_pass", "funnel_narrow_failed"]
    write_tsv(out_dir / "e203_funnel_rollout.tsv", metrics, rollout_cols)

    write_tsv(out_dir / "e203_vs_prg_comparison.tsv", comparison_rows)
    (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(out_dir, payload, metrics)

    print(json.dumps(payload["counts"], sort_keys=True))
    print(f"14-gate: {payload['funnel_14gate']}")
    print(f"comparison matched: {comparison_summary['matched_cases']}/{comparison_summary['e203_cases']}"
          f" sources={comparison_summary['prior_sources']}")
    print(f"outputs -> {rel(out_dir)}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
