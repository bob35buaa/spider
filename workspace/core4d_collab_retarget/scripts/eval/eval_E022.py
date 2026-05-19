#!/usr/bin/env python3
"""E022 evaluation for box023_p1 contact-mask semantics repair."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
SCRIPT_EVAL = REPO / "workspace/core4d_collab_retarget/scripts/eval"
if str(SCRIPT_EVAL) not in sys.path:
    sys.path.insert(0, str(SCRIPT_EVAL))

import eval_E018b as e018b  # noqa: E402


BASE = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
RESULTS = REPO / "workspace/core4d_collab_retarget/results/E022"
MANIFEST = RESULTS / "manifest.tsv"
E018B_MASK = (
    REPO
    / "workspace/core4d_collab_retarget/results/E018b/contact_masks/box023_person1/raw_contact_mask_3cm.npz"
)


def _read_manifest() -> dict[str, dict[str, str]]:
    with MANIFEST.open("r", encoding="utf-8", newline="") as f:
        return {row["variant"]: row for row in csv.DictReader(f, delimiter="\t")}


def _resize_mask(mask: np.ndarray, target_len: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if len(mask) == target_len:
        return mask
    if len(mask) == 0:
        return np.zeros((target_len,) + mask.shape[1:], dtype=bool)
    idx = np.round(np.linspace(0, len(mask) - 1, target_len)).astype(int)
    return mask[idx]


def _load_ref_contact(task: str, target_len: int) -> np.ndarray:
    path = BASE / task / "0/trajectory_kinematic.npz"
    data = np.load(path, allow_pickle=True)
    contact = data["contact"][:, :2].astype(bool)
    return _resize_mask(contact, target_len)


def _mask_key(row: dict[str, str], data: np.lib.npyio.NpzFile) -> str:
    axis = row.get("contact_hdmi_mask_time_axis", "auto")
    if axis in {"spider", "eval"}:
        key = f"{axis}_contact_mask_3cm"
        if key in data:
            return key
    if "eval_contact_mask_3cm" in data:
        return "eval_contact_mask_3cm"
    return "spider_contact_mask_3cm"


def _mask_metrics(row: dict[str, str]) -> dict[str, Any]:
    mask_path = REPO / row["contact_hdmi_mask_path"]
    data = np.load(mask_path, allow_pickle=True)
    key = _mask_key(row, data)
    person_idx = int(row["person_idx"])
    used = data[key][:, person_idx, :2].astype(bool)
    current = _load_ref_contact(row["derived_task"], len(used))
    mismatch = float(np.mean(current != used) * 100.0)
    overclaim = float(np.mean(np.logical_and(current, ~used)) * 100.0)
    underclaim = float(np.mean(np.logical_and(~current, used)) * 100.0)

    source = np.load(E018B_MASK, allow_pickle=True)
    source_raw = source["spider_contact_mask_3cm"][:, person_idx, :2].astype(bool)
    source_aligned = _resize_mask(source_raw, len(current))
    source_mismatch = float(np.mean(current != source_aligned) * 100.0)
    source_overclaim = float(np.mean(np.logical_and(current, ~source_aligned)) * 100.0)

    return {
        "E022_mask_key": key,
        "E022_mask_mode": row.get("E022_mask_mode", ""),
        "E022_mask_axis": row.get("E022_mask_axis", ""),
        "E022_contact_patch_applied": row.get("E022_contact_patch_applied", ""),
        "E022_used_mask_left_pct": float(used[:, 0].mean() * 100.0),
        "E022_used_mask_right_pct": float(used[:, 1].mean() * 100.0),
        "E022_used_mask_any_pct": float(used.any(axis=1).mean() * 100.0),
        "E022_ref_contact_left_pct": float(current[:, 0].mean() * 100.0),
        "E022_ref_contact_right_pct": float(current[:, 1].mean() * 100.0),
        "E022_ref_contact_any_pct": float(current.any(axis=1).mean() * 100.0),
        "E022_mask_mismatch_pct": mismatch,
        "E022_mask_overclaim_pct": overclaim,
        "E022_mask_underclaim_pct": underclaim,
        "E022_source_raw_mismatch_pct": source_mismatch,
        "E022_source_raw_overclaim_pct": source_overclaim,
        "E022_mask_semantics_pass": bool(overclaim < 20.0 and mismatch < 20.0),
    }


def _write_summary(summary: dict[str, Any]) -> None:
    variant = str(summary["variant"])
    (RESULTS / f"eval_summary_{variant}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (RESULTS / f"eval_summary_{variant}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def evaluate_variant(variant: str, manifest: dict[str, dict[str, str]]) -> dict[str, Any]:
    e018b.RESULTS = RESULTS
    e018b.MANIFEST = MANIFEST
    variants = e018b.read_manifest()
    summary = e018b.evaluate_variant(variant, variants)
    row = manifest[variant]
    summary.update(_mask_metrics(row))
    summary["E022_contact_goal_pass"] = bool(
        float(summary.get("paper_omniretarget_contact_preservation_5cm_pct", 0.0)) >= 70.0
    )
    summary["E022_object_no_regression_pass"] = bool(
        float(summary.get("paper_object_Epos_case_m", 999.0)) < 0.10
        and float(summary.get("paper_object_Erot_case_deg", 999.0)) < 25.0
        and bool(summary.get("paper_transport_success", False))
    )
    summary["E022_artifact_no_regression_pass"] = bool(
        not bool(summary.get("E018b_robot_fall_detected", True))
        and float(summary.get("paper_omniretarget_robot_object_deep_penetration_duration_pct", 999.0))
        <= 10.0
    )
    summary["E022_success"] = bool(
        summary["E022_mask_semantics_pass"]
        and summary["E022_contact_goal_pass"]
        and summary["E022_object_no_regression_pass"]
        and summary["E022_artifact_no_regression_pass"]
    )
    _write_summary(summary)
    return summary


def _normalize_args(args: list[str], manifest: dict[str, dict[str, str]]) -> list[str]:
    if not args or args == ["--all"]:
        return list(manifest.keys())
    out: list[str] = []
    for arg in args:
        if arg == "--all":
            out.extend(manifest.keys())
        else:
            out.append(arg)
    return out


def main() -> None:
    manifest = _read_manifest()
    selected = _normalize_args(sys.argv[1:], manifest)
    summaries: list[dict[str, Any]] = []
    for variant in selected:
        if variant not in manifest:
            print(f"[SKIP] unknown variant {variant}")
            continue
        try:
            summaries.append(evaluate_variant(variant, manifest))
        except FileNotFoundError as exc:
            print(f"[SKIP] {exc}")
            continue
    if not summaries:
        raise SystemExit("No E022 variant results found.")

    keys = sorted({key for row in summaries for key in row.keys()})
    comparison = RESULTS / "comparison.csv"
    with comparison.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summaries)

    aggregate = {
        "num_results": len(summaries),
        "num_E022_success": sum(bool(r["E022_success"]) for r in summaries),
        "num_mask_semantics_pass": sum(bool(r["E022_mask_semantics_pass"]) for r in summaries),
        "num_contact_goal_pass": sum(bool(r["E022_contact_goal_pass"]) for r in summaries),
        "num_object_no_regression_pass": sum(bool(r["E022_object_no_regression_pass"]) for r in summaries),
        "num_artifact_no_regression_pass": sum(bool(r["E022_artifact_no_regression_pass"]) for r in summaries),
        "best_contact_pct": max(
            float(r.get("paper_omniretarget_contact_preservation_5cm_pct", 0.0))
            for r in summaries
        ),
        "best_variant_by_contact": max(
            summaries,
            key=lambda r: float(r.get("paper_omniretarget_contact_preservation_5cm_pct", 0.0)),
        )["variant"],
        "variants": [str(r["variant"]) for r in summaries],
    }
    (RESULTS / "aggregate_summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Wrote {comparison}")
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
