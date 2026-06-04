#!/usr/bin/env python3
"""Evaluate E122 snap warmstart carry-prior diagnostic outputs."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


THIS = Path(__file__).resolve()
REPO = THIS.parents[4]
TASK_ROOT = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
E115_EVAL = REPO / "workspace/core4d/scripts/eval/eval_E115_lowerbody_contact.py"
E105_EVAL = REPO / "workspace/core4d/scripts/eval/eval_E105_box026_clean_cem.py"
E122_VARIANTS = REPO / "workspace/core4d/scripts/E122/variants.tsv"

LEG_GEOMS = [
    "left_hip_collision",
    "right_hip_collision",
    "left_thigh_collision",
    "right_thigh_collision",
    "left_shin_collision",
    "right_shin_collision",
    "left_linkage_brace_collision",
    "right_linkage_brace_collision",
    "lf0",
    "lf1",
    "lf2",
    "lf3",
    "rf0",
    "rf1",
    "rf2",
    "rf3",
]
ROBOT_OBJECT_GEOMS = [
    "head_collision",
    "torso_collision",
    "pelvis_collision",
    "left_shoulder_yaw_collision",
    "right_shoulder_yaw_collision",
    "left_elbow_yaw_collision",
    "right_elbow_yaw_collision",
]
NONHAND_SUPPORT_GEOMS = ROBOT_OBJECT_GEOMS + LEG_GEOMS
HAND_GEOMS = ["lh", "rh"]


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


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_e115_eval():
    module = load_module("eval_E115_reused_for_E122", E115_EVAL)
    module.VARIANTS_TSV = E122_VARIANTS

    def read_e122_variants() -> list[dict[str, str]]:
        lines = E122_VARIANTS.read_text(encoding="utf-8").splitlines()
        header: list[str] | None = None
        rows: list[dict[str, str]] = []
        for line in lines:
            if not line.strip():
                continue
            if line.startswith("#"):
                maybe = line.lstrip("# ").split("\t")
                if maybe and maybe[0] == "ordinal":
                    header = maybe
                continue
            if header is None:
                raise RuntimeError(f"{E122_VARIANTS} missing commented header")
            rows.append(dict(zip(header, line.split("\t"))))
        return rows

    module.read_variants = read_e122_variants
    return module


def load_e105_eval():
    return load_module("eval_E105_for_E122", E105_EVAL)


def trajectory_info_metrics(npz_path: Path) -> dict[str, Any]:
    if not npz_path.is_file():
        return {}
    data = np.load(npz_path, allow_pickle=True)
    out: dict[str, Any] = {}
    for key in [
        "terminal_carry_gate_violation_mean",
        "terminal_carry_gate_valid_mean",
        "terminal_carry_gate_pelvis_z_mean",
        "terminal_carry_gate_obj_rot_err_mean",
        "terminal_carry_gate_nonhand_sdf_mean",
        "terminal_carry_gate_hand_near_frac_mean",
        "cem_gate_valid_frac",
        "cem_gate_fallback_used",
        "cem_gate_selected_valid_frac",
    ]:
        if key not in data.files:
            continue
        arr = np.asarray(data[key], dtype=np.float64)
        if arr.size == 0:
            continue
        out[f"{key}_mean"] = float(np.nanmean(arr))
        out[f"{key}_min"] = float(np.nanmin(arr))
        out[f"{key}_max"] = float(np.nanmax(arr))
    return out


def support_metrics(
    e105: Any,
    tag: str,
    npz_path: Path,
    scene_xml: Path,
    results_dir: Path,
) -> dict[str, Any]:
    qpos = e105.rg.load_qpos(npz_path, channel="sim")
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    object_geom = e105._geom_id(model, "object_collision")
    half = model.geom_size[object_geom, :3].copy()
    nonhand_gids = [
        e105._geom_id(model, name)
        for name in NONHAND_SUPPORT_GEOMS
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0
    ]
    hand_gids = [
        e105._geom_id(model, name)
        for name in HAND_GEOMS
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0
    ]
    rows: list[dict[str, Any]] = []
    for frame, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        obj_pos = data.geom_xpos[object_geom].copy()
        obj_mat = data.geom_xmat[object_geom].reshape(3, 3).copy()
        nonhand_vals = [
            (
                e105._geom_box_adjusted_sdf(model, data, gid, obj_pos, obj_mat, half),
                e105._geom_name(model, gid),
            )
            for gid in nonhand_gids
        ]
        hand_vals = [
            (
                e105._geom_box_adjusted_sdf(model, data, gid, obj_pos, obj_mat, half),
                e105._geom_name(model, gid),
            )
            for gid in hand_gids
        ]
        nonhand_sdf, nonhand_arg = min(nonhand_vals, key=lambda x: x[0])
        hand_sdf, hand_arg = min(hand_vals, key=lambda x: x[0])
        rows.append(
            {
                "variant": tag,
                "frame": frame,
                "nonhand_support_sdf_min_m": float(nonhand_sdf),
                "nonhand_support_sdf_arg": nonhand_arg,
                "hand_support_sdf_min_m": float(hand_sdf),
                "hand_support_sdf_arg": hand_arg,
            }
        )

    ts_path = results_dir / f"support_decomp_timeseries_{tag}.csv"
    with ts_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    nonhand_sdf = np.asarray(
        [float(row["nonhand_support_sdf_min_m"]) for row in rows], dtype=np.float64
    )
    hand_sdf = np.asarray(
        [float(row["hand_support_sdf_min_m"]) for row in rows], dtype=np.float64
    )
    argmin = int(np.argmin(nonhand_sdf))
    return {
        "support_decomp_timeseries_csv": rel(ts_path),
        "nonhand_support_geom_count": len(nonhand_gids),
        "nonhand_object_support_frac": float((nonhand_sdf < 0.02).mean()),
        "nonhand_object_interference_frac": float((nonhand_sdf < 0.0).mean()),
        "nonhand_support_sdf_min_m": float(nonhand_sdf.min()),
        "nonhand_support_sdf_mean_m": float(nonhand_sdf.mean()),
        "nonhand_support_sdf_argmin": str(rows[argmin]["nonhand_support_sdf_arg"]),
        "hand_support_geom_count": len(hand_gids),
        "hand_support_near_zero_frac": float((np.abs(hand_sdf) <= 0.01).mean()),
        "hand_support_near_2cm_frac": float((np.abs(hand_sdf) <= 0.02).mean()),
        "hand_support_sdf_min_m": float(hand_sdf.min()),
        "hand_support_sdf_mean_m": float(hand_sdf.mean()),
    }


def augment_method_rows(
    method_rows: list[dict[str, Any]],
    results_dir: Path,
) -> None:
    e105 = load_e105_eval()
    for row in method_rows:
        npz = repo_path(str(row["npz_path"]))
        scene = repo_path(str(row["scene_xml"]))
        if not npz.is_file() or not scene.is_file():
            continue
        tag = f"{row['variant']}_{row['method']}"
        row.update(support_metrics(e105, tag, npz, scene, results_dir))
        row.update(trajectory_info_metrics(npz))


def augment_decisions(
    decisions: list[dict[str, Any]],
    method_rows: list[dict[str, Any]],
) -> None:
    by_variant_method = {
        (row["variant"], row["method"]): row
        for row in method_rows
        if "nonhand_object_support_frac" in row
    }
    for row in decisions:
        variant = row["variant"]
        ablation = row["ablation"]
        baseline = by_variant_method.get((variant, "baseline_ref_fk_cache"), {})
        e113 = by_variant_method.get((variant, "e113_hold_band_ref"), {})
        test = by_variant_method.get((variant, ablation), {})
        row["baseline_nonhand_object_support_frac"] = baseline.get(
            "nonhand_object_support_frac", ""
        )
        row["e113_nonhand_object_support_frac"] = e113.get(
            "nonhand_object_support_frac", ""
        )
        row["test_nonhand_object_support_frac"] = test.get(
            "nonhand_object_support_frac", ""
        )
        row["test_nonhand_object_interference_frac"] = test.get(
            "nonhand_object_interference_frac", ""
        )
        row["test_nonhand_support_sdf_argmin"] = test.get(
            "nonhand_support_sdf_argmin", ""
        )
        row["test_hand_support_near_zero_frac"] = test.get(
            "hand_support_near_zero_frac", ""
        )
        row["test_hand_support_near_2cm_frac"] = test.get(
            "hand_support_near_2cm_frac", ""
        )
        row["test_terminal_gate_violation_mean"] = test.get(
            "terminal_carry_gate_violation_mean_mean", ""
        )
        row["test_terminal_gate_valid_mean"] = test.get(
            "terminal_carry_gate_valid_mean_mean", ""
        )
        row["test_terminal_gate_pelvis_z_mean"] = test.get(
            "terminal_carry_gate_pelvis_z_mean_mean", ""
        )
        row["test_terminal_gate_obj_rot_err_mean"] = test.get(
            "terminal_carry_gate_obj_rot_err_mean_mean", ""
        )
        row["test_cem_gate_valid_frac_mean"] = test.get(
            "cem_gate_valid_frac_mean", ""
        )
        row["test_cem_gate_fallback_used_mean"] = test.get(
            "cem_gate_fallback_used_mean", ""
        )
        support_ok = (
            test
            and float(test["nonhand_object_support_frac"]) <= 0.05
            and float(test["hand_support_near_zero_frac"]) >= 0.20
        )
        hard_variant = "hard" in ablation
        terminal_gate_pass = True
        if hard_variant:
            valid_frac = float(row["test_cem_gate_valid_frac_mean"] or 0.0)
            fallback = float(row["test_cem_gate_fallback_used_mean"] or 1.0)
            terminal_gate_pass = valid_frac >= 0.02 and fallback < 0.95
        row["terminal_gate_pass"] = bool(terminal_gate_pass)
        row["support_decomp_pass"] = bool(support_ok)
        strict_work = row["test_strict_status"] == "WORK"
        release = (
            row["phase_scope"] not in {"strict_guard", "box004_contact_guard"}
            and strict_work
            and bool(row["contact_ok"])
            and bool(row["penetration_ok"])
            and bool(row["pelvis_ok"])
            and bool(row["lowerbody_ok"])
            and bool(row["object_ok"])
            and bool(row["support_decomp_pass"])
            and bool(row["terminal_gate_pass"])
        )
        guard_pass = (
            row["phase_scope"] in {"strict_guard", "box004_contact_guard"}
            and strict_work
            and bool(row["penetration_ok"])
            and bool(row["lowerbody_ok"])
            and bool(row["object_ok"])
            and bool(row["support_decomp_pass"])
            and bool(row["terminal_gate_pass"])
        )
        row["phase_b_release_candidate"] = release
        row["guard_pass"] = guard_pass
        if release:
            row["pareto_decision"] = "release_candidate"
        elif row["phase_scope"] in {"strict_guard", "box004_contact_guard"} and guard_pass:
            row["pareto_decision"] = "guard_pass"
        elif not row["terminal_gate_pass"]:
            row["pareto_decision"] = "terminal_gate_starved"
        elif not row["support_decomp_pass"]:
            row["pareto_decision"] = "support_decomp_fail"


def rewrite_summary(
    stage: str,
    decisions: list[dict[str, Any]],
    method_rows: list[dict[str, Any]],
    missing: list[dict[str, str]],
    results_dir: Path,
) -> None:
    lines = [
        f"# E122 Snap Warmstart Carry Prior {stage} Summary",
        "",
        f"- evaluated variants: `{len(decisions)}`",
        f"- method rows: `{len(method_rows)}`",
        f"- missing variants: `{len(missing)}`",
        f"- variants file: `{rel(E122_VARIANTS)}`",
        "",
        "| case | ablation | phase | physics base->test | leg E113->test | non-hand support | hand near-zero | gate valid | fallback | strict | decision |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in sorted(decisions, key=lambda item: (item["case_id"], item["ablation"])):
        nonhand = row["test_nonhand_object_support_frac"]
        hand = row["test_hand_support_near_zero_frac"]
        gate_valid = row.get("test_cem_gate_valid_frac_mean", "")
        fallback = row.get("test_cem_gate_fallback_used_mean", "")
        nonhand_txt = "" if nonhand == "" else f"{float(nonhand) * 100:.1f}%"
        hand_txt = "" if hand == "" else f"{float(hand) * 100:.1f}%"
        gate_txt = "" if gate_valid == "" else f"{float(gate_valid) * 100:.1f}%"
        fallback_txt = "" if fallback == "" else f"{float(fallback) * 100:.1f}%"
        lines.append(
            f"| `{row['case_id']}` | `{row['ablation']}` | `{row['phase_scope']}` | "
            f"{float(row['baseline_physics_contact']) * 100:.1f}% -> {float(row['test_physics_contact']) * 100:.1f}% | "
            f"{float(row['e113_leg_interference_frac']) * 100:.1f}% -> {float(row['test_leg_interference_frac']) * 100:.1f}% | "
            f"{nonhand_txt} | {hand_txt} | {gate_txt} | {fallback_txt} | `{row['test_strict_status']}` | `{row['pareto_decision']}` |"
        )
    if missing:
        lines.extend(["", "## Missing Outputs", ""])
        for row in missing:
            lines.append(f"- `{row['variant']}`: {row['missing']}")
    (results_dir / f"{stage}_eval_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="full", choices=["smoke", "full"])
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=None)
    args = parser.parse_args()

    e115 = load_e115_eval()
    results_dir = repo_path_from_env(
        "RESULTS",
        args.results_dir or REPO / f"workspace/core4d/results/E122/cem/{args.stage}",
    )
    rows = e115.read_variants()
    missing = e115.fail_if_missing(args.stage, rows, results_dir, args.allow_missing)
    method_rows, decisions = e115.evaluate(args.stage, rows, results_dir)
    augment_method_rows(method_rows, results_dir)
    augment_decisions(decisions, method_rows)
    e115.write_outputs(args.stage, method_rows, decisions, missing, results_dir)
    rewrite_summary(args.stage, decisions, method_rows, missing, results_dir)
    print(f"wrote {rel(results_dir / f'{args.stage}_eval_summary.md')}")


if __name__ == "__main__":
    main()
