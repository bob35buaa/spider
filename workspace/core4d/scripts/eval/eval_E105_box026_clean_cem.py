#!/usr/bin/env python3
"""Evaluate E105 Box026 clean-scene CEM rollouts and build comparison tables."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

THIS = Path(__file__).resolve()
REPO = THIS.parents[4]
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/E105"))
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/E098"))

from e105_common import FIELDS, RESULTS_ROOT, TASK_ROOT, rel, read_variants  # noqa: E402
import replay_gate as rg  # noqa: E402


E090_EVAL = REPO / "workspace/core4d/scripts/eval/eval_E090.py"

LEG_FOOT_GEOMS = [
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
HAND_GEOMS = ["lh", "rh"]
LOWERBODY_INTERFERENCE_STRICT_FRAC = 0.05


def repo_path_from_env(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    path = Path(raw) if raw else default
    return path if path.is_absolute() else REPO / path


def load_e090_eval():
    spec = importlib.util.spec_from_file_location("eval_E090_for_E105", E090_EVAL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {E090_EVAL}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def status(stage: str, metrics: dict[str, Any]) -> tuple[str, bool]:
    T = int(metrics["T"])
    obj_mean = float(metrics["obj_err_mean_m"])
    obj_max = float(metrics["obj_err_max_m"])
    pelvis = float(metrics["pelvis_min_m"])
    head = float(metrics["head_pen_frac"])
    upper = float(metrics["upper_pen_frac"])
    lh_floor = float(metrics["handL_floor_lt_5cm_frac"])
    rh_floor = float(metrics["handR_floor_lt_5cm_frac"])
    contact = float(metrics["contact_frac_either"])
    replay_gate_pass = bool(metrics.get("replay_gate_pass", True))
    if stage == "smoke":
        passed = (
            T >= 80
            and obj_mean <= 0.15
            and pelvis >= 0.45
            and max(head, upper, lh_floor, rh_floor) <= 0.10
            and replay_gate_pass
        )
        return ("PASS", True) if passed else ("FAIL", False)
    work = (
        T >= 80
        and obj_mean <= 0.10
        and obj_max <= 0.30
        and pelvis >= 0.55
        and max(head, upper, lh_floor, rh_floor) <= 0.05
        and contact >= 0.30
        and replay_gate_pass
    )
    if work:
        return "WORK", True
    review = (
        T >= 80
        and obj_mean <= 0.12
        and obj_max <= 0.35
        and pelvis >= 0.45
        and max(head, upper, lh_floor, rh_floor) <= 0.08
        and replay_gate_pass
    )
    return ("REVIEW+", False) if review else ("FAIL", False)


def read_old_manifest(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {row["new_variant"]: row for row in csv.DictReader(f, delimiter="\t")}


def read_tsv_by(path: Path, key: str) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {row[key]: row for row in csv.DictReader(f, delimiter="\t")}


def replay_metrics(npz_path: Path, scene_xml: Path) -> dict[str, Any]:
    qpos = rg.load_qpos(npz_path, channel="sim")
    gate = rg.evaluate(qpos, scene_xml)
    return {
        "replay_status": "ok",
        "replay_T": int(gate["T"]),
        "pelvis_end_z_m": float(gate["pelvis_end_z"]),
        "pelvis_tilt_max_deg": float(gate["pelvis_tilt_max_deg"]),
        "pelvis_tilt_end_deg": float(gate["pelvis_tilt_end_deg"]),
        "lie_on_box_frac": float(gate["lie_on_box_frac"]),
        "gate_pelvis_low": bool(gate["gate_pelvis_low"]),
        "gate_pelvis_tilt": bool(gate["gate_pelvis_tilt"]),
        "gate_lie_on_box": bool(gate["gate_lie_on_box"]),
        "replay_gate_fail": bool(gate["gate_overall_fail"]),
        "replay_gate_pass": not bool(gate["gate_overall_fail"]),
    }


def _geom_id(model: mujoco.MjModel, name: str) -> int:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if gid < 0:
        raise ValueError(f"geom {name!r} not found")
    return int(gid)


def _geom_name(model: mujoco.MjModel, gid: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom{gid}"


def _signed_point_box(point: np.ndarray, obj_pos: np.ndarray, obj_mat: np.ndarray, half: np.ndarray) -> float:
    local = obj_mat.T @ (point - obj_pos)
    q = np.abs(local) - half
    outside = np.linalg.norm(np.maximum(q, 0.0))
    inside = min(max(q[0], q[1], q[2]), 0.0)
    return float(outside + inside)


def _geom_box_adjusted_sdf(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_id: int,
    obj_pos: np.ndarray,
    obj_mat: np.ndarray,
    half: np.ndarray,
) -> float:
    """E081/E026 leg-object proxy: geom surface signed distance to object box."""
    geom_type = int(model.geom_type[geom_id])
    radius = float(model.geom_size[geom_id, 0])
    center = data.geom_xpos[geom_id].copy()
    mat = data.geom_xmat[geom_id].reshape(3, 3).copy()
    points = [center]
    if geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        half_len = float(model.geom_size[geom_id, 1])
        axis = mat[:, 2]
        points = [center + axis * s for s in np.linspace(-half_len, half_len, 9)]
    signed = min(_signed_point_box(p, obj_pos, obj_mat, half) for p in points)
    return float(signed - radius)


def leg_object_metrics(
    variant: str,
    npz_path: Path,
    scene_xml: Path,
    results_dir: Path,
) -> dict[str, Any]:
    """Add E026/E081-style lower-body/object interference metrics.

    This is a proxy, not a GT label. Negative leg_box_sdf means a lower-body
    collision geom overlaps the object collision box.
    """
    qpos = rg.load_qpos(npz_path, channel="sim")
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    object_geom = _geom_id(model, "object_collision")
    floor_geom = _geom_id(model, "floor")
    half = model.geom_size[object_geom, :3].copy()
    leg_gids = [_geom_id(model, name) for name in LEG_FOOT_GEOMS]
    hand_gids = [_geom_id(model, name) for name in HAND_GEOMS]

    rows: list[dict[str, Any]] = []
    for frame, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        obj_pos = data.geom_xpos[object_geom].copy()
        obj_mat = data.geom_xmat[object_geom].reshape(3, 3).copy()
        leg_vals = [
            (
                _geom_box_adjusted_sdf(model, data, gid, obj_pos, obj_mat, half),
                _geom_name(model, gid),
            )
            for gid in leg_gids
        ]
        hand_vals = [
            (
                _geom_box_adjusted_sdf(model, data, gid, obj_pos, obj_mat, half),
                _geom_name(model, gid),
            )
            for gid in hand_gids
        ]
        leg_sdf, leg_arg = min(leg_vals, key=lambda x: x[0])
        hand_sdf, hand_arg = min(hand_vals, key=lambda x: x[0])

        leg_contacts = 0
        hand_contacts = 0
        floor_contacts = 0
        for ci in range(data.ncon):
            con = data.contact[ci]
            pair = {int(con.geom1), int(con.geom2)}
            if object_geom in pair and floor_geom in pair:
                floor_contacts += 1
            if object_geom not in pair:
                continue
            other = next(iter(pair - {object_geom}))
            if other in leg_gids:
                leg_contacts += 1
            if other in hand_gids:
                hand_contacts += 1

        rows.append(
            {
                "variant": variant,
                "frame": frame,
                "leg_box_sdf_min_m": float(leg_sdf),
                "leg_box_sdf_arg": leg_arg,
                "hand_box_sdf_min_m": float(hand_sdf),
                "hand_box_sdf_arg": hand_arg,
                "leg_object_contact_count": int(leg_contacts),
                "hand_object_contact_count": int(hand_contacts),
                "object_floor_contact_count": int(floor_contacts),
                "object_bottom_proxy_m": float(obj_pos[2] - half[2]),
                "object_z_m": float(obj_pos[2]),
            }
        )

    ts_path = results_dir / f"legobj_timeseries_{variant}.csv"
    with ts_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    leg_sdf = np.asarray([float(row["leg_box_sdf_min_m"]) for row in rows], dtype=np.float64)
    hand_sdf = np.asarray([float(row["hand_box_sdf_min_m"]) for row in rows], dtype=np.float64)
    leg_contact = np.asarray([int(row["leg_object_contact_count"]) > 0 for row in rows], dtype=bool)
    hand_contact = np.asarray([int(row["hand_object_contact_count"]) > 0 for row in rows], dtype=bool)
    floor_contact = np.asarray([int(row["object_floor_contact_count"]) > 0 for row in rows], dtype=bool)
    argmin = int(np.argmin(leg_sdf))
    leg_intf_frac = float((leg_sdf < 0.0).mean())
    lowerbody_pass = leg_intf_frac <= LOWERBODY_INTERFERENCE_STRICT_FRAC
    return {
        "legobj_timeseries_csv": rel(ts_path),
        "legobj_geom_count": len(LEG_FOOT_GEOMS),
        "leg_box_sdf_min_m": float(leg_sdf.min()),
        "leg_box_sdf_mean_m": float(leg_sdf.mean()),
        "leg_box_sdf_argmin": str(rows[argmin]["leg_box_sdf_arg"]),
        "leg_box_interference_frac": leg_intf_frac,
        "leg_box_interference_pct": leg_intf_frac * 100.0,
        "leg_box_near_2cm_frac": float((leg_sdf < 0.02).mean()),
        "leg_object_contact_frac": float(leg_contact.mean()),
        "hand_box_sdf_min_m": float(hand_sdf.min()),
        "hand_box_sdf_mean_m": float(hand_sdf.mean()),
        "hand_object_contact_physics_frac": float(hand_contact.mean()),
        "object_floor_contact_frac": float(floor_contact.mean()),
        "lowerbody_strict_threshold_frac": LOWERBODY_INTERFERENCE_STRICT_FRAC,
        "lowerbody_strict_pass": lowerbody_pass,
    }


def write_summary(results: dict[str, dict[str, Any]], results_dir: Path, stage: str) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    out_json = results_dir / f"{stage}_eval_summary.json"
    out_json.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    keys = sorted({key for row in results.values() for key in row})
    out_csv = results_dir / f"{stage}_eval_summary.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results.values())
    out_md = results_dir / f"{stage}_eval_summary.md"
    lines = [
        f"# E105 Box026 clean-scene CEM {stage} summary",
        "",
        "| variant | route | T | contact | obj_mean | obj_max | pelvis | head | upper | LH floor | RH floor | status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for variant, row in results.items():
        lines.append(
            f"| `{variant}` | `{row['route']}` | {row['T']} | "
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
    lines.extend(
        [
            "",
            "Replay gate thresholds: pelvis_end_z>=0.55m, pelvis_tilt_end<=75deg, lie_on_box_frac<0.30.",
            "",
            "| variant | pelvis_end | tilt_end | lie_on_box | replay gate |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for variant, row in results.items():
        lines.append(
            f"| `{variant}` | {float(row['pelvis_end_z_m']):.3f}m | "
            f"{float(row['pelvis_tilt_end_deg']):.1f}deg | "
            f"{float(row['lie_on_box_frac']) * 100:.1f}% | "
            f"{'PASS' if row['replay_gate_pass'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "Lower-body strict proxy follows E026/E081: leg_box_interference_frac<=5%.",
            "",
            "| variant | leg interference | leg contact | min leg SDF | argmin geom | lower-body strict |",
            "|---|---:|---:|---:|---|---|",
        ]
    )
    for variant, row in results.items():
        lines.append(
            f"| `{variant}` | {float(row['leg_box_interference_frac']) * 100:.1f}% | "
            f"{float(row['leg_object_contact_frac']) * 100:.1f}% | "
            f"{float(row['leg_box_sdf_min_m']):.3f}m | "
            f"`{row['leg_box_sdf_argmin']}` | "
            f"{'PASS' if row['lowerbody_strict_pass'] else 'FAIL'} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {rel(out_json)}")
    print(f"wrote {rel(out_csv)}")
    print(f"wrote {rel(out_md)}")


def write_comparison(results: dict[str, dict[str, Any]]) -> None:
    old = read_old_manifest(RESULTS_ROOT / "box026_historical_full_cem_manifest.tsv")
    visual = read_tsv_by(RESULTS_ROOT / "pre_cem_visual_gate.tsv", "variant")
    preflight = read_tsv_by(RESULTS_ROOT / "clean_scene_preflight.tsv", "task")
    out_dir = RESULTS_ROOT / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_out = []
    for variant, row in results.items():
        old_row = old.get(variant, {})
        visual_row = visual.get(variant, {})
        target_clean_row = preflight.get(row["derived_task"], {})
        source_clean_row = preflight.get(row["source_task"], {})
        rows_out.append(
            {
                "variant": variant,
                "route": row["route"],
                "source_task": row["source_task"],
                "derived_task": row["derived_task"],
                "source_clean_gate": source_clean_row.get("gate", ""),
                "source_robot_polluted_mass_29_632": source_clean_row.get("robot_polluted_mass_29_632", ""),
                "target_clean_gate": target_clean_row.get("gate", ""),
                "target_robot_polluted_mass_29_632": target_clean_row.get("robot_polluted_mass_29_632", ""),
                "pre_cem_review_status": visual_row.get("review_status", ""),
                "target_npz": row.get("target_npz", ""),
                "target_route": row.get("target_route", ""),
                "old_variant": old_row.get("old_variant", ""),
                "old_experiment": old_row.get("old_experiment", ""),
                "old_status": old_row.get("old_status", ""),
                "new_status": row["work_status"],
                "old_validity": old_row.get("validity", ""),
                "old_contact_frac_either": old_row.get("old_contact_frac_either", ""),
                "new_contact_frac_either": row["contact_frac_either"],
                "old_obj_err_mean_m": old_row.get("old_obj_err_mean_m", ""),
                "new_obj_err_mean_m": row["obj_err_mean_m"],
                "old_obj_err_max_m": old_row.get("old_obj_err_max_m", ""),
                "new_obj_err_max_m": row["obj_err_max_m"],
                "old_pelvis_min_m": old_row.get("old_pelvis_min_m", ""),
                "new_pelvis_min_m": row["pelvis_min_m"],
                "old_head_pen_frac": old_row.get("old_head_pen_frac", ""),
                "new_head_pen_frac": row["head_pen_frac"],
                "old_upper_pen_frac": old_row.get("old_upper_pen_frac", ""),
                "new_upper_pen_frac": row["upper_pen_frac"],
                "old_handL_floor_lt_5cm_frac": old_row.get("old_handL_floor_lt_5cm_frac", ""),
                "new_handL_floor_lt_5cm_frac": row["handL_floor_lt_5cm_frac"],
                "old_handR_floor_lt_5cm_frac": old_row.get("old_handR_floor_lt_5cm_frac", ""),
                "new_handR_floor_lt_5cm_frac": row["handR_floor_lt_5cm_frac"],
                "new_pelvis_end_z_m": row["pelvis_end_z_m"],
                "new_pelvis_tilt_end_deg": row["pelvis_tilt_end_deg"],
                "new_lie_on_box_frac": row["lie_on_box_frac"],
                "new_replay_gate_pass": row["replay_gate_pass"],
                "new_leg_box_interference_frac": row["leg_box_interference_frac"],
                "new_leg_object_contact_frac": row["leg_object_contact_frac"],
                "new_leg_box_sdf_min_m": row["leg_box_sdf_min_m"],
                "new_leg_box_sdf_argmin": row["leg_box_sdf_argmin"],
                "new_lowerbody_strict_pass": row["lowerbody_strict_pass"],
                "new_lowerbody_strict_threshold_frac": row["lowerbody_strict_threshold_frac"],
                "old_scene_status": old_row.get("validity", "invalidated_by_e103_scene_inertial_bug") if old_row else "",
                "new_scene_status": "clean_e105_derived",
                "interpretation": "secondary_ablation_no_old_full_cem" if not old_row else "historical_old_vs_clean_rerun",
            }
        )
    fields = [
        "variant",
        "route",
        "source_task",
        "derived_task",
        "source_clean_gate",
        "source_robot_polluted_mass_29_632",
        "target_clean_gate",
        "target_robot_polluted_mass_29_632",
        "pre_cem_review_status",
        "target_route",
        "target_npz",
        "old_variant",
        "old_experiment",
        "old_status",
        "new_status",
        "old_validity",
        "old_contact_frac_either",
        "new_contact_frac_either",
        "old_obj_err_mean_m",
        "new_obj_err_mean_m",
        "old_obj_err_max_m",
        "new_obj_err_max_m",
        "old_pelvis_min_m",
        "new_pelvis_min_m",
        "old_head_pen_frac",
        "new_head_pen_frac",
        "old_upper_pen_frac",
        "new_upper_pen_frac",
        "old_handL_floor_lt_5cm_frac",
        "new_handL_floor_lt_5cm_frac",
        "old_handR_floor_lt_5cm_frac",
        "new_handR_floor_lt_5cm_frac",
        "new_pelvis_end_z_m",
        "new_pelvis_tilt_end_deg",
        "new_lie_on_box_frac",
        "new_replay_gate_pass",
        "new_leg_box_interference_frac",
        "new_leg_object_contact_frac",
        "new_leg_box_sdf_min_m",
        "new_leg_box_sdf_argmin",
        "new_lowerbody_strict_pass",
        "new_lowerbody_strict_threshold_frac",
        "old_scene_status",
        "new_scene_status",
        "interpretation",
    ]
    out_csv = out_dir / "box026_clean_vs_old_comparison.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_out)
    out_md = out_dir / "box026_clean_vs_old_comparison.md"
    lines = [
        "# E105 Box026 Clean-vs-Old Comparison",
        "",
        "| variant | route | old variant | old status | new status | old contact | new contact | old pelvis | new pelvis | interpretation |",
        "|---|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for item in rows_out:
        lines.append(
            f"| `{item['variant']}` | `{item['route']}` | `{item['old_variant']}` | "
            f"{item['old_status']} | {item['new_status']} | {item['old_contact_frac_either']} | "
            f"{float(item['new_contact_frac_either']):.3f} | {item['old_pelvis_min_m']} | "
            f"{float(item['new_pelvis_min_m']):.3f} | {item['interpretation']} |"
        )
    lines.extend(
        [
            "",
            "| variant | clean gate | safety new (head/upper/LH/RH) | replay gate (pelvis_end/tilt/lie) | old validity |",
            "|---|---|---|---|---|",
        ]
    )
    for item in rows_out:
        lines.append(
            f"| `{item['variant']}` | source={item['source_clean_gate']}, target={item['target_clean_gate']} | "
            f"{float(item['new_head_pen_frac']) * 100:.1f}%/"
            f"{float(item['new_upper_pen_frac']) * 100:.1f}%/"
            f"{float(item['new_handL_floor_lt_5cm_frac']) * 100:.1f}%/"
            f"{float(item['new_handR_floor_lt_5cm_frac']) * 100:.1f}% | "
            f"{'PASS' if item['new_replay_gate_pass'] else 'FAIL'} "
            f"({float(item['new_pelvis_end_z_m']):.3f}m/"
            f"{float(item['new_pelvis_tilt_end_deg']):.1f}deg/"
            f"{float(item['new_lie_on_box_frac']) * 100:.1f}%) | "
            f"{item['old_validity']} |"
        )
    lines.extend(
        [
            "",
            "| variant | lower-body strict | leg interference | leg contact | min leg SDF | argmin geom |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for item in rows_out:
        lines.append(
            f"| `{item['variant']}` | {'PASS' if item['new_lowerbody_strict_pass'] else 'FAIL'} | "
            f"{float(item['new_leg_box_interference_frac']) * 100:.1f}% | "
            f"{float(item['new_leg_object_contact_frac']) * 100:.1f}% | "
            f"{float(item['new_leg_box_sdf_min_m']):.3f}m | "
            f"`{item['new_leg_box_sdf_argmin']}` |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {rel(out_csv)}")
    print(f"wrote {rel(out_md)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["smoke", "full"], default="full")
    parser.add_argument("variants", nargs="*")
    args = parser.parse_args()

    eval_e090 = load_e090_eval()
    results_dir = repo_path_from_env("RESULTS", RESULTS_ROOT / f"cem/{args.stage}")
    variant_rows = {row["variant"]: row for row in read_variants()}
    selected = args.variants or list(variant_rows)
    results: dict[str, dict[str, Any]] = {}
    for variant in selected:
        row = variant_rows.get(variant)
        if row is None:
            print(f"[SKIP] unknown E105 variant {variant}")
            continue
        npz = results_dir / f"{variant}_outdir_{args.stage}/trajectory_mjwp_act.npz"
        scene = TASK_ROOT / row["derived_task"] / "scene_act.xml"
        if not npz.is_file():
            print(f"[SKIP] missing rollout {npz}")
            continue
        metrics = eval_e090.compute_sim_metrics(npz, scene)
        metrics.update(replay_metrics(npz, scene))
        metrics.update(leg_object_metrics(variant, npz, scene, results_dir))
        work_status, stage_pass = status(args.stage, metrics)
        lowerbody_pass = bool(metrics["lowerbody_strict_pass"])
        strict_work_status = "WORK" if work_status == "WORK" and lowerbody_pass else "FAIL"
        metrics.update({key: row.get(key, "") for key in FIELDS})
        metrics.update(
            {
                "stage": args.stage,
                "npz_path": rel(npz),
                "scene_xml": rel(scene),
                "work_status": work_status,
                "work_status_lowerbody_strict": strict_work_status,
                "stage_pass": stage_pass,
                "advance_to_rl": bool(args.stage == "full" and work_status == "WORK"),
                "advance_to_rl_lowerbody_strict": bool(args.stage == "full" and strict_work_status == "WORK"),
            }
        )
        results[variant] = metrics
    if not results:
        raise SystemExit("No E105 rollout results found.")
    write_summary(results, results_dir, args.stage)
    if args.stage == "full":
        write_comparison(results)


if __name__ == "__main__":
    main()
