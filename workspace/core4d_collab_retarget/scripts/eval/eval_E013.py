#!/usr/bin/env python3
"""E013 evaluation wrapper for true-freejoint object oracle."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
SCRIPT_EVAL = REPO / "workspace/core4d_collab_retarget/scripts/eval"
if str(SCRIPT_EVAL) not in sys.path:
    sys.path.insert(0, str(SCRIPT_EVAL))

import eval_E002 as e002  # noqa: E402
import eval_E011 as e011  # noqa: E402


RESULTS = REPO / "workspace/core4d_collab_retarget/results/E013"
VARIANTS_FILE = REPO / "workspace/core4d_collab_retarget/scripts/E013/variants.tsv"

FIELDNAMES = [
    "variant",
    "source_task",
    "mask_slug",
    "person_idx",
    "queue",
    "role",
    "wave",
]

E081_BASELINES = e011.E081_BASELINES


def read_variants() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    with VARIANTS_FILE.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(
            (line for line in f if line.strip() and not line.startswith("#")),
            delimiter="\t",
            fieldnames=FIELDNAMES,
        )
        for row in reader:
            out[row["variant"]] = {
                "name": row["variant"],
                "case": row["source_task"],
                "source_task": row["source_task"],
                "override": f"core4d_collab_{row['variant']}",
                "split": row["queue"],
                "role": row["role"],
                **row,
            }
    return out


def _flatten_npz(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray | None:
    if key not in data:
        return None
    return e002.e072.flatten_time_major(data[key])


def _xy_disp(pos: np.ndarray) -> float:
    if len(pos) < 2:
        return 0.0
    return float(np.linalg.norm(pos[-1, :2] - pos[0, :2]))


def _quat_angle_deg(q0: np.ndarray, q1: np.ndarray) -> float:
    q0 = q0.astype(np.float64)
    q1 = q1.astype(np.float64)
    q0 = q0 / np.clip(np.linalg.norm(q0), 1e-8, None)
    q1 = q1 / np.clip(np.linalg.norm(q1), 1e-8, None)
    dot = abs(float(np.dot(q0, q1)))
    dot = min(1.0, max(-1.0, dot))
    return float(np.degrees(2.0 * np.arccos(dot)))


def _add_oracle_motion_metrics(
    summary: dict[str, object], npz_path: Path, qpos_ref: np.ndarray
) -> None:
    data = np.load(npz_path, allow_pickle=True)
    qpos = _flatten_npz(data, "qpos")
    if qpos is None or qpos.shape[1] < 7 or qpos_ref.shape[1] < 7:
        summary["E013_oracle_motion_metrics_present"] = False
        return
    T = min(len(qpos), len(qpos_ref))
    qpos = qpos[:T]
    qpos_ref = qpos_ref[:T]
    obj_pos = qpos[:, -7:-4].astype(np.float64)
    obj_quat = qpos[:, -4:].astype(np.float64)
    ref_obj_pos = qpos_ref[:, -7:-4].astype(np.float64)
    ref_obj_quat = qpos_ref[:, -4:].astype(np.float64)

    start = min(int(summary["case_window_start_frame"]), T - 1)
    end = min(int(summary["case_window_end_frame"]) + 1, T)
    obj_cw = obj_pos[start:end]
    ref_cw = ref_obj_pos[start:end]

    summary["E013_oracle_motion_metrics_present"] = True
    summary["E013_object_xy_disp_m"] = _xy_disp(obj_pos)
    summary["E013_ref_object_xy_disp_m"] = _xy_disp(ref_obj_pos)
    summary["E013_object_xy_disp_ratio"] = float(
        summary["E013_object_xy_disp_m"] / max(summary["E013_ref_object_xy_disp_m"], 1e-8)
    )
    summary["E013_case_window_object_xy_disp_m"] = _xy_disp(obj_cw)
    summary["E013_case_window_ref_object_xy_disp_m"] = _xy_disp(ref_cw)
    summary["E013_case_window_object_xy_disp_ratio"] = float(
        summary["E013_case_window_object_xy_disp_m"]
        / max(summary["E013_case_window_ref_object_xy_disp_m"], 1e-8)
    )
    summary["E013_object_rot_deg"] = _quat_angle_deg(obj_quat[0], obj_quat[-1])
    summary["E013_ref_object_rot_deg"] = _quat_angle_deg(
        ref_obj_quat[0], ref_obj_quat[-1]
    )


def evaluate_variant(
    variant: str, variants: dict[str, dict[str, str]]
) -> dict[str, object]:
    if variant not in variants:
        raise ValueError(f"Unknown E013 variant: {variant}")

    e002.RESULTS = RESULTS
    e002.VARIANTS_FILE = VARIANTS_FILE
    summary = e002.evaluate_variant(variant, variants)
    meta = variants[variant]
    npz_path = RESULTS / f"{variant}.npz"
    qpos_ref, _ctrl_ref, cfg = e002.load_ref(str(meta["override"]), str(meta["case"]))
    _add_oracle_motion_metrics(summary, npz_path, qpos_ref)

    summary["E013_wave"] = meta["wave"]
    summary["E013_queue"] = meta["queue"]
    summary["E013_object_kinematic_override"] = bool(cfg.object_kinematic_override)
    summary["E013_object_kinematic_ref_dt"] = float(cfg.object_kinematic_ref_dt)
    summary["E013_object_kinematic_set_qvel"] = bool(cfg.object_kinematic_set_qvel)
    summary["E013_freejoint_oracle_config_ok"] = bool(
        not summary["config_contact_guidance"]
        and str(summary["config_scene_name"]) == "scene"
        and summary["config_nu"] == 29
        and summary["config_nq_obj"] == 7
        and summary["config_object_action_dims"] == 0
        and len(summary["config_object_actuator_ids"]) == 0
        and bool(cfg.object_kinematic_override)
    )

    baseline = E081_BASELINES["guard" if summary["role"] == "guard" else "main"]
    summary["E013_e081_baseline_variant"] = baseline["variant"]
    summary["E013_vs_E081_obj_mean_delta_m"] = float(
        summary["case_window_obj_err_mean_m"] - baseline["obj_mean"]
    )
    summary["E013_vs_E081_obj_max_delta_m"] = float(
        summary["case_window_obj_err_max_m"] - baseline["obj_max"]
    )
    summary["E013_vs_E081_hand_contact_delta_pp"] = float(
        summary["case_window_sim_contact_frames_pct"] - baseline["hand_pct"]
    )
    summary["E013_vs_E081_floor_contact_delta_pp"] = float(
        summary["case_window_sim_object_floor_contact_frames_pct"]
        - baseline["floor_pct"]
    )
    summary["E013_vs_E081_leg_intf_delta_pp"] = float(
        summary["case_window_sim_leg_box_interference_frames_pct"]
        - baseline["leg_intf_pct"]
    )
    summary["E013_oracle_for_soft_target"] = True
    summary["E013_near_e081_obj_oracle"] = bool(
        summary["case_window_obj_err_mean_m"] <= baseline["obj_mean"] + 0.05
        and summary["case_window_obj_err_max_m"] <= baseline["obj_max"] + 0.10
    )
    summary["E013_guard_stable"] = bool(
        summary["role"] == "guard"
        and summary["post2_pelvis_z_min_m"] >= 0.55
        and summary["case_window_sim_leg_box_interference_frames_pct"]
        <= E081_BASELINES["guard"]["leg_intf_pct"] + 5.0
        and summary["E013_freejoint_oracle_config_ok"]
    )

    e002.write_variant_summary(summary)
    return summary


def _soft_target_for(row: dict[str, object]) -> dict[str, float | str]:
    role = str(row["role"])
    baseline = E081_BASELINES["guard" if role == "guard" else "main"]
    return {
        "role": role,
        "variant": str(row["variant"]),
        "obj_mean_target_m": float(
            max(baseline["obj_mean"] + 0.05, row["case_window_obj_err_mean_m"] + 0.05)
        ),
        "obj_max_target_m": float(
            max(baseline["obj_max"] + 0.10, row["case_window_obj_err_max_m"] + 0.10)
        ),
        "hand_contact_min_pct": float(
            min(baseline["hand_pct"], row["case_window_sim_contact_frames_pct"]) - 5.0
        ),
        "floor_contact_max_pct": float(
            max(
                baseline["floor_pct"],
                row["case_window_sim_object_floor_contact_frames_pct"],
            )
            + 10.0
        ),
        "leg_interference_max_pct": float(
            max(
                baseline["leg_intf_pct"],
                row["case_window_sim_leg_box_interference_frames_pct"],
            )
            + 5.0
        ),
        "lag_free_obj_mean_m": 0.28,
        "push_vs_carry_leg_interference_max_pct": 15.0,
        "push_vs_carry_floor_contact_max_pct": 70.0,
    }


def normalize_args(args: list[str], variants: dict[str, dict[str, str]]) -> list[str]:
    if not args or args == ["--all"]:
        return list(variants.keys())
    selected: list[str] = []
    for arg in args:
        if arg == "--all":
            selected.extend(variants.keys())
        else:
            selected.append(arg)
    return selected


def main() -> None:
    variants = read_variants()
    selected = normalize_args(sys.argv[1:], variants)

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "plots").mkdir(parents=True, exist_ok=True)

    summaries = []
    for variant in selected:
        if variant not in variants:
            print(f"[SKIP] unknown variant {variant}")
            continue
        try:
            summaries.append(evaluate_variant(variant, variants))
        except FileNotFoundError as exc:
            print(f"[SKIP] {exc}")
            continue

    if not summaries:
        raise SystemExit("No E013 variant results found.")

    keys = sorted({k for row in summaries for k in row.keys()})
    comparison = RESULTS / "comparison.csv"
    with comparison.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summaries)

    main_rows = [r for r in summaries if r["role"] == "main"]
    guard_rows = [r for r in summaries if r["role"] == "guard"]
    aggregate = {
        "num_results": len(summaries),
        "num_main_results": len(main_rows),
        "num_guard_results": len(guard_rows),
        "num_freejoint_oracle_config_ok": sum(
            bool(r["E013_freejoint_oracle_config_ok"]) for r in summaries
        ),
        "num_near_e081_obj_oracle": sum(
            bool(r["E013_near_e081_obj_oracle"]) for r in summaries
        ),
        "num_guard_stable": sum(bool(r["E013_guard_stable"]) for r in guard_rows),
        "main_results": [r["variant"] for r in main_rows],
        "guard_results": [r["variant"] for r in guard_rows],
    }
    (RESULTS / "aggregate_summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8"
    )

    soft_targets = {
        "source": "E013 true-freejoint object oracle",
        "main": [_soft_target_for(r) for r in main_rows],
        "guard": [_soft_target_for(r) for r in guard_rows],
    }
    (RESULTS / "e014_soft_targets.json").write_text(
        json.dumps(soft_targets, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(f"Wrote {comparison}")
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
