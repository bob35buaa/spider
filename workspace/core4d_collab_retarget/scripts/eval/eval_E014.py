#!/usr/bin/env python3
"""E014 evaluation wrapper for COLA-B soft-weld support anchors."""

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


BASE = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
RESULTS = REPO / "workspace/core4d_collab_retarget/results/E014"
VARIANTS_FILE = REPO / "workspace/core4d_collab_retarget/scripts/E014/variants.tsv"
SOFT_TARGETS = REPO / "workspace/core4d_collab_retarget/results/E013/e014_soft_targets.json"

FIELDNAMES = [
    "variant",
    "source_task",
    "mask_slug",
    "person_idx",
    "queue",
    "role",
    "wave",
    "scene_name",
    "support_proxy_point_local_x",
    "support_proxy_point_local_y",
    "support_proxy_point_local_z",
    "weld_solref_timeconst",
    "weld_solimp_1",
    "weld_solimp_2",
    "weld_solimp_width",
    "support_proxy_gravity_scale",
    "hold_contact_rew_scale",
    "hold_contact_sigma",
    "hold_contact_start_eval_time",
    "hold_contact_end_eval_time",
    "hold_contact_require_ref_contact",
]


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


def _write_summary(summary: dict[str, object]) -> None:
    variant = str(summary["variant"])
    summary_json = RESULTS / f"eval_summary_{variant}.json"
    summary_csv = RESULTS / f"eval_summary_{variant}.csv"
    summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def _flatten_npz(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray | None:
    if key not in data:
        return None
    return e002.e072.flatten_time_major(data[key])


def _quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8, None)
    qvec = q[..., 1:]
    t = 2.0 * np.cross(qvec, v)
    return v + q[..., :1] * t + np.cross(qvec, t)


def _quat_angle_deg(q0: np.ndarray, q1: np.ndarray) -> float:
    q0 = q0.astype(np.float64)
    q1 = q1.astype(np.float64)
    q0 = q0 / np.clip(np.linalg.norm(q0), 1e-8, None)
    q1 = q1 / np.clip(np.linalg.norm(q1), 1e-8, None)
    dot = abs(float(np.dot(q0, q1)))
    dot = min(1.0, max(-1.0, dot))
    return float(np.degrees(2.0 * np.arccos(dot)))


def _xy_disp(pos: np.ndarray) -> float:
    if len(pos) < 2:
        return 0.0
    return float(np.linalg.norm(pos[-1, :2] - pos[0, :2]))


def _window(arr: np.ndarray, summary: dict[str, object]) -> np.ndarray:
    start = min(int(summary["case_window_start_frame"]), len(arr) - 1)
    end = min(int(summary["case_window_end_frame"]) + 1, len(arr))
    return arr[start:end]


def _point_local(meta: dict[str, str]) -> np.ndarray:
    return np.array(
        [
            float(meta["support_proxy_point_local_x"]),
            float(meta["support_proxy_point_local_y"]),
            float(meta["support_proxy_point_local_z"]),
        ],
        dtype=np.float64,
    )


def _load_soft_targets() -> dict[str, dict[str, float]]:
    data = json.loads(SOFT_TARGETS.read_text(encoding="utf-8"))
    out: dict[str, dict[str, float]] = {}
    for role in ("main", "guard"):
        rows = data.get(role, [])
        if rows:
            out[role] = rows[0]
    return out


def _scene_anchor_not_oracle(meta: dict[str, str]) -> bool:
    scene_path = BASE / meta["source_task"] / f"{meta['scene_name']}.xml"
    if not scene_path.is_file():
        return False
    text = scene_path.read_text(encoding="utf-8")
    point_norm = float(np.linalg.norm(_point_local(meta)))
    return bool(
        point_norm > 0.05
        and meta["scene_name"] != "scene_weld"
        and "object_target" not in text
        and "support_weld_anchor" in text
        and "e014_support_weld" in text
        and 'relpose="0 0 0 1 0 0 0"' not in text
    )


def _add_motion_and_support_metrics(
    summary: dict[str, object],
    meta: dict[str, str],
    npz_path: Path,
    qpos_ref: np.ndarray,
) -> None:
    data = np.load(npz_path, allow_pickle=True)
    qpos = _flatten_npz(data, "qpos")
    force = _flatten_npz(data, "support_proxy_force")
    torque = _flatten_npz(data, "support_proxy_torque")
    proxy_pos = _flatten_npz(data, "support_proxy_pos")
    support_point = _flatten_npz(data, "support_point_pos")
    point_local = _point_local(meta)

    summary["E014_point_local"] = point_local.tolist()
    summary["E014_support_proxy_metrics_present"] = bool(
        force is not None
        and torque is not None
        and proxy_pos is not None
        and support_point is not None
    )

    if qpos is not None and qpos.shape[1] >= 7 and qpos_ref.shape[1] >= 7:
        T = min(len(qpos), len(qpos_ref))
        qpos = qpos[:T]
        qpos_ref = qpos_ref[:T]
        obj_pos = qpos[:, -7:-4].astype(np.float64)
        obj_quat = qpos[:, -4:].astype(np.float64)
        ref_obj_pos = qpos_ref[:, -7:-4].astype(np.float64)
        ref_obj_quat = qpos_ref[:, -4:].astype(np.float64)
        ref_support = ref_obj_pos + _quat_apply(
            ref_obj_quat, np.broadcast_to(point_local, (T, 3))
        )

        summary["E014_object_xy_disp_m"] = _xy_disp(obj_pos)
        summary["E014_ref_object_xy_disp_m"] = _xy_disp(ref_obj_pos)
        summary["E014_ref_support_xy_disp_m"] = _xy_disp(ref_support)
        summary["E014_object_xy_disp_ratio"] = float(
            summary["E014_object_xy_disp_m"]
            / max(summary["E014_ref_object_xy_disp_m"], 1e-8)
        )
        summary["E014_object_xy_disp_ratio_vs_ref_support"] = float(
            summary["E014_object_xy_disp_m"]
            / max(summary["E014_ref_support_xy_disp_m"], 1e-8)
        )
        summary["E014_object_rot_deg"] = _quat_angle_deg(obj_quat[0], obj_quat[-1])
        summary["E014_ref_object_rot_deg"] = _quat_angle_deg(
            ref_obj_quat[0], ref_obj_quat[-1]
        )

        start = min(int(summary["case_window_start_frame"]), T - 1)
        end = min(int(summary["case_window_end_frame"]) + 1, T)
        cw_obj_xy = _xy_disp(obj_pos[start:end])
        cw_ref_xy = _xy_disp(ref_obj_pos[start:end])
        summary["E014_case_window_object_xy_disp_m"] = cw_obj_xy
        summary["E014_case_window_ref_object_xy_disp_m"] = cw_ref_xy
        summary["E014_case_window_object_xy_disp_ratio"] = float(
            cw_obj_xy / max(cw_ref_xy, 1e-8)
        )

    if not summary["E014_support_proxy_metrics_present"]:
        return

    force_norm = np.linalg.norm(force, axis=1)
    torque_norm = np.linalg.norm(torque, axis=1)
    gap = proxy_pos - support_point
    gap_norm = np.linalg.norm(gap, axis=1)
    gap_h = np.linalg.norm(gap[:, :2], axis=1)

    for name, arr in [
        ("support_force_norm_n", force_norm),
        ("support_torque_norm_nm", torque_norm),
        ("support_gap_norm_m", gap_norm),
        ("support_gap_horizontal_m", gap_h),
    ]:
        cw = _window(arr, summary)
        summary[f"E014_case_window_{name}_mean"] = float(cw.mean())
        summary[f"E014_case_window_{name}_max"] = float(cw.max())

    if proxy_pos is not None and qpos_ref.shape[1] >= 7:
        T = min(len(proxy_pos), len(qpos_ref))
        ref_obj_pos = qpos_ref[:T, -7:-4].astype(np.float64)
        ref_obj_quat = qpos_ref[:T, -4:].astype(np.float64)
        ref_support = ref_obj_pos + _quat_apply(
            ref_obj_quat, np.broadcast_to(point_local, (T, 3))
        )
        proxy_gap_to_ref = np.linalg.norm(proxy_pos[:T] - ref_support, axis=1)
        cw = _window(proxy_gap_to_ref, summary)
        summary["E014_case_window_proxy_gap_to_ref_support_mean_m"] = float(
            cw.mean()
        )
        summary["E014_case_window_proxy_gap_to_ref_support_max_m"] = float(cw.max())
        summary["E014_proxy_support_tracking_ok"] = bool(
            summary["E014_case_window_proxy_gap_to_ref_support_mean_m"] <= 0.03
            and summary["E014_case_window_proxy_gap_to_ref_support_max_m"] <= 0.08
        )


def evaluate_variant(
    variant: str,
    variants: dict[str, dict[str, str]],
    targets: dict[str, dict[str, float]],
) -> dict[str, object]:
    if variant not in variants:
        raise ValueError(f"Unknown E014 variant: {variant}")

    e002.RESULTS = RESULTS
    e002.VARIANTS_FILE = VARIANTS_FILE
    summary = e002.evaluate_variant(variant, variants)
    meta = variants[variant]
    npz_path = RESULTS / f"{variant}.npz"
    qpos_ref, _ctrl_ref, cfg = e002.load_ref(str(meta["override"]), str(meta["case"]))
    _add_motion_and_support_metrics(summary, meta, npz_path, qpos_ref)

    summary["E014_wave"] = meta["wave"]
    summary["E014_queue"] = meta["queue"]
    summary["E014_scene_name"] = meta["scene_name"]
    summary["E014_weld_solref_timeconst"] = float(meta["weld_solref_timeconst"])
    summary["E014_weld_solimp_1"] = float(meta["weld_solimp_1"])
    summary["E014_weld_solimp_2"] = float(meta["weld_solimp_2"])
    summary["E014_weld_solimp_width"] = float(meta["weld_solimp_width"])
    summary["E014_support_proxy_enabled"] = bool(cfg.support_proxy_enabled)
    summary["E014_support_proxy_mode"] = str(cfg.support_proxy_mode)
    summary["E014_support_proxy_mocap_body_name"] = str(
        cfg.support_proxy_mocap_body_name
    )
    summary["E014_support_proxy_mocap_quat_mode"] = str(
        cfg.support_proxy_mocap_quat_mode
    )
    summary["E014_support_proxy_gravity_scale"] = float(
        cfg.support_proxy_gravity_scale
    )
    summary["E014_hold_contact_rew_scale"] = float(meta["hold_contact_rew_scale"])

    summary["E014_freejoint_parity_ok"] = bool(
        not summary["config_contact_guidance"]
        and str(summary["config_scene_name"]).startswith("scene_e014_jointB_")
        and summary["config_nu"] == 29
        and summary["config_nq_obj"] == 7
        and summary["config_object_action_dims"] == 0
        and len(summary["config_object_actuator_ids"]) == 0
        and bool(cfg.support_proxy_enabled)
        and str(cfg.support_proxy_mode) == "mocap_pad"
        and str(cfg.support_proxy_mocap_body_name) == "support_weld_anchor"
        and str(cfg.support_proxy_mocap_quat_mode) == "object_ref"
        and not bool(cfg.object_kinematic_override)
        and not bool(cfg.object_pd_override)
        and float(cfg.partner_force_scale) == 0.0
        and float(cfg.partner_force_spring_kp) == 0.0
    )
    summary["E014_anchor_not_com_oracle"] = _scene_anchor_not_oracle(meta)
    summary["E014_no_direct_wrench"] = bool(
        str(cfg.support_proxy_mode) == "mocap_pad"
        and float(cfg.partner_force_scale) == 0.0
        and float(cfg.partner_force_spring_kp) == 0.0
    )

    target = targets["guard" if summary["role"] == "guard" else "main"]
    summary["E014_target_obj_mean_m"] = float(target["obj_mean_target_m"])
    summary["E014_target_obj_max_m"] = float(target["obj_max_target_m"])
    summary["E014_target_hand_contact_min_pct"] = float(
        target["hand_contact_min_pct"]
    )
    summary["E014_target_floor_contact_max_pct"] = float(
        target["floor_contact_max_pct"]
    )
    summary["E014_target_leg_interference_max_pct"] = float(
        target["leg_interference_max_pct"]
    )
    summary["E014_target_lag_free_obj_mean_m"] = float(
        target["lag_free_obj_mean_m"]
    )
    summary["E014_target_push_floor_max_pct"] = float(
        target["push_vs_carry_floor_contact_max_pct"]
    )
    summary["E014_target_push_leg_max_pct"] = float(
        target["push_vs_carry_leg_interference_max_pct"]
    )

    obj_ok = bool(
        summary["case_window_obj_err_mean_m"] <= summary["E014_target_obj_mean_m"]
        and summary["case_window_obj_err_max_m"] <= summary["E014_target_obj_max_m"]
    )
    hand_ok = bool(
        summary["case_window_sim_contact_frames_pct"]
        >= summary["E014_target_hand_contact_min_pct"]
    )
    floor_ok = bool(
        summary["case_window_sim_object_floor_contact_frames_pct"]
        <= summary["E014_target_floor_contact_max_pct"]
    )
    leg_ok = bool(
        summary["case_window_sim_leg_box_interference_frames_pct"]
        <= summary["E014_target_leg_interference_max_pct"]
    )
    push_vs_carry_ok = bool(
        summary["case_window_sim_object_floor_contact_frames_pct"]
        <= summary["E014_target_push_floor_max_pct"]
        and summary["case_window_sim_leg_box_interference_frames_pct"]
        <= summary["E014_target_push_leg_max_pct"]
    )
    lag_free = bool(
        summary["case_window_obj_err_mean_m"]
        < summary["E014_target_lag_free_obj_mean_m"]
    )
    summary["E014_soft_obj_ok"] = obj_ok
    summary["E014_soft_hand_ok"] = hand_ok
    summary["E014_soft_floor_ok"] = floor_ok
    summary["E014_soft_leg_ok"] = leg_ok
    summary["E014_push_vs_carry_ok"] = push_vs_carry_ok
    summary["E014_lag_free"] = lag_free
    summary["E014_support_gap_reasonable"] = bool(
        summary.get("E014_case_window_support_gap_norm_m_mean", 999.0) <= 0.12
        and summary.get("E014_case_window_support_gap_norm_m_max", 999.0) <= 0.35
    )

    summary["E014_soft_target_pass"] = bool(
        obj_ok
        and hand_ok
        and floor_ok
        and leg_ok
        and push_vs_carry_ok
        and summary["E014_freejoint_parity_ok"]
        and summary["E014_anchor_not_com_oracle"]
        and summary["E014_no_direct_wrench"]
    )
    summary["E014_guard_stable"] = bool(
        summary["role"] == "guard"
        and summary["post2_pelvis_z_min_m"] >= 0.55
        and leg_ok
        and floor_ok
        and summary["E014_freejoint_parity_ok"]
        and summary["E014_anchor_not_com_oracle"]
    )

    if summary["E014_soft_target_pass"]:
        diagnostic = "soft_target_pass"
    elif summary["role"] == "guard" and not summary["E014_guard_stable"]:
        diagnostic = "guard_unstable"
    elif not lag_free:
        diagnostic = "constraint_lag"
    elif summary.get("E014_object_rot_deg", 0.0) > 30.0:
        diagnostic = "rotation_shortcut"
    elif not hand_ok:
        diagnostic = "robot_side_contact_gap"
    elif not push_vs_carry_ok or not floor_ok or not leg_ok:
        diagnostic = "push_vs_carry_failed"
    else:
        diagnostic = "near_soft_target"
    summary["E014_diagnostic_class"] = diagnostic

    _write_summary(summary)
    return summary


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
    targets = _load_soft_targets()
    selected = normalize_args(sys.argv[1:], variants)

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "plots").mkdir(parents=True, exist_ok=True)

    summaries = []
    for variant in selected:
        if variant not in variants:
            print(f"[SKIP] unknown variant {variant}")
            continue
        try:
            summaries.append(evaluate_variant(variant, variants, targets))
        except FileNotFoundError as exc:
            print(f"[SKIP] {exc}")
            continue

    if not summaries:
        raise SystemExit("No E014 variant results found.")

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
        "num_freejoint_parity_ok": sum(
            bool(r["E014_freejoint_parity_ok"]) for r in summaries
        ),
        "num_anchor_not_com_oracle": sum(
            bool(r["E014_anchor_not_com_oracle"]) for r in summaries
        ),
        "num_no_direct_wrench": sum(
            bool(r["E014_no_direct_wrench"]) for r in summaries
        ),
        "num_support_proxy_metrics_present": sum(
            bool(r["E014_support_proxy_metrics_present"]) for r in summaries
        ),
        "num_main_soft_target_pass": sum(
            bool(r["E014_soft_target_pass"]) for r in main_rows
        ),
        "num_main_lag_free": sum(bool(r["E014_lag_free"]) for r in main_rows),
        "num_main_push_vs_carry_ok": sum(
            bool(r["E014_push_vs_carry_ok"]) for r in main_rows
        ),
        "num_guard_stable": sum(bool(r["E014_guard_stable"]) for r in guard_rows),
        "diagnostic_classes": {
            name: sum(r.get("E014_diagnostic_class") == name for r in summaries)
            for name in sorted({str(r.get("E014_diagnostic_class")) for r in summaries})
        },
        "main_results": [r["variant"] for r in main_rows],
        "guard_results": [r["variant"] for r in guard_rows],
    }
    (RESULTS / "aggregate_summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(f"Wrote {comparison}")
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
