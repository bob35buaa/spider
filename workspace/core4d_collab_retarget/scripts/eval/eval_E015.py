#!/usr/bin/env python3
"""E015 evaluation wrapper for dynamic support + PD command."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
SCRIPT_EVAL = REPO / "workspace/core4d_collab_retarget/scripts/eval"
if str(SCRIPT_EVAL) not in sys.path:
    sys.path.insert(0, str(SCRIPT_EVAL))

import eval_E002 as e002  # noqa: E402


BASE = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
RESULTS = REPO / "workspace/core4d_collab_retarget/results/E015"
VARIANTS_FILE = REPO / "workspace/core4d_collab_retarget/scripts/E015/variants.tsv"
SOFT_TARGETS = REPO / "workspace/core4d_collab_retarget/results/E013/e014_soft_targets.json"

FIELDNAMES = [
    "variant",
    "source_task",
    "mask_slug",
    "person_idx",
    "queue",
    "role",
    "scene_name",
    "data_relpath",
    "support_proxy_point_local_x",
    "support_proxy_point_local_y",
    "support_proxy_point_local_z",
    "support_dynamic_mass",
    "support_dynamic_pos_kp",
    "support_dynamic_pos_kd",
    "support_dynamic_rot_kp",
    "support_dynamic_rot_kd",
    "support_dynamic_force_clamp",
    "support_dynamic_torque_clamp",
    "weld_solref_timeconst",
    "weld_solimp_1",
    "weld_solimp_2",
    "weld_solimp_width",
    "hold_contact_rew_scale",
    "hold_contact_sigma",
    "hold_contact_start_eval_time",
    "hold_contact_end_eval_time",
    "hold_contact_require_ref_contact",
]

_CURRENT_META: dict[str, str] | None = None


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


def _load_dynamic_scene_model(case: str) -> tuple[mujoco.MjModel, Path]:
    if _CURRENT_META is None:
        raise RuntimeError("E015 eval current variant metadata is not set")
    _qpos_ref, _ctrl_ref, cfg = e002.load_ref(
        str(_CURRENT_META["override"]), str(case)
    )
    path = Path(cfg.model_path)
    return mujoco.MjModel.from_xml_path(str(path)), path


def _support_dynamic_scene_ok(model: mujoco.MjModel) -> dict[str, object]:
    support_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "support_dynamic_anchor"
    )
    object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    equality_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "e015_support_weld")
    out: dict[str, object] = {
        "E015_support_body_found": bool(support_id >= 0),
        "E015_object_body_found": bool(object_id >= 0),
        "E015_weld_found": bool(equality_id >= 0),
        "E015_support_is_mocap": None,
        "E015_support_joint_count": None,
        "E015_support_qadr": None,
        "E015_support_dadr": None,
        "E015_object_qadr": None,
        "E015_object_dadr": None,
        "E015_support_joint_types": [],
    }
    if support_id >= 0:
        out["E015_support_is_mocap"] = bool(model.body_mocapid[support_id] >= 0)
        jadr = int(model.body_jntadr[support_id])
        jnum = int(model.body_jntnum[support_id])
        out["E015_support_joint_count"] = jnum
        if jadr >= 0 and jnum > 0:
            out["E015_support_qadr"] = int(model.jnt_qposadr[jadr])
            out["E015_support_dadr"] = int(model.jnt_dofadr[jadr])
            out["E015_support_joint_types"] = [
                int(model.jnt_type[jadr + i]) for i in range(jnum)
            ]
    if object_id >= 0:
        jadr = int(model.body_jntadr[object_id])
        if jadr >= 0:
            out["E015_object_qadr"] = int(model.jnt_qposadr[jadr])
            out["E015_object_dadr"] = int(model.jnt_dofadr[jadr])
    out["E015_object_last_qpos"] = bool(out["E015_object_qadr"] == model.nq - 7)
    out["E015_object_last_qvel"] = bool(out["E015_object_dadr"] == model.nv - 6)
    out["E015_support_dynamic_scene_ok"] = bool(
        model.nq == 49
        and model.nv == 47
        and model.nu == 29
        and support_id >= 0
        and object_id >= 0
        and equality_id >= 0
        and out["E015_support_is_mocap"] is False
        and out["E015_support_joint_count"] == 6
        and out["E015_support_qadr"] == 36
        and out["E015_support_dadr"] == 35
        and out["E015_object_last_qpos"]
        and out["E015_object_last_qvel"]
        and out["E015_support_joint_types"] == [2, 2, 2, 3, 3, 3]
    )
    return out


def _scene_not_oracle(meta: dict[str, str]) -> bool:
    scene_path = BASE / meta["source_task"] / f"{meta['scene_name']}.xml"
    if not scene_path.is_file():
        return False
    text = scene_path.read_text(encoding="utf-8")
    return bool(
        "support_dynamic_anchor" in text
        and "e015_support_weld" in text
        and "mocap=\"true\"" not in text
        and "object_target" not in text
        and "support_weld_anchor" not in text
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

    summary["E015_point_local"] = point_local.tolist()
    summary["E015_pd_metrics_present"] = bool(
        force is not None
        and torque is not None
        and proxy_pos is not None
        and support_point is not None
    )

    if qpos is not None and qpos.shape[1] >= 49 and qpos_ref.shape[1] >= 49:
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
        support_qpos = qpos[:, 36:42].astype(np.float64)
        ref_support_qpos = qpos_ref[:, 36:42].astype(np.float64)

        summary["E015_object_xy_disp_m"] = _xy_disp(obj_pos)
        summary["E015_ref_object_xy_disp_m"] = _xy_disp(ref_obj_pos)
        summary["E015_ref_support_xy_disp_m"] = _xy_disp(ref_support)
        summary["E015_object_xy_disp_ratio"] = float(
            summary["E015_object_xy_disp_m"]
            / max(summary["E015_ref_object_xy_disp_m"], 1e-8)
        )
        summary["E015_object_xy_disp_ratio_vs_ref_support"] = float(
            summary["E015_object_xy_disp_m"]
            / max(summary["E015_ref_support_xy_disp_m"], 1e-8)
        )
        summary["E015_object_rot_deg"] = _quat_angle_deg(obj_quat[0], obj_quat[-1])
        summary["E015_ref_object_rot_deg"] = _quat_angle_deg(
            ref_obj_quat[0], ref_obj_quat[-1]
        )

        start = min(int(summary["case_window_start_frame"]), T - 1)
        end = min(int(summary["case_window_end_frame"]) + 1, T)
        cw_obj_xy = _xy_disp(obj_pos[start:end])
        cw_ref_xy = _xy_disp(ref_obj_pos[start:end])
        summary["E015_case_window_object_xy_disp_m"] = cw_obj_xy
        summary["E015_case_window_ref_object_xy_disp_m"] = cw_ref_xy
        summary["E015_case_window_object_xy_disp_ratio"] = float(
            cw_obj_xy / max(cw_ref_xy, 1e-8)
        )
        target_gap = np.linalg.norm(support_qpos[:, :3] - ref_support_qpos[:, :3], axis=1)
        cw = _window(target_gap, summary)
        summary["E015_case_window_support_target_gap_mean_m"] = float(cw.mean())
        summary["E015_case_window_support_target_gap_max_m"] = float(cw.max())
        euler_gap = np.abs(support_qpos[:, 3:6] - ref_support_qpos[:, 3:6])
        euler_gap = (euler_gap + np.pi) % (2.0 * np.pi) - np.pi
        cw_euler = _window(np.linalg.norm(euler_gap, axis=1), summary)
        summary["E015_case_window_support_target_rot_gap_mean_rad"] = float(
            cw_euler.mean()
        )
        summary["E015_case_window_support_target_rot_gap_max_rad"] = float(
            cw_euler.max()
        )

    if not summary["E015_pd_metrics_present"]:
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
        summary[f"E015_case_window_{name}_mean"] = float(cw.mean())
        summary[f"E015_case_window_{name}_max"] = float(cw.max())

    summary["E015_effort_reasonable"] = bool(
        summary["E015_case_window_support_force_norm_n_mean"] <= 100.0
        and summary["E015_case_window_support_force_norm_n_max"] <= 250.0
    )


def evaluate_variant(
    variant: str,
    variants: dict[str, dict[str, str]],
    targets: dict[str, dict[str, float]],
) -> dict[str, object]:
    if variant not in variants:
        raise ValueError(f"Unknown E015 variant: {variant}")

    e002.RESULTS = RESULTS
    e002.VARIANTS_FILE = VARIANTS_FILE
    e002.load_scene_model = _load_dynamic_scene_model

    global _CURRENT_META
    meta = variants[variant]
    _CURRENT_META = meta
    summary = e002.evaluate_variant(variant, variants)
    _CURRENT_META = None

    npz_path = RESULTS / f"{variant}.npz"
    qpos_ref, _ctrl_ref, cfg = e002.load_ref(str(meta["override"]), str(meta["case"]))
    model = mujoco.MjModel.from_xml_path(str(cfg.model_path))
    _add_motion_and_support_metrics(summary, meta, npz_path, qpos_ref)

    summary["E015_queue"] = meta["queue"]
    summary["E015_scene_name"] = meta["scene_name"]
    summary["E015_data_relpath"] = meta["data_relpath"]
    summary["E015_weld_solref_timeconst"] = float(meta["weld_solref_timeconst"])
    summary["E015_support_dynamic_mass"] = float(cfg.support_dynamic_mass)
    summary["E015_support_dynamic_pos_kp"] = float(cfg.support_dynamic_pos_kp)
    summary["E015_support_dynamic_pos_kd"] = float(cfg.support_dynamic_pos_kd)
    summary["E015_support_dynamic_rot_kp"] = float(cfg.support_dynamic_rot_kp)
    summary["E015_support_dynamic_rot_kd"] = float(cfg.support_dynamic_rot_kd)
    summary["E015_support_dynamic_force_clamp"] = float(
        cfg.support_dynamic_force_clamp
    )
    summary["E015_support_dynamic_torque_clamp"] = float(
        cfg.support_dynamic_torque_clamp
    )
    summary["E015_support_proxy_enabled"] = bool(cfg.support_proxy_enabled)
    summary["E015_support_proxy_mode"] = str(cfg.support_proxy_mode)
    summary["E015_support_proxy_ref_dt"] = float(cfg.support_proxy_ref_dt)
    summary["E015_hold_contact_rew_scale"] = float(meta["hold_contact_rew_scale"])
    summary.update(_support_dynamic_scene_ok(model))

    summary["E015_freejoint_parity_ok"] = bool(
        not summary["config_contact_guidance"]
        and str(summary["config_scene_name"]).startswith("scene_e015_dyn_")
        and summary["config_nq"] == 49
        and summary["config_nv"] == 47
        and summary["config_nu"] == 29
        and summary["config_nq_obj"] == 7
        and summary["config_object_action_dims"] == 0
        and len(summary["config_object_actuator_ids"]) == 0
        and bool(cfg.support_proxy_enabled)
        and str(cfg.support_proxy_mode) == "dynamic_weld"
        and not bool(cfg.object_kinematic_override)
        and not bool(cfg.object_pd_override)
        and float(cfg.partner_force_scale) == 0.0
        and float(cfg.partner_force_spring_kp) == 0.0
        and summary["E015_support_dynamic_scene_ok"]
    )
    summary["E015_anchor_not_com_oracle"] = _scene_not_oracle(meta)
    summary["E015_no_direct_wrench"] = bool(
        str(cfg.support_proxy_mode) == "dynamic_weld"
        and float(cfg.partner_force_scale) == 0.0
        and float(cfg.partner_force_spring_kp) == 0.0
        and not bool(cfg.object_kinematic_override)
    )

    target = targets["guard" if summary["role"] == "guard" else "main"]
    summary["E015_target_obj_mean_m"] = float(target["obj_mean_target_m"])
    summary["E015_target_obj_max_m"] = float(target["obj_max_target_m"])
    summary["E015_target_hand_contact_min_pct"] = float(
        target["hand_contact_min_pct"]
    )
    summary["E015_target_floor_contact_max_pct"] = float(
        target["floor_contact_max_pct"]
    )
    summary["E015_target_leg_interference_max_pct"] = float(
        target["leg_interference_max_pct"]
    )
    summary["E015_target_lag_free_obj_mean_m"] = float(
        target["lag_free_obj_mean_m"]
    )
    summary["E015_target_push_floor_max_pct"] = float(
        target["push_vs_carry_floor_contact_max_pct"]
    )
    summary["E015_target_push_leg_max_pct"] = float(
        target["push_vs_carry_leg_interference_max_pct"]
    )

    obj_ok = bool(
        summary["case_window_obj_err_mean_m"] <= summary["E015_target_obj_mean_m"]
        and summary["case_window_obj_err_max_m"] <= summary["E015_target_obj_max_m"]
    )
    hand_ok = bool(
        summary["case_window_sim_contact_frames_pct"]
        >= summary["E015_target_hand_contact_min_pct"]
    )
    floor_ok = bool(
        summary["case_window_sim_object_floor_contact_frames_pct"]
        <= summary["E015_target_floor_contact_max_pct"]
    )
    leg_ok = bool(
        summary["case_window_sim_leg_box_interference_frames_pct"]
        <= summary["E015_target_leg_interference_max_pct"]
    )
    push_vs_carry_ok = bool(
        summary["case_window_sim_object_floor_contact_frames_pct"]
        <= summary["E015_target_push_floor_max_pct"]
        and summary["case_window_sim_leg_box_interference_frames_pct"]
        <= summary["E015_target_push_leg_max_pct"]
    )
    lag_free = bool(
        summary["case_window_obj_err_mean_m"]
        < summary["E015_target_lag_free_obj_mean_m"]
    )
    effort_ok = bool(summary.get("E015_effort_reasonable", False))

    summary["E015_soft_obj_ok"] = obj_ok
    summary["E015_soft_hand_ok"] = hand_ok
    summary["E015_soft_floor_ok"] = floor_ok
    summary["E015_soft_leg_ok"] = leg_ok
    summary["E015_push_vs_carry_ok"] = push_vs_carry_ok
    summary["E015_lag_free"] = lag_free
    summary["E015_soft_target_pass"] = bool(
        obj_ok
        and hand_ok
        and floor_ok
        and leg_ok
        and push_vs_carry_ok
        and summary["E015_freejoint_parity_ok"]
        and summary["E015_anchor_not_com_oracle"]
        and summary["E015_no_direct_wrench"]
    )
    summary["E015_full_success"] = bool(summary["E015_soft_target_pass"] and effort_ok)
    summary["E015_guard_stable"] = bool(
        summary["role"] == "guard"
        and summary["post2_pelvis_z_min_m"] >= 0.55
        and leg_ok
        and floor_ok
        and summary["E015_freejoint_parity_ok"]
        and summary["E015_anchor_not_com_oracle"]
    )

    if summary["E015_full_success"]:
        diagnostic = "soft_target_effort_ok"
    elif summary["E015_soft_target_pass"] and not effort_ok:
        diagnostic = "target_pass_effort_overdrive"
    elif summary["role"] == "guard" and not summary["E015_guard_stable"]:
        diagnostic = "guard_unstable"
    elif not lag_free:
        diagnostic = "dynamic_support_lag"
    elif summary.get("E015_object_rot_deg", 0.0) > 30.0:
        diagnostic = "rotation_shortcut"
    elif not hand_ok:
        diagnostic = "robot_side_contact_gap"
    elif not push_vs_carry_ok or not floor_ok or not leg_ok:
        diagnostic = "push_vs_carry_failed"
    else:
        diagnostic = "near_soft_target"
    summary["E015_diagnostic_class"] = diagnostic

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
        raise SystemExit("No E015 variant results found.")

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
            bool(r["E015_freejoint_parity_ok"]) for r in summaries
        ),
        "num_support_dynamic_scene_ok": sum(
            bool(r["E015_support_dynamic_scene_ok"]) for r in summaries
        ),
        "num_object_last_ok": sum(
            bool(r["E015_object_last_qpos"]) and bool(r["E015_object_last_qvel"])
            for r in summaries
        ),
        "num_no_direct_wrench": sum(
            bool(r["E015_no_direct_wrench"]) for r in summaries
        ),
        "num_pd_metrics_present": sum(
            bool(r["E015_pd_metrics_present"]) for r in summaries
        ),
        "num_main_soft_target_pass": sum(
            bool(r["E015_soft_target_pass"]) for r in main_rows
        ),
        "num_main_effort_reasonable": sum(
            bool(r.get("E015_effort_reasonable", False)) for r in main_rows
        ),
        "num_main_full_success": sum(bool(r["E015_full_success"]) for r in main_rows),
        "num_main_lag_free": sum(bool(r["E015_lag_free"]) for r in main_rows),
        "num_main_push_vs_carry_ok": sum(
            bool(r["E015_push_vs_carry_ok"]) for r in main_rows
        ),
        "num_guard_stable": sum(bool(r["E015_guard_stable"]) for r in guard_rows),
        "diagnostic_classes": {
            name: sum(r.get("E015_diagnostic_class") == name for r in summaries)
            for name in sorted({str(r.get("E015_diagnostic_class")) for r in summaries})
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
