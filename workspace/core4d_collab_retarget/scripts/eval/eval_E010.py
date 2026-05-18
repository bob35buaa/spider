#!/usr/bin/env python3
"""E010 evaluation wrapper for COLA-style support-body proxy sweep."""

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


RESULTS = REPO / "workspace/core4d_collab_retarget/results/E010"
VARIANTS_FILE = REPO / "workspace/core4d_collab_retarget/scripts/E010/variants.tsv"

FIELDNAMES = [
    "variant",
    "source_task",
    "mask_slug",
    "person_idx",
    "queue",
    "role",
    "wave",
    "point_local_x",
    "point_local_y",
    "point_local_z",
    "pad_size",
    "support_proxy_max_xy_speed",
    "support_proxy_ref_dt",
    "hold_contact_rew_scale",
    "hold_contact_sigma",
    "hold_contact_start_eval_time",
    "hold_contact_end_eval_time",
    "hold_contact_require_ref_contact",
]

E081_BASELINES = {
    "main": {
        "variant": "E081_box025_p2_legobj",
        "obj_mean": 0.14264468689907375,
        "obj_max": 0.2709930028783483,
        "hand_pct": 89.01734104046243,
        "floor_pct": 59.53757225433526,
        "leg_intf_pct": 7.514450867052023,
        "bottom_mean": -0.07451320489038624,
    },
    "guard": {
        "variant": "E081_box023_p2_legobj",
        "obj_mean": 0.16384071511984152,
        "obj_max": 0.31690969044561995,
        "hand_pct": 66.66666666666666,
        "floor_pct": 34.66666666666667,
        "leg_intf_pct": 2.666666666666667,
        "bottom_mean": 0.14377670659614467,
    },
}

E008_BEST_MAIN = {
    "variant": "E008_box025_p2_ypos_k20_vmax2",
    "obj_mean": 0.3630407740420036,
    "obj_max": 0.6836066873092047,
    "hand_pct": 78.03468208092485,
    "floor_pct": 64.73988439306359,
    "leg_intf_pct": 2.8901734104046244,
    "xy_ratio": 0.724105046574222,
    "rot_deg": 13.595337781924936,
}


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


def _flatten_npz(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray | None:
    if key not in data:
        return None
    return e002.e072.flatten_time_major(data[key])


def _window(arr: np.ndarray, summary: dict[str, object]) -> np.ndarray:
    start = min(int(summary["case_window_start_frame"]), len(arr) - 1)
    end = min(int(summary["case_window_end_frame"]) + 1, len(arr))
    return arr[start:end]


def _window2(arr: np.ndarray, summary: dict[str, object]) -> np.ndarray:
    start = min(int(summary["case_window_start_frame"]), len(arr) - 1)
    end = min(int(summary["case_window_end_frame"]) + 1, len(arr))
    return arr[start:end]


def _add_support_proxy_metrics(
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

    point_local = np.array(
        [
            float(meta["point_local_x"]),
            float(meta["point_local_y"]),
            float(meta["point_local_z"]),
        ],
        dtype=np.float64,
    )
    summary["E010_point_local"] = point_local.tolist()
    summary["E010_support_proxy_metrics_present"] = bool(
        force is not None and torque is not None and proxy_pos is not None and support_point is not None
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
            ref_obj_quat, np.broadcast_to(point_local, (len(qpos_ref), 3))
        )

        obj_xy = _xy_disp(obj_pos)
        ref_xy = _xy_disp(ref_obj_pos)
        ref_support_xy = _xy_disp(ref_support)
        summary["E010_object_xy_disp_m"] = obj_xy
        summary["E010_ref_object_xy_disp_m"] = ref_xy
        summary["E010_ref_support_xy_disp_m"] = ref_support_xy
        summary["E010_object_xy_disp_ratio"] = float(obj_xy / max(ref_xy, 1e-8))
        summary["E010_object_xy_disp_ratio_vs_ref_support"] = float(
            obj_xy / max(ref_support_xy, 1e-8)
        )
        summary["E010_object_rot_deg"] = _quat_angle_deg(obj_quat[0], obj_quat[-1])
        summary["E010_ref_object_rot_deg"] = _quat_angle_deg(
            ref_obj_quat[0], ref_obj_quat[-1]
        )

        obj_pos_cw = _window2(obj_pos, summary)
        ref_pos_cw = _window2(ref_obj_pos, summary)
        cw_obj_xy = _xy_disp(obj_pos_cw)
        cw_ref_xy = _xy_disp(ref_pos_cw)
        summary["E010_case_window_object_xy_disp_m"] = cw_obj_xy
        summary["E010_case_window_ref_object_xy_disp_m"] = cw_ref_xy
        summary["E010_case_window_object_xy_disp_ratio"] = float(
            cw_obj_xy / max(cw_ref_xy, 1e-8)
        )

        support = obj_pos + _quat_apply(
            obj_quat, np.broadcast_to(point_local, (len(qpos), 3))
        )
        opposite_local = np.array(
            [-point_local[0], -point_local[1], point_local[2]], dtype=np.float64
        )
        opposite = obj_pos + _quat_apply(
            obj_quat, np.broadcast_to(opposite_local, (len(qpos), 3))
        )
        end_height_diff = np.abs(support[:, 2] - opposite[:, 2])
        cw_end_height = _window(end_height_diff, summary)
        summary["E010_case_window_end_height_diff_mean_m"] = float(
            cw_end_height.mean()
        )
        summary["E010_case_window_end_height_diff_max_m"] = float(cw_end_height.max())

    if not summary["E010_support_proxy_metrics_present"]:
        return

    if proxy_pos is not None and qpos is not None and qpos_ref.shape[1] >= 7:
        T = min(len(proxy_pos), len(qpos_ref))
        proxy_xy = _xy_disp(proxy_pos[:T])
        ref_obj_pos = qpos_ref[:T, -7:-4].astype(np.float64)
        ref_obj_quat = qpos_ref[:T, -4:].astype(np.float64)
        ref_support = ref_obj_pos + _quat_apply(
            ref_obj_quat, np.broadcast_to(point_local, (T, 3))
        )
        ref_xy = _xy_disp(ref_obj_pos)
        ref_support_xy = _xy_disp(ref_support)
        final_gap = float(np.linalg.norm(proxy_pos[T - 1] - ref_support[T - 1]))
        summary["E010_proxy_xy_disp_m"] = proxy_xy
        summary["E010_proxy_xy_disp_ratio_vs_ref_obj"] = float(
            proxy_xy / max(ref_xy, 1e-8)
        )
        summary["E010_proxy_xy_disp_ratio_vs_ref_support"] = float(
            proxy_xy / max(ref_support_xy, 1e-8)
        )
        summary["E010_proxy_final_gap_to_ref_support_m"] = final_gap
        summary["E010_proxy_support_tracking_ok"] = bool(
            summary["E010_proxy_xy_disp_ratio_vs_ref_support"] >= 0.95
            and final_gap <= 0.08
        )
        summary["E010_proxy_timebase_ok"] = summary["E010_proxy_support_tracking_ok"]

    force_norm = np.linalg.norm(force, axis=1)
    force_h = np.linalg.norm(force[:, :2], axis=1)
    force_z = force[:, 2]
    torque_norm = np.linalg.norm(torque, axis=1)
    gap = proxy_pos - support_point
    gap_norm = np.linalg.norm(gap, axis=1)
    gap_h = np.linalg.norm(gap[:, :2], axis=1)

    for name, arr in [
        ("force_norm_n", force_norm),
        ("force_horizontal_n", force_h),
        ("force_z_n", force_z),
        ("torque_norm_nm", torque_norm),
        ("connector_gap_norm_m", gap_norm),
        ("connector_gap_horizontal_m", gap_h),
    ]:
        cw = _window(arr, summary)
        summary[f"E010_case_window_{name}_mean"] = float(cw.mean())
        summary[f"E010_case_window_{name}_max"] = float(cw.max())


def evaluate_variant(
    variant: str, variants: dict[str, dict[str, str]]
) -> dict[str, object]:
    if variant not in variants:
        raise ValueError(f"Unknown E010 variant: {variant}")

    e002.RESULTS = RESULTS
    e002.VARIANTS_FILE = VARIANTS_FILE
    summary = e002.evaluate_variant(variant, variants)
    meta = variants[variant]
    npz_path = RESULTS / f"{variant}.npz"

    _qpos_ref, _ctrl_ref, cfg = e002.load_ref(str(meta["override"]), str(meta["case"]))
    _add_support_proxy_metrics(summary, meta, npz_path, _qpos_ref)

    summary["E010_wave"] = meta["wave"]
    summary["E010_queue"] = meta["queue"]
    summary["E010_pad_size_m"] = float(meta["pad_size"])
    summary["E010_support_proxy_mode"] = str(cfg.support_proxy_mode)
    summary["E010_support_proxy_gravity_scale"] = float(cfg.support_proxy_gravity_scale)
    summary["E010_support_proxy_connector_kp"] = float(cfg.support_proxy_connector_kp)
    summary["E010_support_proxy_connector_kd"] = float(cfg.support_proxy_connector_kd)
    summary["E010_support_proxy_xy_velocity_scale"] = float(
        cfg.support_proxy_xy_velocity_scale
    )
    summary["E010_support_proxy_max_xy_speed"] = float(
        meta["support_proxy_max_xy_speed"]
    )
    summary["E010_support_proxy_height_tau"] = float(cfg.support_proxy_height_tau)
    summary["E010_support_proxy_ref_dt"] = float(meta["support_proxy_ref_dt"])
    summary["E010_support_proxy_force_clamp"] = float(cfg.support_proxy_force_clamp)
    summary["E010_support_proxy_torque_clamp"] = float(cfg.support_proxy_torque_clamp)
    summary["E010_hold_contact_rew_scale"] = float(meta["hold_contact_rew_scale"])
    summary["E010_hold_contact_start_eval_time"] = float(
        meta["hold_contact_start_eval_time"]
    )
    summary["E010_hold_contact_end_eval_time"] = float(
        meta["hold_contact_end_eval_time"]
    )
    summary["E010_freejoint_parity_ok"] = bool(
        not summary["config_contact_guidance"]
        and str(summary["config_scene_name"]).startswith("scene_contact_pad")
        and summary["config_nu"] == 29
        and summary["config_nq_obj"] == 7
        and summary["config_object_action_dims"] == 0
        and len(summary["config_object_actuator_ids"]) == 0
        and bool(cfg.support_proxy_enabled)
        and str(cfg.support_proxy_mode) == "mocap_pad"
        and float(cfg.partner_force_scale) == 0.0
    )

    baseline = E081_BASELINES["guard" if summary["role"] == "guard" else "main"]
    summary["E010_e081_baseline_variant"] = baseline["variant"]
    summary["E010_vs_E081_obj_mean_delta_m"] = float(
        summary["case_window_obj_err_mean_m"] - baseline["obj_mean"]
    )
    summary["E010_vs_E081_obj_max_delta_m"] = float(
        summary["case_window_obj_err_max_m"] - baseline["obj_max"]
    )
    summary["E010_vs_E081_hand_contact_delta_pp"] = float(
        summary["case_window_sim_contact_frames_pct"] - baseline["hand_pct"]
    )
    summary["E010_vs_E081_floor_contact_delta_pp"] = float(
        summary["case_window_sim_object_floor_contact_frames_pct"]
        - baseline["floor_pct"]
    )
    summary["E010_vs_E081_leg_intf_delta_pp"] = float(
        summary["case_window_sim_leg_box_interference_frames_pct"]
        - baseline["leg_intf_pct"]
    )
    summary["E010_vs_E081_bottom_mean_delta_m"] = float(
        summary["case_window_sim_object_bottom_proxy_mean_m"]
        - baseline["bottom_mean"]
    )

    majority_checks = {
        "obj_mean": summary["case_window_obj_err_mean_m"]
        <= baseline["obj_mean"] + 0.05,
        "obj_max": summary["case_window_obj_err_max_m"]
        <= baseline["obj_max"] + 0.10,
        "hand_contact": summary["case_window_sim_contact_frames_pct"]
        >= baseline["hand_pct"] - 10.0,
        "floor_contact": summary["case_window_sim_object_floor_contact_frames_pct"]
        <= baseline["floor_pct"] + 15.0,
        "leg_intf": summary["case_window_sim_leg_box_interference_frames_pct"]
        <= baseline["leg_intf_pct"] + 5.0,
        "xy_transport": summary.get("E010_object_xy_disp_ratio", 0.0) >= 0.75,
    }
    summary["E010_e081_majority_score"] = int(sum(majority_checks.values()))
    summary["E010_e081_majority_checks"] = json.dumps(
        majority_checks, sort_keys=True
    )
    summary["E010_beats_or_matches_E081_majority"] = bool(
        summary["role"] == "main" and summary["E010_e081_majority_score"] >= 4
    )

    if summary["role"] == "main":
        summary["E010_e008_best_variant"] = E008_BEST_MAIN["variant"]
        summary["E010_vs_E008_best_obj_mean_delta_m"] = float(
            summary["case_window_obj_err_mean_m"] - E008_BEST_MAIN["obj_mean"]
        )
        summary["E010_vs_E008_best_obj_max_delta_m"] = float(
            summary["case_window_obj_err_max_m"] - E008_BEST_MAIN["obj_max"]
        )
        summary["E010_vs_E008_best_hand_contact_delta_pp"] = float(
            summary["case_window_sim_contact_frames_pct"] - E008_BEST_MAIN["hand_pct"]
        )
        summary["E010_vs_E008_best_floor_contact_delta_pp"] = float(
            summary["case_window_sim_object_floor_contact_frames_pct"]
            - E008_BEST_MAIN["floor_pct"]
        )
        summary["E010_vs_E008_best_leg_intf_delta_pp"] = float(
            summary["case_window_sim_leg_box_interference_frames_pct"]
            - E008_BEST_MAIN["leg_intf_pct"]
        )
        summary["E010_vs_E008_best_xy_ratio_delta"] = float(
            summary.get("E010_object_xy_disp_ratio", 0.0)
            - E008_BEST_MAIN["xy_ratio"]
        )
        summary["E010_vs_E008_best_rot_delta_deg"] = float(
            summary.get("E010_object_rot_deg", 180.0) - E008_BEST_MAIN["rot_deg"]
        )
        summary["E010_improves_E008_best"] = bool(
            summary["case_window_obj_err_mean_m"] <= E008_BEST_MAIN["obj_mean"]
            and summary["case_window_obj_err_max_m"] <= E008_BEST_MAIN["obj_max"]
            and summary["case_window_sim_contact_frames_pct"] >= 80.0
            and summary.get("E010_object_xy_disp_ratio", 0.0)
            >= E008_BEST_MAIN["xy_ratio"]
            and summary["case_window_sim_leg_box_interference_frames_pct"]
            <= E081_BASELINES["main"]["leg_intf_pct"] + 5.0
        )
    else:
        summary["E010_improves_E008_best"] = False

    effort_ok = True
    if summary["E010_support_proxy_metrics_present"]:
        effort_ok = bool(
            summary["E010_case_window_force_norm_n_mean"] < 80.0
            and summary["E010_case_window_force_norm_n_max"] < 160.0
            and summary["E010_case_window_torque_norm_nm_max"] < 25.0
        )
    summary["E010_effort_reasonable_proxy"] = effort_ok
    summary["E010_reaches_E081_transport_proxy"] = bool(
        summary["role"] == "main"
        and summary.get("E010_proxy_support_tracking_ok", False)
        and summary["case_window_obj_err_mean_m"] <= 0.20
        and summary["case_window_obj_err_max_m"] <= 0.40
        and summary["case_window_sim_object_floor_contact_frames_pct"] <= 75.0
        and summary["case_window_sim_contact_frames_pct"] >= 80.0
        and summary.get("E010_object_xy_disp_ratio", 0.0) >= 0.75
        and summary.get("E010_object_rot_deg", 180.0) <= 15.0
        and effort_ok
        and summary["E010_freejoint_parity_ok"]
    )
    summary["E010_guard_stable_proxy"] = bool(
        summary["role"] == "guard"
        and summary["post2_pelvis_z_min_m"] >= 0.55
        and summary["case_window_sim_leg_box_interference_frames_pct"]
        <= E081_BASELINES["guard"]["leg_intf_pct"] + 5.0
        and summary["E010_freejoint_parity_ok"]
    )

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
        raise SystemExit("No E010 variant results found.")

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
            bool(r["E010_freejoint_parity_ok"]) for r in summaries
        ),
        "num_support_proxy_metrics_present": sum(
            bool(r["E010_support_proxy_metrics_present"]) for r in summaries
        ),
        "num_main_proxy_timebase_ok": sum(
            bool(r.get("E010_proxy_timebase_ok", False)) for r in main_rows
        ),
        "num_main_proxy_support_tracking_ok": sum(
            bool(r.get("E010_proxy_support_tracking_ok", False)) for r in main_rows
        ),
        "num_main_reaches_E081_transport_proxy": sum(
            bool(r["E010_reaches_E081_transport_proxy"]) for r in main_rows
        ),
        "num_main_beats_or_matches_E081_majority": sum(
            bool(r["E010_beats_or_matches_E081_majority"]) for r in main_rows
        ),
        "num_main_improves_E008_best": sum(
            bool(r.get("E010_improves_E008_best", False)) for r in main_rows
        ),
        "num_guard_stable_proxy": sum(
            bool(r["E010_guard_stable_proxy"]) for r in guard_rows
        ),
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
