#!/usr/bin/env python3
"""E006 evaluation wrapper for COLA-style support-body proxy sweep."""

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


RESULTS = REPO / "workspace/core4d_collab_retarget/results/E006"
VARIANTS_FILE = REPO / "workspace/core4d_collab_retarget/scripts/E006/variants.tsv"

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
    "support_proxy_gravity_scale",
    "support_proxy_connector_kp",
    "support_proxy_connector_kd",
    "support_proxy_xy_velocity_scale",
    "support_proxy_max_xy_speed",
    "support_proxy_height_tau",
    "support_proxy_ref_dt",
    "support_proxy_force_clamp",
    "support_proxy_torque_clamp",
    "hold_contact_rew_scale",
    "hold_contact_sigma",
    "hold_contact_start_eval_time",
    "hold_contact_end_eval_time",
    "hold_contact_require_ref_contact",
]

E005_SUPPORT_BASELINES = {
    "yneg": {"obj_mean": 0.759939, "floor_pct": 96.53},
    "ypos": {"obj_mean": 0.709838, "floor_pct": 94.22},
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


def _flatten_npz(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray | None:
    if key not in data:
        return None
    return e002.e072.flatten_time_major(data[key])


def _window(arr: np.ndarray, summary: dict[str, object]) -> np.ndarray:
    start = min(int(summary["case_window_start_frame"]), len(arr) - 1)
    end = min(int(summary["case_window_end_frame"]) + 1, len(arr))
    return arr[start:end]


def _add_support_proxy_metrics(
    summary: dict[str, object],
    meta: dict[str, str],
    npz_path: Path,
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
    summary["E006_point_local"] = point_local.tolist()
    summary["E006_support_proxy_metrics_present"] = bool(
        force is not None and torque is not None and proxy_pos is not None and support_point is not None
    )

    if qpos is not None and qpos.shape[1] >= 7:
        obj_pos = qpos[:, -7:-4].astype(np.float64)
        obj_quat = qpos[:, -4:].astype(np.float64)
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
        summary["E006_case_window_end_height_diff_mean_m"] = float(
            cw_end_height.mean()
        )
        summary["E006_case_window_end_height_diff_max_m"] = float(cw_end_height.max())

    if not summary["E006_support_proxy_metrics_present"]:
        return

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
        summary[f"E006_case_window_{name}_mean"] = float(cw.mean())
        summary[f"E006_case_window_{name}_max"] = float(cw.max())


def evaluate_variant(
    variant: str, variants: dict[str, dict[str, str]]
) -> dict[str, object]:
    if variant not in variants:
        raise ValueError(f"Unknown E006 variant: {variant}")

    e002.RESULTS = RESULTS
    e002.VARIANTS_FILE = VARIANTS_FILE
    summary = e002.evaluate_variant(variant, variants)
    meta = variants[variant]
    npz_path = RESULTS / f"{variant}.npz"

    _add_support_proxy_metrics(summary, meta, npz_path)
    _qpos_ref, _ctrl_ref, cfg = e002.load_ref(str(meta["override"]), str(meta["case"]))

    summary["E006_wave"] = meta["wave"]
    summary["E006_queue"] = meta["queue"]
    summary["E006_support_proxy_gravity_scale"] = float(
        meta["support_proxy_gravity_scale"]
    )
    summary["E006_support_proxy_connector_kp"] = float(
        meta["support_proxy_connector_kp"]
    )
    summary["E006_support_proxy_connector_kd"] = float(
        meta["support_proxy_connector_kd"]
    )
    summary["E006_support_proxy_xy_velocity_scale"] = float(
        meta["support_proxy_xy_velocity_scale"]
    )
    summary["E006_support_proxy_max_xy_speed"] = float(
        meta["support_proxy_max_xy_speed"]
    )
    summary["E006_support_proxy_height_tau"] = float(meta["support_proxy_height_tau"])
    summary["E006_support_proxy_ref_dt"] = float(meta["support_proxy_ref_dt"])
    summary["E006_support_proxy_force_clamp"] = float(
        meta["support_proxy_force_clamp"]
    )
    summary["E006_support_proxy_torque_clamp"] = float(
        meta["support_proxy_torque_clamp"]
    )
    summary["E006_hold_contact_rew_scale"] = float(meta["hold_contact_rew_scale"])
    summary["E006_freejoint_parity_ok"] = bool(
        not summary["config_contact_guidance"]
        and summary["config_scene_name"] == ""
        and summary["config_nu"] == 29
        and summary["config_nq_obj"] == 7
        and summary["config_object_action_dims"] == 0
        and len(summary["config_object_actuator_ids"]) == 0
        and bool(cfg.support_proxy_enabled)
        and float(cfg.partner_force_scale) == 0.0
    )

    side = "ypos" if float(meta["point_local_y"]) > 0 else "yneg"
    baseline = E005_SUPPORT_BASELINES.get(side)
    summary["E006_beats_E005_support_site_proxy"] = False
    if summary["role"] == "main" and baseline:
        summary["E006_beats_E005_support_site_proxy"] = bool(
            summary["case_window_obj_err_mean_m"] <= baseline["obj_mean"] - 0.10
            or summary["case_window_sim_object_floor_contact_frames_pct"]
            <= baseline["floor_pct"] - 15.0
        )

    effort_ok = True
    if summary["E006_support_proxy_metrics_present"]:
        effort_ok = bool(
            summary["E006_case_window_force_norm_n_mean"] < 80.0
            and summary["E006_case_window_force_norm_n_max"] < 160.0
            and summary["E006_case_window_torque_norm_nm_max"] < 25.0
        )
    summary["E006_effort_reasonable_proxy"] = effort_ok
    summary["E006_useful_main_proxy"] = bool(
        summary["role"] == "main"
        and summary["case_window_obj_err_mean_m"] < 0.30
        and summary["case_window_obj_err_max_m"] < 0.70
        and summary["case_window_sim_object_floor_contact_frames_pct"] < 76.9
        and summary["case_window_sim_contact_frames_pct"] >= 80.0
        and effort_ok
        and summary["E006_freejoint_parity_ok"]
    )
    summary["E006_guard_stable_proxy"] = bool(
        summary["role"] == "guard"
        and summary["post2_pelvis_z_min_m"] >= 0.55
        and summary["case_window_sim_leg_box_interference_frames_pct"] <= 5.0
        and summary["E006_freejoint_parity_ok"]
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
        raise SystemExit("No E006 variant results found.")

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
            bool(r["E006_freejoint_parity_ok"]) for r in summaries
        ),
        "num_support_proxy_metrics_present": sum(
            bool(r["E006_support_proxy_metrics_present"]) for r in summaries
        ),
        "num_main_beats_E005_support_site_proxy": sum(
            bool(r["E006_beats_E005_support_site_proxy"]) for r in main_rows
        ),
        "num_main_useful_proxy": sum(
            bool(r["E006_useful_main_proxy"]) for r in main_rows
        ),
        "num_guard_stable_proxy": sum(
            bool(r["E006_guard_stable_proxy"]) for r in guard_rows
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
