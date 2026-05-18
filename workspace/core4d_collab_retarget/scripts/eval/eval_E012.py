#!/usr/bin/env python3
"""E012 evaluation wrapper for dual-point partner pose closure."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
SCRIPT_EVAL = REPO / "workspace/core4d_collab_retarget/scripts/eval"
if str(SCRIPT_EVAL) not in sys.path:
    sys.path.insert(0, str(SCRIPT_EVAL))

import eval_E011 as e011  # noqa: E402


RESULTS = REPO / "workspace/core4d_collab_retarget/results/E012"
VARIANTS_FILE = REPO / "workspace/core4d_collab_retarget/scripts/E012/variants.tsv"

FIELDNAMES = [
    *e011.FIELDNAMES,
    "partner_force_points_local",
    "task_obj_pos_rew_scale",
    "task_obj_rot_rew_scale",
    "contact_hdmi_gain",
]

E011_BEST_MAIN = {
    "variant": "E011_box025_p2_com_xyz_k100",
    "obj_mean": 0.34043773857065575,
    "obj_max": 0.6725313166356337,
    "hand_pct": 86.1271676300578,
    "floor_pct": 61.27167630057804,
    "xy_ratio": 0.9047539606395267,
    "rot_deg": 3.508741448970702,
}


def _configure_e011() -> None:
    e011.RESULTS = RESULTS
    e011.VARIANTS_FILE = VARIANTS_FILE
    e011.FIELDNAMES = FIELDNAMES


def read_variants() -> dict[str, dict[str, str]]:
    _configure_e011()
    return e011.read_variants()


def parse_points(value: str) -> list[list[float]]:
    points: list[list[float]] = []
    for raw_point in value.split(";"):
        raw_point = raw_point.strip()
        if not raw_point:
            continue
        coords = [float(v.strip()) for v in raw_point.split(",")]
        if len(coords) != 3:
            raise ValueError(f"Invalid partner_force_points_local entry: {raw_point}")
        points.append(coords)
    return points


def _alias_e011_fields(summary: dict[str, object]) -> None:
    for key, value in list(summary.items()):
        if key.startswith("E011_"):
            summary[f"E012_{key[5:]}"] = value


def evaluate_variant(
    variant: str, variants: dict[str, dict[str, str]]
) -> dict[str, object]:
    _configure_e011()
    if variant not in variants:
        raise ValueError(f"Unknown E012 variant: {variant}")

    summary = e011.evaluate_variant(variant, variants)
    meta = variants[variant]
    points = parse_points(meta["partner_force_points_local"])
    _qpos_ref, _ctrl_ref, cfg = e011.e002.load_ref(
        str(meta["override"]), str(meta["case"])
    )

    cfg_points = [list(map(float, p)) for p in list(cfg.partner_force_points_local)]
    summary["E012_partner_force_points_local"] = points
    summary["E012_partner_force_points_local_config"] = cfg_points
    summary["E012_dual_point_count"] = len(points)
    summary["E012_task_obj_pos_rew_scale"] = float(meta["task_obj_pos_rew_scale"])
    summary["E012_task_obj_rot_rew_scale"] = float(meta["task_obj_rot_rew_scale"])
    summary["E012_contact_hdmi_gain"] = float(meta["contact_hdmi_gain"])
    summary["E012_vs_E011_best_variant"] = E011_BEST_MAIN["variant"]
    summary["E012_vs_E011_best_obj_mean_delta_m"] = float(
        summary["case_window_obj_err_mean_m"] - E011_BEST_MAIN["obj_mean"]
    )
    summary["E012_vs_E011_best_obj_max_delta_m"] = float(
        summary["case_window_obj_err_max_m"] - E011_BEST_MAIN["obj_max"]
    )
    summary["E012_vs_E011_best_hand_contact_delta_pp"] = float(
        summary["case_window_sim_contact_frames_pct"] - E011_BEST_MAIN["hand_pct"]
    )
    summary["E012_vs_E011_best_floor_contact_delta_pp"] = float(
        summary["case_window_sim_object_floor_contact_frames_pct"]
        - E011_BEST_MAIN["floor_pct"]
    )
    summary["E012_vs_E011_best_xy_ratio_delta"] = float(
        summary.get("E011_object_xy_disp_ratio", 0.0) - E011_BEST_MAIN["xy_ratio"]
    )
    summary["E012_vs_E011_best_rot_delta_deg"] = float(
        summary.get("E011_object_rot_deg", 180.0) - E011_BEST_MAIN["rot_deg"]
    )

    _alias_e011_fields(summary)

    dual_points_config_ok = bool(
        len(cfg_points) >= 2
        and len(list(cfg.partner_force_point_local or [])) == 0
        and len(cfg_points) == len(points)
    )
    summary["E012_dual_points_config_ok"] = dual_points_config_ok
    summary["E012_freejoint_parity_ok"] = bool(
        summary["E011_freejoint_parity_ok"] and dual_points_config_ok
    )

    partner_effort_ok = False
    if summary["E011_partner_force_metrics_present"]:
        partner_effort_ok = bool(
            summary["E011_case_window_partner_force_norm_n_mean"] <= 120.0
            and summary["E011_case_window_partner_force_norm_n_max"] <= 250.0
            and summary["E011_case_window_partner_torque_norm_nm_max"] <= 30.0
        )
    summary["E012_partner_effort_reasonable"] = partner_effort_ok

    object_gate = bool(
        summary["role"] == "main"
        and summary["case_window_obj_err_mean_m"] <= 0.20
        and summary["case_window_obj_err_max_m"] <= 0.40
        and summary.get("E011_object_xy_disp_ratio", 0.0) >= 0.75
        and summary.get("E011_object_rot_deg", 180.0) <= 15.0
    )
    contact_gate = bool(
        summary["case_window_sim_contact_frames_pct"] >= 80.0
        and summary["case_window_sim_object_floor_contact_frames_pct"] <= 75.0
    )
    summary["E012_reaches_E081_transport"] = bool(
        object_gate
        and contact_gate
        and partner_effort_ok
        and summary["E012_freejoint_parity_ok"]
    )
    summary["E012_external_only_success"] = bool(object_gate and not contact_gate)
    summary["E012_pose_closure_helped"] = bool(
        summary["role"] == "main"
        and summary["case_window_obj_err_mean_m"] <= E011_BEST_MAIN["obj_mean"] - 0.04
        and summary["case_window_obj_err_max_m"] <= E011_BEST_MAIN["obj_max"] - 0.08
        and summary.get("E011_object_xy_disp_ratio", 0.0) >= 0.75
        and summary.get("E011_object_rot_deg", 180.0) <= 15.0
    )
    summary["E012_guard_stable"] = bool(
        summary["role"] == "guard"
        and summary["post2_pelvis_z_min_m"] >= 0.55
        and summary["case_window_sim_leg_box_interference_frames_pct"]
        <= e011.E081_BASELINES["guard"]["leg_intf_pct"] + 5.0
        and summary["E012_freejoint_parity_ok"]
    )

    if summary["E012_reaches_E081_transport"]:
        diagnostic_class = "physical_candidate"
    elif summary["E012_external_only_success"]:
        diagnostic_class = "external_only_success"
    elif summary["E012_pose_closure_helped"]:
        diagnostic_class = "pose_closure_helped"
    elif summary["role"] == "guard" and not summary["E012_guard_stable"]:
        diagnostic_class = "guard_unstable"
    elif (
        summary.get("E011_object_rot_deg", 0.0) > 30.0
        or summary["case_window_sim_object_floor_contact_frames_pct"] > 80.0
    ):
        diagnostic_class = "rotation_shortcut"
    else:
        diagnostic_class = "insufficient_coupling"
    summary["E012_diagnostic_class"] = diagnostic_class

    e011._write_summary(summary)
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
        raise SystemExit("No E012 variant results found.")

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
            bool(r["E012_freejoint_parity_ok"]) for r in summaries
        ),
        "num_dual_points_config_ok": sum(
            bool(r["E012_dual_points_config_ok"]) for r in summaries
        ),
        "num_partner_force_metrics_present": sum(
            bool(r["E011_partner_force_metrics_present"]) for r in summaries
        ),
        "num_main_reaches_E081_transport": sum(
            bool(r["E012_reaches_E081_transport"]) for r in main_rows
        ),
        "num_main_external_only_success": sum(
            bool(r["E012_external_only_success"]) for r in main_rows
        ),
        "num_main_pose_closure_helped": sum(
            bool(r["E012_pose_closure_helped"]) for r in main_rows
        ),
        "num_guard_stable": sum(bool(r["E012_guard_stable"]) for r in guard_rows),
        "diagnostic_classes": {
            name: sum(r.get("E012_diagnostic_class") == name for r in summaries)
            for name in sorted({str(r.get("E012_diagnostic_class")) for r in summaries})
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
