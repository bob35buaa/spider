#!/usr/bin/env python3
"""E018b evaluation: anchor audit/selection candidates."""

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
import paper_metrics  # noqa: E402


BASE = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
RESULTS = REPO / "workspace/core4d_collab_retarget/results/E018b"
MANIFEST = RESULTS / "manifest.tsv"


def read_manifest() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    with MANIFEST.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            out[row["variant"]] = {
                "name": row["variant"],
                "case": row["derived_task"],
                "source_task": row["source_task"],
                "override": f"core4d_collab_{row['variant']}",
                "split": row["queue"],
                "role": row["role"],
                **row,
            }
    return out


def _write_summary(summary: dict[str, object]) -> None:
    variant = str(summary["variant"])
    (RESULTS / f"eval_summary_{variant}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (RESULTS / f"eval_summary_{variant}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def _point_local(meta: dict[str, str]) -> np.ndarray:
    return np.array(
        [
            float(meta["support_proxy_point_local_x"]),
            float(meta["support_proxy_point_local_y"]),
            float(meta["support_proxy_point_local_z"]),
        ],
        dtype=np.float64,
    )


def _gt_point(meta: dict[str, str]) -> np.ndarray | None:
    if str(meta.get("gt_anchor_available", "")).lower() != "true":
        return None
    return np.array(
        [
            float(meta["gt_anchor_x"]),
            float(meta["gt_anchor_y"]),
            float(meta["gt_anchor_z"]),
        ],
        dtype=np.float64,
    )


def _scene_anchor_not_oracle(meta: dict[str, str]) -> bool:
    scene = BASE / meta["derived_task"] / f"{meta['scene_name']}.xml"
    if not scene.is_file():
        return False
    text = scene.read_text(encoding="utf-8")
    return bool(
        np.linalg.norm(_point_local(meta)) > 0.05
        and "object_target" not in text
        and "support_weld_anchor" in text
        and "e018b_support_weld" in text
    )


def _canonical_anchor_ok(meta: dict[str, str], point: np.ndarray, z_frac: float) -> bool:
    face = meta.get("anchor_face", "")
    half = np.array(
        [
            float(meta.get("object_half_x", "nan")),
            float(meta.get("object_half_y", "nan")),
            float(meta.get("object_half_z", "nan")),
        ],
        dtype=np.float64,
    )
    if face not in {"+x", "-x", "+y", "-y"} or not np.all(np.isfinite(half)):
        return False
    axis = 0 if face.endswith("x") else 1
    other = 1 - axis
    sign = 1.0 if face.startswith("+") else -1.0
    return bool(
        abs(point[axis] - sign * half[axis]) <= 1e-5
        and abs(point[other]) <= 1e-5
        and 0.55 <= z_frac <= 0.70
        and point[2] > 0.0
    )


def _as_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def evaluate_variant(variant: str, variants: dict[str, dict[str, str]]) -> dict[str, object]:
    if variant not in variants:
        raise ValueError(f"Unknown E018b variant: {variant}")

    e002.RESULTS = RESULTS
    e002.VARIANTS_FILE = MANIFEST
    meta = variants[variant]
    summary = e002.evaluate_variant(variant, variants)
    npz_path = RESULTS / f"{variant}.npz"
    qpos_ref, _ctrl_ref, cfg = e002.load_ref(str(meta["override"]), str(meta["case"]))
    model, _scene_used = e002.load_scene_model(str(meta["case"]))
    data_npz = np.load(npz_path, allow_pickle=True)
    qpos = e002.e072.flatten_time_major(data_npz["qpos"])
    summary.update(
        paper_metrics.add_paper_metrics(
            summary,
            repo=REPO,
            results_dir=RESULTS,
            model=model,
            qpos=qpos,
            qpos_ref=qpos_ref,
            person_idx=int(meta["person_idx"]),
        )
    )

    summary["E018b_queue"] = meta["queue"]
    summary["E018b_wave"] = meta["wave"]
    summary["E018b_role"] = meta["role"]
    summary["E018b_scene_name"] = meta["scene_name"]
    summary["E018b_point_local"] = _point_local(meta).tolist()
    summary["E018b_support_point_method"] = meta["support_point_method"]
    summary["E018b_anchor_policy"] = meta.get("anchor_policy", "")
    summary["E018b_anchor_face"] = meta.get("anchor_face", "")
    summary["E018b_anchor_face_source"] = meta.get("anchor_face_source", "")
    summary["E018b_anchor_audit_class"] = meta.get("anchor_audit_class", "")
    summary["E018b_anchor_current_face"] = meta.get("anchor_current_face", "")
    summary["E018b_anchor_selected_face"] = meta.get("anchor_selected_face", "")
    summary["E018b_anchor_top_face"] = meta.get("anchor_top_face", "")
    summary["E018b_anchor_top_face_frac"] = float(meta.get("anchor_top_face_frac") or "nan")
    summary["E018b_anchor_partner_top_face"] = meta.get("anchor_partner_top_face", "")
    summary["E018b_anchor_partner_top_face_frac"] = float(meta.get("anchor_partner_top_face_frac") or "nan")
    summary["E018b_anchor_selected_partner_top_relation"] = meta.get("anchor_selected_partner_top_relation", "")
    summary["E018b_anchor_centroid_cancellation"] = str(meta.get("anchor_centroid_cancellation", "")).lower() == "true"
    summary["E018b_anchor_low_support"] = str(meta.get("anchor_low_support", "")).lower() == "true"
    summary["E018b_canonical_z_frac"] = float(meta.get("canonical_z_frac", "nan"))
    summary["E018b_source_variant"] = meta.get("source_variant", "")
    summary["E018b_gt_anchor_available"] = str(meta.get("gt_anchor_available", "")).lower() == "true"
    summary["E018b_gt_anchor_dist_manifest_m"] = float(meta.get("gt_anchor_dist_m", "nan"))
    summary["E018b_gt_anchor_face"] = meta.get("gt_anchor_face", "")
    summary["E018b_gt_anchor_face_match"] = bool(
        summary["E018b_gt_anchor_available"]
        and str(summary["E018b_anchor_face"]).lower() == str(summary["E018b_gt_anchor_face"]).lower()
    )
    point = _point_local(meta)
    gt = _gt_point(meta)
    if gt is not None:
        summary["E018b_gt_anchor_dist_m"] = float(np.linalg.norm(point - gt))
        summary["E018b_gt_anchor_xy_dist_m"] = float(np.linalg.norm(point[:2] - gt[:2]))
        summary["E018b_gt_anchor_z_abs_m"] = float(abs(point[2] - gt[2]))
    else:
        summary["E018b_gt_anchor_dist_m"] = float("nan")
        summary["E018b_gt_anchor_xy_dist_m"] = float("nan")
        summary["E018b_gt_anchor_z_abs_m"] = float("nan")
    half_z = float(meta.get("object_half_z", "nan"))
    z_frac = float(point[2] / half_z) if half_z > 1e-8 else float("nan")
    summary["E018b_anchor_z_frac_of_half"] = z_frac
    summary["E018b_canonical_anchor_pass"] = _canonical_anchor_ok(meta, point, z_frac)
    summary["E018b_gt_anchor_pass"] = bool(
        summary["E018b_gt_anchor_available"]
        and summary["E018b_gt_anchor_face_match"]
        and summary["E018b_gt_anchor_dist_m"] <= 0.03
        and 0.55 <= z_frac <= 0.70
    )
    summary["E018b_freejoint_parity_ok"] = bool(
        not summary["config_contact_guidance"]
        and str(summary["config_scene_name"]).startswith("scene_e018b_jointB_")
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
    summary["E018b_anchor_not_com_oracle"] = _scene_anchor_not_oracle(meta)
    summary["E018b_no_direct_wrench"] = bool(
        float(cfg.partner_force_scale) == 0.0
        and float(cfg.partner_force_spring_kp) == 0.0
        and float(cfg.support_proxy_connector_kp) == 0.0
    )
    summary["E018b_config_ok"] = bool(
        summary["E018b_freejoint_parity_ok"]
        and summary["E018b_anchor_not_com_oracle"]
        and summary["E018b_no_direct_wrench"]
    )

    artifact_ok = bool(
        summary.get("paper_omniretarget_robot_object_deep_penetration_duration_pct", 100.0) <= 20.0
        and summary.get("paper_omniretarget_robot_object_max_penetration_cm", 999.0) <= 5.0
        and summary.get("paper_omniretarget_foot_skating_duration_pct", 100.0) <= 60.0
        and summary.get("paper_omniretarget_contact_preservation_5cm_pct", 0.0) >= 50.0
    )
    floor_leg_ok = bool(
        summary["case_window_sim_object_floor_contact_frames_pct"] <= 80.0
        and summary["case_window_sim_leg_box_interference_frames_pct"] <= 15.0
    )
    pelvis_z_min = _as_float(summary.get("full_pelvis_z_min_m"))
    first_pelvis_lt45 = _as_float(summary.get("first_pelvis_z_lt_45cm_frame"), -1.0)
    robot_fall_detected = bool(np.isfinite(pelvis_z_min) and (pelvis_z_min < 0.45 or first_pelvis_lt45 >= 0))
    robot_upright_ok = not robot_fall_detected
    summary["E018b_robot_fall_detected"] = robot_fall_detected
    summary["E018b_robot_upright_ok"] = robot_upright_ok
    summary["E018b_visual_stability_ok"] = robot_upright_ok
    summary["E018b_artifact_ok"] = artifact_ok
    summary["E018b_floor_leg_ok"] = floor_leg_ok
    summary["E018b_generalization_pass"] = bool(
        summary["E018b_config_ok"]
        and summary["E018b_canonical_anchor_pass"]
        and summary["paper_dynaretarget_object_success"]
        and summary["paper_transport_success"]
        and robot_upright_ok
        and artifact_ok
        and floor_leg_ok
    )
    summary["E018b_gt_gate_pass"] = bool(
        summary["E018b_config_ok"]
        and summary["E018b_canonical_anchor_pass"]
        and summary["E018b_gt_anchor_pass"]
    )

    if not summary["E018b_config_ok"]:
        diag = "config_failed"
    elif not summary["E018b_canonical_anchor_pass"]:
        diag = "canonical_anchor_failed"
    elif summary["E018b_gt_anchor_available"] and not summary["E018b_gt_anchor_pass"]:
        diag = "gt_anchor_failed"
    elif not summary["paper_dynaretarget_object_success"]:
        diag = "object_tracking_failed"
    elif not summary["paper_transport_success"]:
        diag = "transport_failed"
    elif not robot_upright_ok:
        diag = "robot_fall_visual_fail"
    elif not summary.get("paper_omniretarget_contact_preservation_ok", False):
        diag = "contact_preservation_gap"
    elif not floor_leg_ok:
        diag = "push_or_leg_shortcut"
    elif not artifact_ok:
        diag = "artifact_failed"
    else:
        diag = "paper_generalization_pass"
    summary["E018b_diagnostic_class"] = diag

    _write_summary(summary)
    return summary


def normalize_args(args: list[str], variants: dict[str, dict[str, str]]) -> list[str]:
    if not args or args == ["--all"]:
        return list(variants.keys())
    out: list[str] = []
    for arg in args:
        if arg == "--all":
            out.extend(variants.keys())
        else:
            out.append(arg)
    return out


def main() -> None:
    variants = read_manifest()
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
        raise SystemExit("No E018b variant results found.")

    keys = sorted({k for row in summaries for k in row.keys()})
    comparison = RESULTS / "comparison.csv"
    with comparison.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summaries)

    aggregate = {
        "num_results": len(summaries),
        "num_config_ok": sum(bool(r["E018b_config_ok"]) for r in summaries),
        "num_paper_spider_success": sum(bool(r["paper_spider_object_success"]) for r in summaries),
        "num_paper_dynaretarget_success": sum(bool(r["paper_dynaretarget_object_success"]) for r in summaries),
        "num_transport_success": sum(bool(r["paper_transport_success"]) for r in summaries),
        "num_contact_preservation_ok": sum(bool(r.get("paper_omniretarget_contact_preservation_ok", False)) for r in summaries),
        "num_deep_penetration_ok": sum(bool(r.get("paper_omniretarget_robot_object_deep_penetration_ok", False)) for r in summaries),
        "num_robot_upright_ok": sum(bool(r["E018b_robot_upright_ok"]) for r in summaries),
        "num_robot_fall_detected": sum(bool(r["E018b_robot_fall_detected"]) for r in summaries),
        "num_visual_stability_ok": sum(bool(r["E018b_visual_stability_ok"]) for r in summaries),
        "num_artifact_ok": sum(bool(r["E018b_artifact_ok"]) for r in summaries),
        "num_generalization_pass": sum(bool(r["E018b_generalization_pass"]) for r in summaries),
        "num_canonical_anchor_pass": sum(bool(r["E018b_canonical_anchor_pass"]) for r in summaries),
        "num_gt_anchor_available": sum(bool(r["E018b_gt_anchor_available"]) for r in summaries),
        "num_gt_anchor_pass": sum(bool(r["E018b_gt_anchor_pass"]) for r in summaries),
        "num_gt_gate_pass": sum(bool(r["E018b_gt_gate_pass"]) for r in summaries),
        "mean_paper_Epos_case_m": float(np.mean([float(r["paper_object_Epos_case_m"]) for r in summaries])),
        "mean_paper_Erot_case_deg": float(np.mean([float(r["paper_object_Erot_case_deg"]) for r in summaries])),
        "mean_contact_preservation_5cm_pct": float(np.mean([float(r.get("paper_omniretarget_contact_preservation_5cm_pct", 0.0)) for r in summaries])),
        "mean_deep_penetration_duration_pct": float(np.mean([float(r.get("paper_omniretarget_robot_object_deep_penetration_duration_pct", 0.0)) for r in summaries])),
        "mean_carry_progress_ratio_case": float(np.mean([float(r["paper_carry_progress_ratio_case"]) for r in summaries])),
        "diagnostic_classes": {
            name: sum(r["E018b_diagnostic_class"] == name for r in summaries)
            for name in sorted({str(r["E018b_diagnostic_class"]) for r in summaries})
        },
        "anchor_policies": {
            name: sum(str(r["E018b_anchor_policy"]) == name for r in summaries)
            for name in sorted({str(r["E018b_anchor_policy"]) for r in summaries})
        },
        "anchor_face_sources": {
            name: sum(str(r["E018b_anchor_face_source"]) == name for r in summaries)
            for name in sorted({str(r["E018b_anchor_face_source"]) for r in summaries})
        },
        "anchor_audit_classes": {
            name: sum(str(r["E018b_anchor_audit_class"]) == name for r in summaries)
            for name in sorted({str(r["E018b_anchor_audit_class"]) for r in summaries})
        },
        "max_gt_anchor_dist_m": float(
            np.max([float(r["E018b_gt_anchor_dist_m"]) for r in summaries if bool(r["E018b_gt_anchor_available"])])
        )
        if any(bool(r["E018b_gt_anchor_available"]) for r in summaries)
        else float("nan"),
        "variants": [str(r["variant"]) for r in summaries],
    }
    (RESULTS / "aggregate_summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Wrote {comparison}")
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
