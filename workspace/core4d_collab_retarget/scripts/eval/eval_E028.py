#!/usr/bin/env python3
"""E028 evaluation for D003 Box021 support-proxy dynamic retarget runs."""

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

import eval_E002 as e002  # noqa: E402
import eval_E018b as e018b  # noqa: E402


BASE = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
RESULTS = REPO / "workspace/core4d_collab_retarget/results/E028"
MANIFEST = RESULTS / "manifest.tsv"


def _read_manifest(path: Path = MANIFEST) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return {row["variant"]: row for row in csv.DictReader(f, delimiter="\t")}


def _as_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _point_local(row: dict[str, str]) -> np.ndarray:
    return np.array(
        [
            float(row["support_proxy_point_local_x"]),
            float(row["support_proxy_point_local_y"]),
            float(row["support_proxy_point_local_z"]),
        ],
        dtype=np.float64,
    )


def _canonical_anchor_ok(row: dict[str, str], point: np.ndarray) -> bool:
    face = row.get("anchor_face", "")
    half = np.array(
        [
            float(row.get("object_half_x", "nan")),
            float(row.get("object_half_y", "nan")),
            float(row.get("object_half_z", "nan")),
        ],
        dtype=np.float64,
    )
    if face not in {"+x", "-x", "+y", "-y"} or not np.all(np.isfinite(half)):
        return False
    axis = 0 if face.endswith("x") else 1
    other = 1 - axis
    sign = 1.0 if face.startswith("+") else -1.0
    z_frac = point[2] / half[2] if half[2] > 1e-8 else float("nan")
    return bool(
        abs(point[axis] - sign * half[axis]) <= 1e-5
        and abs(point[other]) <= 1e-5
        and 0.55 <= z_frac <= 0.70
        and point[2] > 0.0
    )


def _scene_anchor_not_oracle(row: dict[str, str]) -> bool:
    scene = BASE / row["derived_task"] / f"{row['scene_name']}.xml"
    if not scene.is_file():
        return False
    text = scene.read_text(encoding="utf-8")
    return bool(
        np.linalg.norm(_point_local(row)) > 0.05
        and "object_target" not in text
        and "support_weld_anchor" in text
        and "e028_support_weld" in text
    )


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
    if variant not in manifest:
        raise ValueError(f"Unknown E028 variant: {variant}")
    e018b.RESULTS = RESULTS
    e018b.MANIFEST = MANIFEST
    variants = e018b.read_manifest()
    summary = e018b.evaluate_variant(variant, variants)
    row = manifest[variant]
    point = _point_local(row)

    _qpos_ref, _ctrl_ref, cfg = e002.load_ref(f"core4d_collab_{variant}", row["derived_task"])
    model, _scene_used = e002.load_scene_model(row["derived_task"])
    freejoint_parity_ok = bool(
        not summary["config_contact_guidance"]
        and str(summary["config_scene_name"]).startswith("scene_e028_jointB_")
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
        and int(model.nu) == 29
    )
    no_direct_wrench = bool(
        float(cfg.partner_force_scale) == 0.0
        and float(cfg.partner_force_spring_kp) == 0.0
        and float(cfg.support_proxy_connector_kp) == 0.0
    )
    canonical_anchor_pass = _canonical_anchor_ok(row, point)
    scene_anchor_pass = _scene_anchor_not_oracle(row)
    config_ok = bool(freejoint_parity_ok and no_direct_wrench and scene_anchor_pass)

    contact_pct = _as_float(summary.get("paper_omniretarget_contact_preservation_5cm_pct"), 0.0)
    deep_pct = _as_float(summary.get("paper_omniretarget_robot_object_deep_penetration_duration_pct"), 100.0)
    max_pen = _as_float(summary.get("paper_omniretarget_robot_object_max_penetration_cm"), 999.0)
    leg_pct = _as_float(summary.get("case_window_sim_leg_box_interference_frames_pct"), 100.0)
    pelvis_z_min = _as_float(summary.get("full_pelvis_z_min_m"))
    first_pelvis_lt45 = _as_float(summary.get("first_pelvis_z_lt_45cm_frame"), -1.0)
    robot_fall = bool(np.isfinite(pelvis_z_min) and (pelvis_z_min < 0.45 or first_pelvis_lt45 >= 0))
    artifact_ok = bool(deep_pct <= 20.0 and max_pen <= 5.0 and contact_pct >= 50.0)
    floor_leg_ok = bool(leg_pct <= 15.0)

    summary["E028_queue"] = row["queue"]
    summary["E028_role"] = row["role"]
    summary["E028_wave"] = row["wave"]
    summary["E028_source_task"] = row["source_task"]
    summary["E028_derived_task"] = row["derived_task"]
    summary["E028_scene_name"] = row["scene_name"]
    summary["E028_point_local"] = point.tolist()
    summary["E028_anchor_policy"] = row["anchor_policy"]
    summary["E028_anchor_face"] = row["anchor_face"]
    summary["E028_anchor_face_source"] = row["anchor_face_source"]
    summary["E028_anchor_face_review"] = row["anchor_face_review"].lower() == "true"
    summary["E028_anchor_face_review_reason"] = row["anchor_face_review_reason"]
    summary["E028_anchor_top_face"] = row["anchor_top_face"]
    summary["E028_anchor_top_face_frac"] = _as_float(row["anchor_top_face_frac"])
    summary["E028_anchor_side_face_frac"] = _as_float(row["anchor_side_face_frac"])
    summary["E028_anchor_z_face_frac"] = _as_float(row["anchor_z_face_frac"])
    summary["E028_anchor_side_margin"] = _as_float(row["anchor_side_margin"])
    summary["E028_contact_points_used"] = int(row["contact_points_used"])
    summary["E028_canonical_z_frac"] = _as_float(row["canonical_z_frac"])
    summary["E028_freejoint_parity_ok"] = freejoint_parity_ok
    summary["E028_no_direct_wrench"] = no_direct_wrench
    summary["E028_anchor_not_com_oracle"] = scene_anchor_pass
    summary["E028_canonical_anchor_pass"] = canonical_anchor_pass
    summary["E028_config_ok"] = config_ok
    summary["E028_robot_fall_detected"] = robot_fall
    summary["E028_robot_upright_ok"] = not robot_fall
    summary["E028_artifact_ok"] = artifact_ok
    summary["E028_floor_leg_ok"] = floor_leg_ok
    summary["E028_object_transport_pass"] = bool(
        summary.get("paper_dynaretarget_object_success", False)
        and summary.get("paper_transport_success", False)
    )
    summary["E028_generalization_pass"] = bool(
        config_ok
        and canonical_anchor_pass
        and summary["E028_object_transport_pass"]
        and not robot_fall
        and artifact_ok
        and floor_leg_ok
        and not summary["E028_anchor_face_review"]
    )

    if not config_ok:
        diag = "config_failed"
    elif not canonical_anchor_pass:
        diag = "canonical_anchor_failed"
    elif summary["E028_anchor_face_review"]:
        diag = "anchor_face_review"
    elif not summary.get("paper_dynaretarget_object_success", False):
        diag = "object_tracking_failed"
    elif not summary.get("paper_transport_success", False):
        diag = "transport_failed"
    elif robot_fall:
        diag = "robot_fall_visual_fail"
    elif contact_pct < 50.0:
        diag = "contact_preservation_gap"
    elif not floor_leg_ok:
        diag = "push_or_leg_shortcut"
    elif not artifact_ok:
        diag = "artifact_failed"
    else:
        diag = "paper_generalization_pass"
    summary["E028_diagnostic_class"] = diag
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
    RESULTS.mkdir(parents=True, exist_ok=True)
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
        raise SystemExit("No E028 variant results found.")

    keys = sorted({key for row in summaries for key in row.keys()})
    comparison = RESULTS / "comparison.csv"
    with comparison.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summaries)

    aggregate = {
        "num_manifest_rows": len(manifest),
        "num_results": len(summaries),
        "num_config_ok": sum(bool(r["E028_config_ok"]) for r in summaries),
        "num_canonical_anchor_pass": sum(bool(r["E028_canonical_anchor_pass"]) for r in summaries),
        "num_anchor_face_review": sum(bool(r["E028_anchor_face_review"]) for r in summaries),
        "num_paper_spider_success": sum(bool(r["paper_spider_object_success"]) for r in summaries),
        "num_paper_dynaretarget_success": sum(bool(r["paper_dynaretarget_object_success"]) for r in summaries),
        "num_transport_success": sum(bool(r["paper_transport_success"]) for r in summaries),
        "num_object_transport_pass": sum(bool(r["E028_object_transport_pass"]) for r in summaries),
        "num_contact_preservation_ok": sum(
            _as_float(r.get("paper_omniretarget_contact_preservation_5cm_pct"), 0.0) >= 50.0
            for r in summaries
        ),
        "num_robot_upright_ok": sum(bool(r["E028_robot_upright_ok"]) for r in summaries),
        "num_robot_fall_detected": sum(bool(r["E028_robot_fall_detected"]) for r in summaries),
        "num_artifact_ok": sum(bool(r["E028_artifact_ok"]) for r in summaries),
        "num_floor_leg_ok": sum(bool(r["E028_floor_leg_ok"]) for r in summaries),
        "num_generalization_pass": sum(bool(r["E028_generalization_pass"]) for r in summaries),
        "mean_paper_Epos_case_m": float(np.mean([_as_float(r["paper_object_Epos_case_m"]) for r in summaries])),
        "mean_paper_Erot_case_deg": float(np.mean([_as_float(r["paper_object_Erot_case_deg"]) for r in summaries])),
        "mean_contact_preservation_5cm_pct": float(
            np.mean([_as_float(r.get("paper_omniretarget_contact_preservation_5cm_pct"), 0.0) for r in summaries])
        ),
        "mean_deep_penetration_duration_pct": float(
            np.mean(
                [
                    _as_float(r.get("paper_omniretarget_robot_object_deep_penetration_duration_pct"), 0.0)
                    for r in summaries
                ]
            )
        ),
        "diagnostic_classes": {
            name: sum(str(r["E028_diagnostic_class"]) == name for r in summaries)
            for name in sorted({str(r["E028_diagnostic_class"]) for r in summaries})
        },
        "variants": [str(r["variant"]) for r in summaries],
    }
    (RESULTS / "aggregate_summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Wrote {comparison}")
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
