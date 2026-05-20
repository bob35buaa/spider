#!/usr/bin/env python3
"""E027 offline case data / retarget quality audit.

This script only aggregates existing evidence from E020, E026, Holosoma, and
E018b. It does not run MuJoCo or change any reward/core code.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import e027_common as C


def _evidence_label(
    *,
    case: str,
    root: str,
    mask_overclaim: float,
    mask_mismatch: float,
    ref_leg: float,
    holosoma_available: bool,
) -> tuple[str, bool, bool, bool, bool]:
    raw_pass = root != "raw_data"
    annotation_pass = not (mask_overclaim > 50.0 or mask_mismatch > 50.0 or root == "contact_mask")
    retarget_pass = not (root == "retarget_kinematic" or ref_leg > 45.0 or not holosoma_available)
    visual_pass = root != "raw_data"

    failed = sum(not x for x in [raw_pass, annotation_pass, retarget_pass, visual_pass])
    if root == "raw_data" and not holosoma_available and failed >= 2:
        label = "discard_from_success_denominator"
    elif root == "raw_data":
        label = "raw_data_questionable"
    elif not retarget_pass:
        label = "retarget_questionable"
    elif not annotation_pass:
        label = "usable_with_caveat"
    else:
        label = "usable_algorithmic_failure"

    # Keep the known E026 P0 exclusions outside P0 unless E027 later says
    # otherwise. P1 still reports every case.
    discard_from_p0 = case not in C.P0_CASES or label == "discard_from_success_denominator"
    discard_from_denominator = label == "discard_from_success_denominator"
    return label, raw_pass, annotation_pass, retarget_pass, visual_pass, discard_from_p0, discard_from_denominator


def _join_evidence(parts: list[str]) -> str:
    return "; ".join(part for part in parts if part)


def build_rows() -> list[dict[str, Any]]:
    tables = C.load_e020_tables()
    manifest_by_case = C.load_manifest_by_case()
    best_by_case = C.load_best_selection()
    best_metrics_by_case = C.best_dynamic_metrics_by_case()
    holosoma_by_case = C.load_holosoma_by_case()
    rows: list[dict[str, Any]] = []

    for case in C.CASES_13:
        root_row = tables["root"].get(case, {})
        anchor_row = tables["anchor"].get(case, {})
        mask_row = tables["mask"].get(case, {})
        ref_row = tables["ref"].get(case, {})
        sim_row = tables["sim"].get(case, {})
        manifest = manifest_by_case.get(case, {})
        best = best_by_case.get(case, {})
        best_metrics = best_metrics_by_case.get(case, {})
        kin = holosoma_by_case.get(case, {})
        mask_summary = C.mask_summary_for_manifest(manifest)

        root = root_row.get("root_cause", "")
        mask_overclaim = C.as_float(mask_row.get("mask_overclaim_pct", root_row.get("mask_overclaim_pct")))
        mask_mismatch = C.as_float(mask_row.get("mask_mismatch_pct", mask_overclaim))
        ref_leg = C.as_float(ref_row.get("ref_leg_interference_pct", root_row.get("ref_leg_interference_pct")))
        holosoma_available = bool(kin)

        (
            quality_label,
            raw_pass,
            annotation_pass,
            retarget_pass,
            visual_pass,
            discard_from_p0,
            discard_from_denominator,
        ) = _evidence_label(
            case=case,
            root=root,
            mask_overclaim=mask_overclaim,
            mask_mismatch=mask_mismatch,
            ref_leg=ref_leg,
            holosoma_available=holosoma_available,
        )

        failed_classes = sum(
            not x for x in [raw_pass, annotation_pass, retarget_pass, visual_pass]
        )
        primary = _join_evidence(
            [
                f"E020 root_cause={root}" if root else "",
                f"mask_overclaim={mask_overclaim:.1f}%" if mask_overclaim == mask_overclaim else "",
                f"ref_leg_interference={ref_leg:.1f}%" if ref_leg == ref_leg else "",
                "holosoma_missing" if not holosoma_available else "",
            ]
        )
        secondary = _join_evidence(
            [
                root_row.get("evidence", ""),
                f"kin_contact28={C.fmt_pct(kin.get('paper_omniretarget_contact_preservation_local_case_pct'))}"
                if kin
                else "",
                f"best_contact5={C.fmt_pct(best.get('contact_proxy_pct'))}",
                f"best_deep_pen={C.fmt_pct(best.get('deep_pen_pct'))}",
            ]
        )
        counter = _join_evidence(
            [
                "object tracking ok" if C.as_float(best.get("obj_pos_cm")) <= 10.0 else "",
                "no fall" if str(best.get("fall")).lower() == "false" else "",
                "strict pass guard" if str(best.get("strict_success")).lower() == "true" else "",
                "holosoma available" if holosoma_available else "",
            ]
        )
        if not counter:
            counter = "none"

        row: dict[str, Any] = {
            "case": case,
            "source_task": manifest.get("source_task", ""),
            "derived_task": manifest.get("derived_task", ""),
            "person_idx": manifest.get("person_idx", ""),
            "variant_e018b": f"E018b_{case}_canonical_t02",
            "selected_best_variant": best.get("selected_variant", ""),
            "best_method": best.get("selected_method", ""),
            "quality_label": quality_label,
            "discard_from_p0": discard_from_p0,
            "discard_from_success_denominator": discard_from_denominator,
            "recommended_owner": root_row.get("recommended_owner", ""),
            "next_experiment_hint": root_row.get("next_experiment", ""),
            "raw_motion_quality_pass": raw_pass,
            "object_contact_annotation_pass": annotation_pass,
            "retarget_quality_pass": retarget_pass,
            "visual_audit_pass": visual_pass,
            "num_failed_evidence_classes": failed_classes,
            "root_cause_E020": root,
            "S1_anchor_vs_raw_pass": root_row.get("S1_anchor_vs_raw_pass", ""),
            "S2_ref_physics_pass": root_row.get("S2_ref_physics_pass", ""),
            "S3_mask_vs_raw_pass": root_row.get("S3_mask_vs_raw_pass", ""),
            "S4_sim_ref_alignment_pass": root_row.get("S4_sim_ref_alignment_pass", ""),
            "raw_seq_dir": mask_summary.get("seq_dir", ""),
            "raw_mesh_path": mask_summary.get("mesh", ""),
            "trim_start": mask_summary.get("trim_start", ""),
            "spider_frames": mask_summary.get("spider_frames", ""),
            "raw_selected_any_contact_pct": mask_row.get("raw_selected_any_contact_pct", ""),
            "raw_left_pct": mask_row.get("raw_left_pct", ""),
            "raw_right_pct": mask_row.get("raw_right_pct", ""),
            "current_ref_any_contact_pct": mask_row.get("current_ref_any_contact_pct", ""),
            "mask_mismatch_pct": mask_mismatch,
            "mask_overclaim_pct": mask_overclaim,
            "mask_underclaim_pct": mask_row.get("mask_underclaim_pct", ""),
            "partner_anchor_dist_m": anchor_row.get("partner_anchor_dist_m", root_row.get("partner_anchor_dist_m", "")),
            "anchor_face": anchor_row.get("anchor_face", ""),
            "raw_partner_top_face": anchor_row.get("raw_partner_top_face", ""),
            "raw_partner_top_face_frac": anchor_row.get("raw_partner_top_face_frac", ""),
            "ref_leg_interference_pct": ref_leg,
            "ref_hand_contact_pct": ref_row.get("ref_hand_contact_pct", ""),
            "ref_pelvis_z_min_m": ref_row.get("ref_pelvis_z_min_m", ""),
            "holosoma_available": holosoma_available,
            "holosoma_cost": kin.get("cost", ""),
            "holosoma_source_npz": kin.get("source_npz", ""),
            "holosoma_contact28_pct": kin.get("paper_omniretarget_contact_preservation_local_case_pct", ""),
            "holosoma_contact28_demo_frames": kin.get("paper_omniretarget_contact_preservation_local_case_demo_frames", ""),
            "holosoma_smoothness": kin.get("paper_dynaretarget_smoothness", ""),
            "holosoma_mj_pen_duration_pct": kin.get("paper_omniretarget_mj_penetration_duration_pct", ""),
            "best_contact5_pct": best.get("contact_proxy_pct", best_metrics.get("contact_proxy_pct", "")),
            "best_deep_pen_pct": best.get("deep_pen_pct", best_metrics.get("deep_pen_pct", "")),
            "best_max_pen_cm": best.get("max_pen_cm", best_metrics.get("max_pen_cm", "")),
            "best_obj_pos_cm": best.get("obj_pos_cm", best_metrics.get("obj_pos_cm", "")),
            "best_fall": best.get("fall", best_metrics.get("fall", "")),
            "best_strict_success": best.get("strict_success", best_metrics.get("strict_success", "")),
            "visual_evidence_path": str(C.panel_path(case)),
            "online_video_path": C.video_path(case),
            "primary_evidence": primary,
            "secondary_evidence": secondary,
            "counter_evidence": counter,
            "decision_rationale": (
                f"{quality_label}: failed_evidence_classes={failed_classes}; "
                f"discard_from_p0={discard_from_p0}; "
                f"discard_from_success_denominator={discard_from_denominator}"
            ),
        }
        rows.append(row)
    return rows


def write_markdown(rows: list[dict[str, Any]]) -> None:
    lines = [
        "# E027 Data / Retarget Quality Audit",
        "",
        "| Case | Label | Drop P0 | Drop Denom | Failed Evidence | Primary Evidence | Counter Evidence |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {case} | `{quality_label}` | {discard_from_p0} | {discard_from_success_denominator} | "
            "{num_failed_evidence_classes} | {primary_evidence} | {counter_evidence} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Label Counts",
            "",
        ]
    )
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["quality_label"])] = counts.get(str(row["quality_label"]), 0) + 1
    for label, count in sorted(counts.items()):
        lines.append(f"- `{label}`: {count}")
    lines.append("")
    lines.append("All cases remain in the P1/caveat table. `discard_from_success_denominator` only affects the main optimization denominator.")
    (C.DATA_QUALITY / "case_quality_audit.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="audit all 13 cases")
    args = ap.parse_args()
    if not args.all:
        ap.error("E027 first phase expects --all")

    rows = build_rows()
    C.write_rows(C.DATA_QUALITY / "case_quality_audit.csv", rows)
    per_case_dir = C.DATA_QUALITY / "per_case"
    for row in rows:
        C.write_rows(per_case_dir / f"{row['case']}.csv", [row])
    write_markdown(rows)
    C.write_json(
        C.DATA_QUALITY / "case_quality_audit_summary.json",
        {
            "num_cases": len(rows),
            "labels": {label: sum(r["quality_label"] == label for r in rows) for label in sorted({r["quality_label"] for r in rows})},
            "discard_from_success_denominator": [
                r["case"] for r in rows if r["discard_from_success_denominator"]
            ],
        },
    )
    print(f"[E027] wrote {len(rows)} quality rows to {C.DATA_QUALITY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
